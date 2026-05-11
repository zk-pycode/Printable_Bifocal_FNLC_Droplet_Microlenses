"""
src/meshing.py


Mesh pipeline stages
--------------------
1. gmsh base disk
       Rank 0 meshes the circular droplet footprint (2D triangles) using gmsh.

2. Separate inner vs. perimeter nodes
       Perimeter nodes (r ≈ R_base) only appear in the top (dome) layer; inner
       nodes are extruded through all n_layers.  This avoids stacked near-
       coincident rings at the contact line.

3. Extrude to spherical-cap 3D nodes
       Each inner node is lifted to z = α·z_dome(x, y) for α ∈ [0, 1]
       (fraction of cap height).  Perimeter nodes sit exactly on the dome.

4. Split prisms into tetrahedra
       Each extruded prism is split into 3 tetrahedra using a sorted staircase
       decomposition that guarantees consistent face diagonals across adjacent
       cells.  FFCx requires tetrahedra; prism quadrature is not supported.

5. Distribute via DOLFINx create_mesh
       Rank 0 supplies all cell/node data; other ranks pass empty arrays.
       DOLFINx's partitioner distributes the mesh across all MPI processes.

6. Create meshtags
       Bottom facets (z = 0)  →  ID_BOTTOM = 1
       Dome facets (z = z_dome(x,y))  →  ID_DOME = 2
       All volume cells  →  ID_CAP = 10

Classes
-------
MeshHandler
    Orchestrates the full mesh creation pipeline.
FunctionSpaces
    Creates the 5-component CG1 Q-tensor function space on the mesh.
"""

import numpy as np

from mpi4py import MPI
import gmsh
from dolfinx import fem as _fem
from dolfinx.mesh import create_mesh as _dolfinx_create_mesh, locate_entities_boundary, meshtags
import basix.ufl
from basix.ufl import element


# ---------------------------------------------------------------------------
# Mesh handler
# ---------------------------------------------------------------------------

class MeshHandler:
    """
    Creates the spherical-cap tetrahedral mesh and associated meshtags.

    Attributes
    ----------
    mesh : dolfinx.mesh.Mesh
        The distributed DOLFINx mesh.
    boundary_tags : MeshTags
        Facet tags: ID_BOTTOM (base, z=0) and ID_DOME (dome surface).
    cell_tags : MeshTags
        Volume tags: all cells tagged ID_CAP.
    ID_BOTTOM, ID_DOME, ID_CAP : int
        Integer tag identifiers used in Measure("ds") and Measure("dx").
    """

    def __init__(self, params):
        if MPI.COMM_WORLD.rank == 0:
            print("\nMeshHandler: Initializing...")
        self.params        = params
        self.mesh          = None
        self.boundary_tags = None
        self.cell_tags     = None
        self.ID_BOTTOM     = 1
        self.ID_DOME       = 2
        self.ID_CAP        = 10

    def create_mesh(self):
        """
        Execute the full mesh creation pipeline (stages 1–6 described above).

        Only rank 0 runs gmsh and builds the node/cell arrays; all other ranks
        wait with empty arrays until DOLFINx distributes the mesh.
        """
        comm   = MPI.COMM_WORLD
        p      = self.params
        R_base   = p.base_radius
        theta    = p.contact_angle_rad
        R_sphere = R_base / np.sin(theta)
        h_cap    = R_sphere * (1.0 - np.cos(theta))
        z_center = -R_sphere * np.cos(theta)

        n_layers  = int(p.min_bulk_layers)
        mesh_size = p.mesh_size

        if comm.rank == 0:
            print("MeshHandler: Creating layered tetrahedral mesh...")
            print(f"  Cap height:      {h_cap*1e6:.2f} um")
            print(f"  Mesh size:       {mesh_size*1e6:.2f} um")
            print(f"  Vertical layers: {n_layers}")

        # -----------------------------------------------------------------------
        # STAGE 1 - gmsh base disk (rank 0 only)
        # -----------------------------------------------------------------------
        # create_mesh expects cell/coordinate data ONLY on rank 0; other ranks
        # pass empty arrays.  Providing data on every rank causes DOLFINx to
        # treat each rank's copy as unique cells and creates an N× duplicated mesh.
        if comm.rank == 0:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("base_disk")

            disk = gmsh.model.occ.addDisk(0, 0, 0, R_base, R_base)
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [disk], 1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.model.mesh.generate(2)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            base_nodes = node_coords.reshape(-1, 3)[:, :2]   # (N, 2) xy only

            base_triangles = None
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
            for i, etype in enumerate(elem_types):
                if etype == 2:   # 3-node triangle
                    base_triangles = np.array(elem_node_tags[i]).reshape(-1, 3) - 1

            gmsh.finalize()

            n_base_nodes = len(base_nodes)

            # -----------------------------------------------------------------------
            # STAGE 2 - Separate inner vs. perimeter nodes
            # -----------------------------------------------------------------------
            # Perimeter nodes (r ≈ R_base) only appear in the top (dome) layer.
            # All lower layers contain inner nodes only, so the contact-line ring
            # is not stacked at multiple heights.
            r_vals    = np.sqrt(base_nodes[:, 0]**2 + base_nodes[:, 1]**2)
            perim_mask = np.isclose(r_vals, R_base, rtol=1e-4)
            inner_mask = ~perim_mask

            inner_idx = np.where(inner_mask)[0]
            perim_idx  = np.where(perim_mask)[0]
            n_inner    = len(inner_idx)
            n_perim    = len(perim_idx)

            # Local index maps: original base-node index → within-layer index
            inner_local = np.full(n_base_nodes, -1, dtype=np.int64)
            inner_local[inner_idx] = np.arange(n_inner, dtype=np.int64)

            perim_local = np.full(n_base_nodes, -1, dtype=np.int64)
            perim_local[perim_idx] = np.arange(n_perim, dtype=np.int64)

            # -----------------------------------------------------------------------
            # STAGE 3 - Build 3D node positions on the spherical cap
            # -----------------------------------------------------------------------
            # 3D node layout:
            #   layers 0 … n_layers  →  n_inner nodes each  (inner only)
            #   top-layer dome rim   →  n_perim nodes appended at the end
            #   Total = (n_layers+1)*n_inner + n_perim
            #
            # A small minimum dome height prevents z = 0 coincidences that would
            # confuse the boundary-tag locator.
            min_dome_height = mesh_size * 1e-3

            def z_dome(x_, y_):
                r_ = np.sqrt(x_**2 + y_**2)
                return max(z_center + np.sqrt(max(0.0, R_sphere**2 - r_**2)),
                           min_dome_height)

            points_3d = []
            for layer in range(n_layers + 1):
                alpha = layer / n_layers
                for orig in inner_idx:
                    x, y = base_nodes[orig]
                    points_3d.append([x, y, alpha * z_dome(x, y)])
            for orig in perim_idx:
                x, y = base_nodes[orig]
                points_3d.append([x, y, z_dome(x, y)])   # alpha = 1

            points_3d = np.array(points_3d, dtype=np.float64)

            def node_inner(orig, layer):
                return int(layer * n_inner + inner_local[orig])

            def node_perim(orig):
                return int((n_layers + 1) * n_inner + perim_local[orig])

            # -----------------------------------------------------------------------
            # STAGE 4 - Split prisms into tetrahedra (sorted staircase decomposition)
            # -----------------------------------------------------------------------
            # Three triangle types arise from the base mesh:
            #   n_p == 0: all-inner vertices → standard sorted-staircase extrusion
            #             through every layer (3 tets per layer per triangle).
            #   n_p == 1: one perimeter vertex → pentahedron (5 nodes) at the last
            #             layer only, split into 2 tets.
            #   n_p == 2: two perimeter vertices → tetrahedron (4 nodes) at the last
            #             layer only; already a single tet.
            #   n_p == 3: degenerate edge triangle on the rim, skipped.
            tets      = []
            n_flipped = 0

            def add_tet(tet):
                # Ensure positive volume (right-hand orientation) by checking the
                # signed volume and swapping two vertices if negative.
                nonlocal n_flipped
                v0, v1, v2, v3 = points_3d[tet]
                vol = np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))
                if vol < 0:
                    tet[0], tet[1] = tet[1], tet[0]
                    n_flipped += 1
                tets.append(tet)

            for tri in base_triangles:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                n_p = int(perim_mask[a]) + int(perim_mask[b]) + int(perim_mask[c])

                if n_p == 0:
                    # All inner: sorted staircase through every layer
                    for layer in range(n_layers):
                        bot = np.array([node_inner(a, layer),
                                        node_inner(b, layer),
                                        node_inner(c, layer)])
                        top = np.array([node_inner(a, layer + 1),
                                        node_inner(b, layer + 1),
                                        node_inner(c, layer + 1)])
                        order = np.argsort(bot)
                        ba, bb, bc = bot[order]
                        ta, tb, tc = top[order]
                        for tet in [[ba, bb, bc, ta],
                                    [bb, bc, ta, tb],
                                    [bc, ta, tb, tc]]:
                            add_tet(tet)

                elif n_p == 1:
                    # 1 perimeter node: pentahedron → 2 tets (last layer only)
                    if   perim_mask[a]: iv1, iv2, pv = b, c, a
                    elif perim_mask[b]: iv1, iv2, pv = a, c, b
                    else:               iv1, iv2, pv = a, b, c
                    k  = n_layers - 1
                    A0 = node_inner(iv1, k);       B0 = node_inner(iv2, k)
                    A1 = node_inner(iv1, n_layers); B1 = node_inner(iv2, n_layers)
                    C1 = node_perim(pv)
                    add_tet([A0, B0, A1, C1])
                    add_tet([B0, B1, A1, C1])

                elif n_p == 2:
                    # 2 perimeter nodes: 1 tet (last layer only)
                    if   inner_mask[a]: iv, pv1, pv2 = a, b, c
                    elif inner_mask[b]: iv, pv1, pv2 = b, a, c
                    else:               iv, pv1, pv2 = c, a, b
                    k  = n_layers - 1
                    A0 = node_inner(iv, k)
                    A1 = node_inner(iv, n_layers)
                    B1 = node_perim(pv1);  C1 = node_perim(pv2)
                    add_tet([A0, A1, B1, C1])
                # n_p == 3: degenerate rim triangle, skip

            tets = np.array(tets, dtype=np.int64)

            if n_flipped > 0:
                print(f"  Orientation fix: flipped {n_flipped} inverted tets")
            print(f"  Inner nodes: {n_inner}, Dome-rim nodes: {n_perim} "
                  f"(top layer only)")
            print(f"  Total 3D nodes: {len(points_3d)}, Total tets: {len(tets)}")

        else:
            # All other ranks: empty arrays - DOLFINx distributes from rank 0
            tets      = np.empty((0, 4), dtype=np.int64)
            points_3d = np.empty((0, 3), dtype=np.float64)

        # -----------------------------------------------------------------------
        # STAGE 5 - Distribute mesh (all ranks)
        # -----------------------------------------------------------------------
        domain    = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
        self.mesh = _dolfinx_create_mesh(comm, tets, points_3d, domain)
        self.mesh.name = "Cap_Mesh"

        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        self.mesh.topology.create_connectivity(tdim, tdim)

        # -----------------------------------------------------------------------
        # STAGE 6 - Create meshtags for volume cells and boundary facets
        # -----------------------------------------------------------------------
        # Cell tags: every tetrahedral cell is part of the droplet volume (ID_CAP)
        num_cells   = self.mesh.topology.index_map(tdim).size_local
        cell_indices = np.arange(num_cells, dtype=np.int32)
        cell_values  = np.full(num_cells, self.ID_CAP, dtype=np.int32)
        self.cell_tags = meshtags(self.mesh, tdim, cell_indices, cell_values)
        self.cell_tags.name = "Cell Markers"

        # Boundary facet markers
        _R_sphere        = R_sphere
        _z_center        = z_center
        _min_dome_height = mesh_size * 1e-3

        def bottom_marker(x):
            return np.isclose(x[2], 0.0, atol=1e-10)

        def dome_marker(x):
            # Accept facets whose z matches the spherical-cap height within 5%.
            # This tolerance rejects interior side-wall facets that sit below z_dome.
            r_xy       = np.sqrt(x[0]**2 + x[1]**2)
            z_expected = _z_center + np.sqrt(np.maximum(0.0, _R_sphere**2 - r_xy**2))
            z_expected = np.maximum(z_expected, _min_dome_height)
            return np.isclose(x[2], z_expected, rtol=0.05, atol=1e-12)

        bottom_facets = locate_entities_boundary(self.mesh, fdim, bottom_marker)
        dome_facets   = locate_entities_boundary(self.mesh, fdim, dome_marker)

        facet_indices = np.concatenate([bottom_facets, dome_facets]).astype(np.int32)
        facet_values  = np.concatenate([
            np.full(len(bottom_facets), self.ID_BOTTOM, dtype=np.int32),
            np.full(len(dome_facets),   self.ID_DOME,   dtype=np.int32)
        ])

        sort_order    = np.argsort(facet_indices)
        facet_indices = facet_indices[sort_order]
        facet_values  = facet_values[sort_order]

        self.boundary_tags = meshtags(self.mesh, fdim, facet_indices, facet_values)
        self.boundary_tags.name = "Facet Markers"

        if comm.rank == 0:
            n_cells = self.mesh.topology.index_map(tdim).size_global
            print(f"  Mesh created: {n_cells} tetrahedral cells")
            print(f"  Bottom facets: {len(bottom_facets)}, Dome facets: {len(dome_facets)}")


# ---------------------------------------------------------------------------
# Function space setup
# ---------------------------------------------------------------------------

class FunctionSpaces:
    """
    Creates the finite element function space for the Q-tensor field.

    The Q-tensor is stored as a vector of 5 independent components at each
    CG1 (P1) node, giving a (5,)-shaped Lagrange element.

    Parameters
    ----------
    mesh_handler : MeshHandler
        Fully initialised mesh handler (mesh must already be created).

    Attributes
    ----------
    W : fem.FunctionSpace
        5-component CG1 Q-tensor space (also stored as V_vector for compatibility).
    mesh : dolfinx.mesh.Mesh
        Reference to the underlying mesh.
    """

    def __init__(self, mesh_handler):
        if MPI.COMM_WORLD.rank == 0:
            print("FunctionSpaces: Initializing...")
        self.mesh_handler = mesh_handler
        self.mesh         = mesh_handler.mesh
        self.W            = None
        self.V_vector     = None

    def setup_spaces(self):
        """
        Construct and store the 5-component CG1 Q-tensor function space.
        """
        if MPI.COMM_WORLD.rank == 0:
            print("FunctionSpaces: Creating Q-tensor function space (5 components)...")
        msh       = self.mesh
        Q_el      = element("Lagrange", msh.basix_cell(), 1, shape=(5,))
        self.W    = _fem.functionspace(msh, Q_el)
        self.V_vector = self.W
        if MPI.COMM_WORLD.rank == 0:
            print(f"  Q-tensor space (W) DOFs: {self.W.dofmap.index_map.size_global}")
