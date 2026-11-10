import time
import numpy as np
from itertools import combinations

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0

    # 1. Generate candidate clusters of 11 hexagons on a hexagonal lattice.
    # We use axial coordinates (q, r).
    # We generate all connected subsets of size 11 from a small bounding box.
    # To ensure diversity, we include various shapes (bars, triangles, blobs).
    
    candidates = set()
    
    # Helper to convert axial to cartesian
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])

    # Generate all connected sets of size 11 containing (0,0)
    # We restrict the search to a small bounding box to keep it fast and compact.
    # Max q, r range: -3 to 3 is sufficient for 11 hexagons.
    
    def generate_connected_sets():
        sets = set()
        
        # BFS/DFS growth from (0,0)
        # State: frozenset of (q,r)
        # To avoid explosion, we only grow if the set is "compact" (bounding box <= 4x4 or similar)
        
        start = frozenset([(0,0)])
        
        def grow(current_set):
            if len(current_set) == 11:
                # Canonicalize: translate so min q=0, min r=0? 
                # Actually, for rotation optimization, translation doesn't matter for side length.
                # We just need the relative positions.
                # We store the set of relative coordinates.
                # To deduplicate, we can normalize by sorting and taking the set.
                # Since we grow from (0,0), different paths might lead to the same set.
                # Using a set of frozensets handles this.
                sets.add(current_set)
                return
            
            if time.time() > deadline - 10.0:
                return
                
            # Find neighbors
            neighbors = set()
            for q, r in current_set:
                for dq, dr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                    nq, nr = q+dq, r+dr
                    if (nq, nr) not in current_set:
                        neighbors.add((nq, nr))
            
            for nb in neighbors:
                new_set = frozenset(current_set | {nb})
                
                # Pruning: Check bounding box to ensure compactness
                qs = [q for q,r in new_set]
                rs = [r for q,r in new_set]
                # If the shape is too elongated, it will likely have a larger enclosing hexagon.
                # A "perfect" hexagon of side 2 has 7 hexes. Side 3 has 19.
                # 11 hexes should fit in a roughly 3x3 area.
                if max(qs) - min(qs) > 4 or max(rs) - min(rs) > 4:
                    continue
                
                grow(new_set)
        
        grow(start)
        return list(sets)

    cluster_sets = generate_connected_sets()
    
    # Also add some specific "known good" shapes if generation is slow or empty
    # Shape 1: 3-5-3 grid (part of a hexagon of side 2, plus 4 more)
    # Actually, let's just rely on the generator. It's fast for N=11.
    
    if not cluster_sets:
        # Fallback: A simple line or small cluster
        # This shouldn't happen, but safe to have
        cluster_sets = [frozenset([(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (10,0)])]

    best_side = np.inf
    best_inner = None
    best_outer = None

    # 2. For each cluster, find the optimal rotation angle to minimize enclosing hexagon side.
    
    # Precompute cartesian centers for each cluster
    # Note: The cluster is rigid. We rotate the whole cluster.
    
    for cluster in cluster_sets:
        if time.time() > deadline - 5.0:
            break
            
        # Get cartesian coordinates of centers
        centers = np.array([axial_to_cartesian(q, r) for q, r in cluster])
        
        # The "shape" of the cluster is defined by the vertices of all hexagons.
        # For a unit hexagon at (cx, cy) with angle 0 (flat top/bottom or pointy top?),
        # The task defines vertices as:
        # (cx + s*cos(a + 60i), cy + s*sin(a + 60i))
        # If a=0, vertices are at angles 0, 60, 120, 180, 240, 300.
        # This is a "pointy-top" hexagon? No, 0 deg is on x-axis.
        # cos(0)=1, sin(0)=0. So vertex at (1,0).
        # cos(60)=0.5, sin(60)=0.866.
        # This is a hexagon with vertices at 0, 60, ... degrees.
        
        # We assume all inner hexagons have angle 0 relative to the cluster frame.
        # The cluster is then rotated by theta.
        
        # Vertices of a single unit hexagon at origin, angle 0:
        angles = np.deg2rad(np.arange(0, 360, 60))
        base_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # All vertices of the cluster (unrotated):
        # For each center, add base_vertices
        # Total 11 * 6 = 66 vertices.
        all_verts = []
        for c in centers:
            all_verts.append(c[:, np.newaxis] + base_vertices)
        all_verts = np.vstack(all_verts)
        
        # We want to find theta such that the enclosing regular hexagon side length is minimized.
        # The enclosing hexagon has angle phi.
        # Due to symmetry, we can fix the enclosing hexagon angle to 0 and rotate the cluster by theta.
        # Or fix cluster angle 0 and rotate enclosing hexagon.
        # Let's fix enclosing hexagon normals to 0, 60, ..., 300.
        # And rotate the cluster by theta.
        
        # Enclosing hexagon side s(theta):
        # The width of the cluster in direction n_k (normal of enclosing hex edge k) is W_k(theta).
        # The side length s must satisfy: s * sqrt(3) >= W_k(theta) for all k.
        # So s(theta) = max_k(W_k(theta)) / sqrt(3).
        
        # W_k(theta) = max_p (p . n_k) - min_p (p . n_k)
        # where p are the rotated vertices.
        # p_rot = R(theta) p
        # p_rot . n_k = p . (R(-theta) n_k)
        # So we can precompute p, and for each theta, rotate the normals n_k by -theta.
        
        # Normals of enclosing hexagon (angle 0):
        # The edges are perpendicular to the normals.
        # If vertices are at 0, 60..., the edge between 0 and 60 is perpendicular to 30 deg.
        # Wait, the task defines vertices.
        # "A hexagon (cx, cy, s, a) has vertices (cx + s*cos(a + 60i deg), ...)"
        # The edge normals are at angles a + 30 + 60i?
        # Let's check.
        # Vertex 0: angle a. Vertex 1: angle a+60.
        # Edge vector: V1 - V0.
        # Normal to this edge is rotated by 90 deg.
        # The apothem direction is a + 30.
        # So the 6 normals are at angles a + 30, a + 90, ...
        
        # Let's define the normals for the OUTER hexagon.
        # If outer angle is 0, vertices at 0, 60...
        # Normals at 30, 90, 150, 210, 270, 330.
        
        outer_normals_base = np.deg2rad(30 + 60 * np.arange(6))
        outer_normals = np.column_stack((np.cos(outer_normals_base), np.sin(outer_normals_base)))
        
        def get_side(theta_rad):
            # Rotate normals by -theta to get effective normals in cluster frame
            # Or rotate cluster by theta and project on fixed normals.
            # Rotating 66 vertices is cheap.
            c = np.cos(theta_rad)
            s = np.sin(theta_rad)
            R = np.array([[c, -s], [s, c]])
            
            verts_rot = all_verts @ R.T
            
            # Project on fixed outer normals (0, 60... wait, normals are 30, 90...)
            # The outer hexagon angle will be 0. So its normals are fixed at 30, 90...
            # But we are rotating the cluster.
            # The condition is: Cluster inside Outer Hex (angle 0).
            # So we project Cluster Rotated by theta onto Fixed Normals.
            
            proj = verts_rot @ outer_normals.T
            widths = proj.max(axis=0) - proj.min(axis=0)
            return widths.max() / np.sqrt(3.0)
        
        # Minimize get_side(theta) over theta in [0, pi/3) due to 6-fold symmetry of outer hex?
        # The cluster has no symmetry, so we need to search [0, 2*pi).
        # But get_side(theta) is periodic with 2*pi.
        # Also, rotating the outer hexagon by 30 degrees swaps the "flat" and "pointy" orientation?
        # No, a regular hexagon is symmetric under 60 degree rotation.
        # So theta in [0, 60) is sufficient?
        # If we rotate cluster by 60, the projection widths are the same set, just permuted?
        # Yes, because the set of normals {30, 90, ...} is invariant under 60 deg rotation?
        # Normals: 30, 90, 150, 210, 270, 330.
        # Add 60: 90, 150, ... 330, 30. Same set.
        # So yes, theta in [0, 60) is sufficient.
        
        # Grid search + refinement
        thetas = np.linspace(0, np.pi/3, 181) # 1 degree resolution
        sides = np.array([get_side(t) for t in thetas])
        
        min_idx = np.argmin(sides)
        theta_min = thetas[min_idx]
        
        # Refine around theta_min
        # Golden section or just fine grid
        fine_thetas = np.linspace(theta_min - np.deg2rad(2), theta_min + np.deg2rad(2), 100)
        fine_sides = np.array([get_side(t) for t in fine_thetas])
        fine_min_idx = np.argmin(fine_sides)
        theta_opt = fine_thetas[fine_min_idx]
        side_opt = fine_sides[fine_min_idx]
        
        if side_opt < best_side:
            best_side = side_opt
            
            # Construct the result
            # Inner hexagons: centers rotated by theta_opt, angle = theta_opt (in degrees)
            # Wait, the inner hexagons' angle parameter `a` in the return value.
            # The vertices are defined by a.
            # If we rotate the cluster by theta, the local orientation of each hexagon also rotates by theta.
            # So angle_deg = np.rad2deg(theta_opt).
            
            c = np.cos(theta_opt)
            s = np.sin(theta_opt)
            R = np.array([[c, -s], [s, c]])
            
            centers_rot = centers @ R.T
            
            # Outer hexagon:
            # We assumed outer angle 0.
            # Center: The minimal enclosing hexagon center is the midpoint of the projections?
            # For a regular hexagon, the center that minimizes the max distance (or width)
            # when the shape is centered is the centroid?
            # Actually, for a fixed rotation, the minimal side length is determined by the max width.
            # The center should be the midpoint of the min and max projections for EACH normal?
            # This is only possible if the widths are consistent.
            # In general, the center is the Chebyshev center.
            # For a regular hexagon, if we fix the rotation, the minimal side is determined by the 3 pairs of parallel support lines.
            # The center is the intersection of the midlines.
            # Since it's a regular hexagon, the midlines are concurrent at the center.
            # So we can compute the center as the average of the midpoints of the projections.
            
            proj = centers_rot @ outer_normals.T # Using centers is not enough, need vertices!
            # Wait, the constraint is on VERTICES.
            # The enclosing hexagon must contain all vertices.
            # So we must use vertices for the center calculation too.
            
            verts_rot_final = all_verts @ R.T
            proj_final = verts_rot_final @ outer_normals.T
            min_proj = proj_final.min(axis=0)
            max_proj = proj_final.max(axis=0)
            mid_proj = (min_proj + max_proj) / 2.0
            
            # Solve for center (cx, cy) such that:
            # [cx, cy] . n_k = mid_proj_k for all k
            # This is an overdetermined system. Use least squares.
            # outer_normals is (6, 2).
            # We want [cx, cy] such that outer_normals.T @ [cx, cy] = mid_proj
            # So [cx, cy] = lstsq(outer_normals.T, mid_proj)
            
            cx, cy = np.linalg.lstsq(outer_normals.T, mid_proj, rcond=None)[0]
            
            # Shift inner centers by -center
            centers_final = centers_rot - np.array([cx, cy])
            
            inner = np.column_stack((centers_final[:, 0], centers_final[:, 1], np.full(11, np.rad2deg(theta_opt))))
            outer = np.array([0.0, 0.0, 0.0]) # Outer angle is 0 by our assumption
            
            best_inner = inner
            best_outer = outer

    if best_inner is None:
        # Fallback
        inner = np.zeros((11, 3))
        outer = np.zeros(3)
        return inner, outer, 10.0

    return best_inner, best_outer, best_side

# EVOLVE-BLOCK-END
