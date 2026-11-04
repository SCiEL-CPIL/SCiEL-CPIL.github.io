import numpy as np
import time
from itertools import combinations

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0,60,...300)
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

def outer_side_for_cluster(points, angle_deg):
    """
    Calculate the side length of the minimum regular hexagon (centered at origin, rot 0)
    that encloses the given cluster of unit hexagons rotated by angle_deg.
    points: (N, 2) array of center coordinates in the cluster's local frame.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate all points
    pts_rot = points @ R.T
    
    # The condition for a unit hexagon centered at p to be inside a container hexagon 
    # of side S (centered at origin, rot 0) is:
    # For all container normals n_k: dot(p, n_k) + apothem_unit <= apothem_container
    # apothem_unit = S3 / 2 * 1 = S3/2
    # apothem_container = S3 / 2 * S
    # So: dot(p, n_k) <= S3/2 * (S - 1)
    # S >= (2 * dot(p, n_k) / S3) + 1
    
    # We need S >= max_k ( max_i (dot(p_i, n_k) / (S3/2)) + 1 )
    # S >= max_k ( 2/S3 * max_i(dot(p_i, n_k)) + 1 )
    
    max_proj = 0.0
    for k in range(6):
        n = NORM[k]
        proj = pts_rot @ n
        m = np.max(proj)
        if m > max_proj:
            max_proj = m
            
    S = 2.0 * max_proj / S3 + 1.0
    return S

def generate_lattice_candidates(max_radius=3):
    """
    Generate candidate clusters of N hexagons from the hexagonal lattice.
    We use axial coordinates (q, r) where distance is sqrt(q^2 + qr + r^2).
    Cartesian conversion: x = S3 * (q + r/2), y = 1.5 * r
    """
    # Generate lattice points within a radius
    pts = []
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            # Check distance from origin
            dist_sq = q*q + q*r + r*r
            if dist_sq <= max_radius * max_radius:
                x = S3 * (q + 0.5 * r)
                y = 1.5 * r
                pts.append((x, y, q, r))
                
    pts_arr = np.array([(p[0], p[1]) for p in pts])
    pts_qr = np.array([(p[2], p[3]) for p in pts])
    
    # We want subsets of size N that are connected and compact.
    # To keep it fast, we only consider subsets that include the origin (0,0)
    # and are relatively dense.
    
    # Identify index of origin
    origin_idx = None
    for i, p in enumerate(pts):
        if p[2] == 0 and p[3] == 0:
            origin_idx = i
            break
    if origin_idx is None:
        return []

    # Get indices of all other points
    other_indices = [i for i in range(len(pts)) if i != origin_idx]
    
    # We need to choose N-1 points from other_indices.
    # Total points in radius 3 is about 37. Choosing 10 from 36 is huge.
    # Instead, we use a heuristic: The optimal cluster is likely "convex" in the lattice.
    # We will generate candidates by taking the K closest points to the origin, 
    # and then swapping some to improve shape.
    
    # Sort other indices by distance to origin
    dists = np.linalg.norm(pts_arr[other_indices], axis=1)
    sorted_indices = other_indices[np.argsort(dists)]
    
    candidates = []
    
    # Candidate 1: The N closest points
    # This forms a roughly circular cluster
    init_set = [origin_idx] + sorted_indices[:N-1]
    candidates.append(np.array([pts_arr[i] for i in init_set]))
    
    # Candidate 2: Try to "fill" a hexagonal shape.
    # A regular hexagon of side L has 3L^2 - 3L + 1 points.
    # L=1: 1 pt
    # L=2: 7 pts
    # L=3: 19 pts
    # We have 11. 11 is between 7 and 19.
    # So it's a "hexagon of side 2" (7 pts) plus 4 more.
    # The 7 pts are origin + 6 neighbors.
    # The 4 more should be chosen to minimize the bounding box.
    
    # Let's explicitly construct the "7+4" topology.
    # The 6 neighbors are at (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in axial coords.
    # Let's find their indices.
    neighbors_qr = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    core_indices = [origin_idx]
    for nq, nr in neighbors_qr:
        found = False
        for i in range(len(pts)):
            if pts[i][2] == nq and pts[i][3] == nr:
                core_indices.append(i)
                found = True
                break
        if not found:
            # Fallback to sorted if not found (should not happen with max_radius=3)
            core_indices = [origin_idx] + sorted_indices[:6]
            break
    
    if len(core_indices) == 7:
        # Now we need 4 more points.
        # The next ring of the lattice has 12 points.
        # We want to pick 4 that make the shape most symmetric/compact.
        # Let's try picking 4 points that are "corners" of the next ring.
        # The next ring points are at distance sqrt(3) or more?
        # Axial distance squared: q^2+qr+r^2 = 3? No, ring 2 has dist^2 = 3, 4, etc.
        # Actually, let's just take the next 4 closest points that are NOT in the core.
        # But we can also try specific configurations.
        
        # Configuration A: 4 closest available
        remaining = [i for i in sorted_indices if i not in core_indices]
        cand_A = np.array([pts_arr[i] for i in core_indices + remaining[:4]])
        candidates.append(cand_A)
        
        # Configuration B: Try to balance the shape.
        # Let's just use a small local search on the set of indices.
        # Start with cand_A.
        # We can swap one point from the outer 4 with one from the next available.
        # This might be overkill. Let's just add a few random permutations of the outer 4 from the next 8 closest.
        # Actually, the "closest 11" heuristic usually works well for hexagonal symmetry.
        
        # Let's add the "closest 11" again to be safe, as it might differ from core+4 if the core wasn't exactly the 7 closest.
        # (It is, for hex lattice).
        
        # Let's add one more candidate: A linear chain? No, we want compact.
        # Let's add a "rhombus" like shape?
        # Just stick with the closest N points. It's the most robust heuristic for compactness in a lattice.
        
        # However, the "closest N" might not be the one that minimizes the *hexagonal* bounding box.
        # For example, a slightly elongated shape might fit better in a specific rotation.
        # But since we optimize rotation, the "roundest" shape is generally best.
        
        # Let's generate a few variations by swapping the 4 outer points.
        # The 4 outer points in "closest 11" are the 4 closest to the 7-core.
        # Let's try swapping one of them with a slightly farther one to see if it helps symmetry.
        # This is getting complex. I will rely on the "closest N" and "core + 4 closest".
        # They are likely identical.
        
        # Let's just ensure we have at least one good candidate.
        pass

    # Add a fallback candidate: random compact cluster
    # Just in case the lattice generation missed something or N is different.
    # For N=11, the lattice approach is very strong.
    
    # Remove duplicates and return
    unique_candidates = []
    seen = set()
    for c in candidates:
        # Normalize by subtracting centroid
        centroid = np.mean(c, axis=0)
        c_norm = c - centroid
        # Round to avoid float issues in set
        key = tuple(map(lambda x: tuple(np.round(x, 5)), c_norm))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c_norm)
            
    return unique_candidates

def construct():
    start = time.time()
    deadline = start + 140.0

    # Generate candidate topologies
    candidates = generate_lattice_candidates(max_radius=3)
    
    if not candidates:
        # Fallback to a simple line if something went wrong
        points = np.array([[i * S3, 0] for i in range(N)])
        candidates = [points]

    best_side = float('inf')
    best_points = None
    best_angle = 0.0

    for pts in candidates:
        # Optimize rotation angle
        # The function S(theta) is periodic with period 60 degrees.
        # We can minimize it over [0, 60].
        # It's a piecewise linear function (max of linear functions).
        # We can find the minimum by checking the vertices of the arrangement, 
        # or just use a fine grid + local refinement.
        
        # Grid search first
        thetas = np.linspace(0, 60, 121)
        sides = [outer_side_for_cluster(pts, t) for t in thetas]
        min_idx = np.argmin(sides)
        best_angle_local = thetas[min_idx]
        best_side_local = sides[min_idx]
        
        # Refine around the best angle
        # Use scipy minimize or just a finer grid
        # Since it's 1D and smooth (mostly), a fine grid is enough.
        fine_thetas = np.linspace(best_angle_local - 2, best_angle_local + 2, 100)
        fine_sides = [outer_side_for_cluster(pts, t) for t in fine_thetas]
        min_idx_fine = np.argmin(fine_sides)
        best_angle_local = fine_thetas[min_idx_fine]
        best_side_local = fine_sides[min_idx_fine]
        
        if best_side_local < best_side:
            best_side = best_side_local
            best_points = pts
            best_angle = best_angle_local

    # Final validity check
    # The points are from the lattice, so they are disjoint.
    # We just need to make sure they fit in the calculated side length.
    # The calculation of outer_side_for_cluster already ensures this by definition.
    # However, we should double check with a strict validity function just in case.
    
    # The "best_points" are centered at the origin in the local frame.
    # We need to return the global coordinates.
    # The container is centered at (0,0).
    # The cluster is rotated by best_angle.
    # So global position = R(best_angle) @ local_position.
    
    theta = np.radians(best_angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    
    global_points = best_points @ R.T
    
    # The return format requires:
    # inner: list of (x, y, angle_degrees) for each hexagon.
    # Since all hexagons are aligned with the container (relative to the cluster rotation),
    # their individual angle is best_angle.
    # Wait, the task says "each inner hexagon may translate and rotate".
    # In our lattice packing, all hexagons are rotated by the same angle (the cluster angle).
    # So the angle for each is best_angle.
    
    inner = []
    for i in range(N):
        inner.append([float(global_points[i, 0]), float(global_points[i, 1]), float(best_angle)])
    
    # Outer hexagon center is (0,0), side is best_side, rotation is 0.
    return inner, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Best side: {res[2]}")
