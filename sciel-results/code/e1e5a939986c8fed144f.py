"""Combined program: hexagonal lattice packing + local optimization with tight outer fit."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 150s limit, keep 10s margin

    # Lattice vectors for hexagonal packing
    # Distance between centers of touching unit hexagons is sqrt(3)
    # We add a small epsilon to ensure strict disjointness under the center-distance constraint
    eps = 0.01
    d = math.sqrt(3) + eps

    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])

    # Build a cluster of 12 hexagons using ring-based construction
    centers = [np.array([0.0, 0.0])]

    # Ring 1: 6 neighbors of center
    basis = [u, v, v - u, -u, -v, u - v]
    ring1_vectors = [vec.copy() for vec in basis]
    centers.extend(ring1_vectors)

    # Ring 2: find neighbors of ring1 that are not in ring0 or ring1
    ring2_candidates = []
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            # Check if p2 is in Ring 0
            if np.allclose(p2, np.array([0.0, 0.0]), atol=eps*0.5):
                continue
            # Check if p2 is in Ring 1
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j, atol=eps*0.5):
                    is_r1 = True
                    break
            if is_r1:
                continue
            # Avoid duplicates
            duplicate = False
            for pc in ring2_candidates:
                if np.allclose(p2, pc, atol=eps*0.5):
                    duplicate = True
                    break
            if not duplicate:
                ring2_candidates.append(p2.copy())

    # Sort Ring 2 candidates by distance to origin and pick 5 closest
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(min(5, len(ring2_candidates))):
        centers.append(ring2_candidates[i].copy())

    # Ensure we have exactly 12
    while len(centers) < N:
        # Fallback: add random points far away (should not happen with ring construction)
        angle = random.uniform(0, 2*math.pi)
        dist = d * 3.0
        new_c = np.array([dist*math.cos(angle), dist*math.sin(angle)])
        centers.append(new_c)
    
    centers = centers[:N]

    # Convert to list of [x, y, angle]
    # Rotate hexagons by 30 degrees so they are 'flat-top' relative to the x-axis
    # This orientation is optimal for the lattice packing
    inner = [[float(c[0]), float(c[1]), 30.0] for c in centers]

    def check_disjoint(centers, min_dist=math.sqrt(3) + eps):
        """
        Check if all hexagons are disjoint using center distance.
        For regular hexagons of side 1, the minimum center distance to avoid overlap
        is sqrt(3) (when aligned). With eps, we ensure strict separation.
        This is a sufficient condition. It is not necessary if rotations are used,
        but it allows for very fast O(N^2) checks.
        """
        n = len(centers)
        for i in range(n):
            xi, yi = centers[i]
            for j in range(i + 1, n):
                xj, yj = centers[j]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist * min_dist:
                    return False
        return True

    def compute_outer_side(centers, ocx, ocy, otheta):
        """
        Compute the minimum side length of a regular hexagon centered at (ocx, ocy)
        with rotation otheta that contains all inner hexagons.
        The inner hexagons are assumed to be at (cx, cy) with angle 30 deg (flat top).
        We calculate the maximum projection of all inner vertices onto the 6 outward normals.
        """
        max_proj = 0.0
        # Pre-compute vertices for all inner hexagons? 
        # For speed, we compute on the fly but efficiently.
        # Inner hexagons have side 1, angle 30 deg.
        # Vertices relative to center: (cos(30+k*60), sin(30+k*60))
        
        # Normals of the outer hexagon
        # Outer hexagon vertices are at angles otheta + k*60.
        # Normals are at otheta + 30 + k*60.
        # The apothem is side * sqrt(3)/2.
        # So side = 2 * max_proj / sqrt(3).
        
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            max_proj_normal = -1e9
            for i in range(len(centers)):
                cx, cy = centers[i]
                # Check all 6 vertices of the i-th inner hexagon
                # Inner hex angle is 30 deg = pi/6
                for v in range(6):
                    vert_angle = math.pi/6 + v * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj_normal:
                        max_proj_normal = proj
            
            if max_proj_normal > max_proj:
                max_proj = max_proj_normal
        
        return 2.0 * max_proj / math.sqrt(3)

    def optimize_outer(centers):
        """
        Find the best outer hexagon (center, angle) for a fixed set of inner centers.
        """
        # Initial center estimate: centroid of inner centers
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        
        best_side = float('inf')
        best_ocx, best_ocy, best_otheta = cx, cy, 0.0
        
        # Scan angles from 0 to 60 degrees
        # 61 steps is a good balance
        for theta_deg in np.linspace(0, 60, 61):
            otheta = math.radians(theta_deg)
            
            # For each angle, optimize the center slightly
            # We can do a small local search on the center
            # Or just evaluate at the centroid and a few nearby points
            
            # Evaluate at centroid
            side = compute_outer_side(centers, cx, cy, otheta)
            if side < best_side:
                best_side = side
                best_ocx, best_ocy, best_otheta = cx, cy, theta_deg
            
            # Try a few nearby centers to account for asymmetry
            # The optimal center might shift slightly from the centroid
            for dx in [-0.1, 0.0, 0.1]:
                for dy in [-0.1, 0.0, 0.1]:
                    if dx == 0 and dy == 0:
                        continue
                    side = compute_outer_side(centers, cx + dx, cy + dy, otheta)
                    if side < best_side:
                        best_side = side
                        best_ocx, best_ocy, best_otheta = cx + dx, cy + dy, theta_deg
        
        return best_side, best_ocx, best_ocy, best_otheta

    # Initial evaluation
    if check_disjoint(inner):
        outer_side, outer_cx, outer_cy, outer_angle = optimize_outer(inner)
        outer_center = (outer_cx, outer_cy)
    else:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        outer_side, outer_cx, outer_cy, outer_angle = optimize_outer(inner)
        outer_center = (outer_cx, outer_cy)

    best_arr = [list(x) for x in inner]
    best_side = outer_side
    best_center = outer_center
    best_angle = outer_angle

    # Local optimization
    # We will perturb inner positions and re-optimize the outer hexagon.
    # Since check_disjoint is fast, we can do many iterations.
    
    current_time = time.time()
    iterations = 0
    step_size = 0.1
    
    while time.time() < deadline - 10:
        iterations += 1
        
        # Adaptive step size: start larger, get smaller
        # But keep it simple: constant step with random restarts if stuck?
        # Let's use a decaying step size or just a fixed reasonable one.
        # 0.1 is about 10% of the lattice spacing.
        
        new_inner = [list(x) for x in best_arr]
        
        # Perturb a random hexagon
        idx = random.randint(0, N - 1)
        dx = random.uniform(-step_size, step_size)
        dy = random.uniform(-step_size, step_size)
        # Angles: We keep them fixed at 30 deg for the lattice logic.
        # If we change angles, the center-distance check becomes insufficient.
        # So we stick to position-only perturbations to maintain the fast check.
        # This restricts us to the "aligned" family, which is likely optimal for this N.
        
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        
        if check_disjoint(new_inner):
            side, ocx, ocy, otheta = optimize_outer(new_inner)
            if side < best_side - 1e-6:
                best_side = side
                best_arr = new_inner
                best_center = (ocx, ocy)
                best_angle = otheta
                inner = [list(x) for x in best_arr]
            
            # Occasionally try a larger perturbation
            if iterations % 50 == 0:
                new_inner = [list(x) for x in best_arr]
                idx = random.randint(0, N - 1)
                dx = random.uniform(-0.5, 0.5)
                dy = random.uniform(-0.5, 0.5)
                new_inner[idx][0] += dx
                new_inner[idx][1] += dy
                
                if check_disjoint(new_inner):
                    side, ocx, ocy, otheta = optimize_outer(new_inner)
                    if side < best_side - 1e-6:
                        best_side = side
                        best_arr = new_inner
                        best_center = (ocx, ocy)
                        best_angle = otheta
                        inner = [list(x) for x in best_arr]
        
        # Update step size slightly?
        # If we haven't improved in a while, shrink step size?
        # For now, keep it simple.
        
        # Every 1000 iterations, try to re-center the whole cluster?
        # No, the outer optimizer handles centering.

    return inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
