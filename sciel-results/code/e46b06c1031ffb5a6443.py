"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import cKDTree

def _score(centers):
    """Compute the valid radius for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    min_dist_to_wall = np.min(0.5 - np.abs(centers))
    if min_dist_to_wall < 0:
        return 0.0
    
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        # Use numpy to compute distances for pairs
        min_pair_dist = np.inf
        # For speed, if pairs are many, we can just find min
        # But N=200, pairs < 200*199/2 = 19900, which is fine
        # However, doing it in a loop is slow. Let's use vectorized if possible or just loop if small.
        # Actually, cKDTree.query_pairs returns indices.
        # Let's just compute min dist directly.
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
                if min_pair_dist < 1e-9:
                    break

    if not np.isfinite(min_pair_dist):
        min_pair_dist = 1.0 # Should not happen for N=200 in unit square

    r = min(min_dist_to_wall, min_pair_dist / 2.0)
    return max(0.0, r)

def _make_initial_grid():
    """Create a slightly perturbed square grid to break symmetry."""
    m = int(math.ceil(math.sqrt(N)))
    # Create m x m grid
    xs = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, m)
    ys = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, m)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Keep first N points
    pts = pts[:N]
    
    # Add small random perturbation to avoid symmetry traps
    pts += np.random.randn(N, 2) * 0.01
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _md_relax(centers, deadline):
    """
    Force-directed relaxation (Molecular Dynamics) for circle packing.
    Uses a k-d tree to efficiently find neighbors and compute repulsive forces.
    """
    centers = centers.copy()
    n = len(centers)
    
    # Parameters
    dt = 0.005
    max_iter = 50000
    force_exp = 4.0  # Higher exponent = stiffer repulsion, better for jamming
    
    # Initial velocity
    v = np.zeros((n, 2))
    
    best_r = _score(centers)
    best_centers = centers.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Build tree
        tree = cKDTree(centers)
        
        # Find pairs within a cutoff. 
        # We don't know the exact optimal distance, but we can use a generous cutoff
        # like 1.0 (entire box) or tighter. 0.5 is a good start.
        # As packing tightens, we can reduce cutoff, but 0.5 is safe.
        pairs = tree.query_pairs(0.5)
        
        forces = np.zeros((n, 2))
        
        if len(pairs) > 0:
            # Vectorized force calculation is tricky with variable pairs, 
            # but N=200 is small enough that a loop over pairs is acceptable 
            # if the number of pairs is not huge.
            # Worst case ~20k pairs. 20k * 50000 iters = 1e9 ops. Might be too slow in Python loop.
            # Let's try to vectorize.
            
            i_idx, j_idx = zip(*pairs)
            i_idx = np.array(i_idx)
            j_idx = np.array(j_idx)
            
            diffs = centers[i_idx] - centers[j_idx]
            dists = np.linalg.norm(diffs, axis=1)
            
            # Avoid division by zero
            dists[dists < 1e-9] = 1e-9
            
            # Force magnitude: k * (1/d)^exp
            # We want to push them apart.
            # Force on i is +diff/dist * mag
            # Force on j is -diff/dist * mag
            
            mags = (1.0 / dists) ** force_exp
            unit_vecs = diffs / dists[:, np.newaxis]
            
            f_vecs = mags[:, np.newaxis] * unit_vecs
            
            # Accumulate forces
            # np.add.at is slow. Let's use scatter if possible, or just loop if pairs < 1000.
            # For N=200, max pairs is 19900. 
            # Let's use a loop if it's fast enough, or try to optimize.
            # Actually, let's just use the loop for clarity and check speed.
            # If too slow, we can reduce max_iter or cutoff.
            
            # Optimization: Only compute forces for pairs that are actually overlapping or close.
            # With force_exp=4, forces decay fast.
            
            for k in range(len(pairs)):
                i = i_idx[k]
                j = j_idx[k]
                f = f_vecs[k]
                forces[i] += f
                forces[j] -= f
        
        # Boundary forces
        # If center is too close to wall, push it in.
        # Wall distance < 0.1?
        # Let's use a soft boundary.
        # Force = k_wall * (0.1 - dist)^2 if dist < 0.1
        wall_margin = 0.05
        k_wall = 10.0
        
        dist_left = centers[:, 0] + 0.5
        dist_right = 0.5 - centers[:, 0]
        dist_bottom = centers[:, 1] + 0.5
        dist_top = 0.5 - centers[:, 1]
        
        # Left wall
        mask = dist_left < wall_margin
        if np.any(mask):
            forces[mask, 0] += k_wall * (wall_margin - dist_left[mask])
            
        # Right wall
        mask = dist_right < wall_margin
        if np.any(mask):
            forces[mask, 0] -= k_wall * (wall_margin - dist_right[mask])
            
        # Bottom wall
        mask = dist_bottom < wall_margin
        if np.any(mask):
            forces[mask, 1] += k_wall * (wall_margin - dist_bottom[mask])
            
        # Top wall
        mask = dist_top < wall_margin
        if np.any(mask):
            forces[mask, 1] -= k_wall * (wall_margin - dist_top[mask])
        
        # Update velocity and position
        # Damping
        v = 0.9 * v + dt * forces
        centers += dt * v
        
        # Clip to box (hard constraint to prevent escape)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Score check every 100 iterations
        if it % 100 == 0:
            r = _score(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        # Convergence check: if forces are very small, we might be stuck or converged
        # But we want to keep going to "jam" the packing.
        
    return best_centers, best_r

def _local_polish(centers, deadline):
    """
    Targeted local optimization: find the closest pair and try to separate them.
    """
    centers = centers.copy()
    current_r = _score(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    n = len(centers)
    
    for step in [0.01, 0.005, 0.002, 0.001, 0.0005]:
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return best_centers, best_r
            
            # Find closest pair
            tree = cKDTree(centers)
            pairs = tree.query_pairs(0.2) # Generous cutoff
            
            if len(pairs) == 0:
                break
                
            # Find the actual minimum distance pair
            min_d = np.inf
            min_pair = None
            for i, j in pairs:
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
                    
            if min_pair is None:
                break
                
            i, j = min_pair
            # Try to move i and j apart
            # Direction from i to j
            diff = centers[j] - centers[i]
            norm = np.linalg.norm(diff)
            if norm < 1e-9:
                continue
            unit = diff / norm
            
            # Try moving i in -unit direction and j in +unit direction
            # And also try moving other points if they are in the way?
            # For now, just move the pair.
            
            best_pos_i = centers[i].copy()
            best_pos_j = centers[j].copy()
            best_r_local = current_r
            
            for t in [step, step*0.5, step*0.1]:
                new_i = centers[i] - unit * t
                new_j = centers[j] + unit * t
                
                # Check boundaries
                if np.any(np.abs(new_i) > 0.5) or np.any(np.abs(new_j) > 0.5):
                    continue
                    
                centers[i] = new_i
                centers[j] = new_j
                
                r = _score(centers)
                if r > best_r_local:
                    best_r_local = r
                    best_pos_i = new_i.copy()
                    best_pos_j = new_j.copy()
            
            centers[i] = best_pos_i
            centers[j] = best_pos_j
            
            if best_r_local > current_r:
                current_r = best_r_local
                improved = True
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                    
            # Also try moving other points that are close to i or j
            # This is a heuristic.
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. Use 145s.
    deadline = start_time + 145.0
    
    # 1. Initial Grid
    centers = _make_initial_grid()
    
    # 2. MD Relaxation
    # Allocate most of the time to MD
    md_deadline = deadline - 10.0
    centers, r = _md_relax(centers, md_deadline)
    
    # 3. Local Polish
    polish_deadline = deadline - 1.0
    centers, r2 = _local_polish(centers, polish_deadline)
    if r2 > r:
        r = r2
    
    return centers
# EVOLVE-BLOCK-END
