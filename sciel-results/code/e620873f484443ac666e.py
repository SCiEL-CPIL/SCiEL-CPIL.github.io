import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    max_time = 115.0
    
    # 1. Initialization: Concentric Rings
    # We distribute N points across K rings.
    # The number of points on a ring is roughly proportional to its radius (circumference).
    # Let R_k be the radius of ring k. Number of points n_k ~ 2 * pi * R_k / s.
    # Total N = sum n_k.
    # We approximate a hexagonal packing density.
    
    # Heuristic for ring structure:
    # Inner rings have fewer points.
    # Let's try to fit points into rings such that the spacing is uniform.
    
    # Simple approach: 
    # Determine number of rings K.
    # For a hexagonal lattice, the number of points in a hexagon of side m is 3m(m-1)+1.
    # But we are in a circle.
    # Let's just pick a reasonable number of rings, e.g., 10-15.
    
    K = 12  # Number of rings
    
    # Initial guess for radii: uniformly spaced in area? Or linear?
    # Area of circle is pi. Area per point is pi/N.
    # Radius of a disk containing n points is sqrt(n / (pi * density)).
    # Density for hex packing is pi / (2 * sqrt(3)) * (1/r^2) ... wait.
    # Area per circle of radius r in hex packing is 2 * sqrt(3) * r^2.
    # So if we have n points, area ~ n * 2 * sqrt(3) * r^2.
    # r ~ sqrt(1 / (2 * sqrt(3) * N)) for the whole thing? No, that's for the container radius 1.
    # Let's estimate r from the container:
    # N * 2 * sqrt(3) * r^2 <= pi  => r <= sqrt(pi / (2 * sqrt(3) * N))
    r_est = np.sqrt(np.pi / (2 * np.sqrt(3) * N))
    
    # We want to place points on rings.
    # Let's define rings by their radii R_0=0, R_1, ..., R_{K-1} < 1.
    # And angles.
    
    # Strategy:
    # 1. Assign points to rings.
    # 2. Optimize ring radii and angular shifts.
    # 3. Refine individual points with local search.
    
    # Assignment:
    # We want the density to be roughly constant.
    # n_k points on ring k.
    # Let's try to make the spacing between rings and along the ring similar.
    
    # Simple heuristic:
    # Let's just generate a hexagonal lattice first, then project to rings?
    # No, the prompt asks for a ring-based representation.
    
    # Let's generate points on rings.
    # We need to decide n_k.
    # Let's use a simple rule: n_k = round(2 * pi * R_k / s_init)
    # where s_init is the target spacing.
    # s_init ~ 2 * r_est.
    s_init = 2.0 * r_est
    
    # We will iteratively adjust R_k and n_k to sum to N.
    # Start with uniform radial spacing.
    radii = np.linspace(0.05, 0.95, K)
    
    # Calculate initial counts
    counts = np.zeros(K, dtype=int)
    for k in range(K):
        if k == 0:
            counts[k] = 1
        else:
            counts[k] = max(1, int(np.round(2 * np.pi * radii[k] / s_init)))
            
    # Adjust counts to sum to N
    while sum(counts) < N:
        # Add to the ring with the largest gap or random
        # Add to the outermost ring with space? Or balance?
        # Let's add to the ring that increases count least proportionally?
        # Just add to the last ring for simplicity in init, then optimize.
        counts[-1] += 1
    while sum(counts) > N:
        # Remove from the ring with the most points (usually outer)
        idx_max = np.argmax(counts)
        if counts[idx_max] > 1:
            counts[idx_max] -= 1
        else:
            # If all are 1, we have a problem. K is too high.
            # Reduce K?
            # For N=300, K=12, min sum is 12. Max sum is large.
            # If we can't reduce, break.
            break

    # Generate points
    points = []
    angles = np.zeros(K)
    for k in range(K):
        n_k = counts[k]
        if n_k == 1 and k == 0:
            points.append([0.0, 0.0])
        else:
            # Evenly spaced angles with an offset
            base_angles = np.linspace(0, 2*np.pi, n_k, endpoint=False) + angles[k]
            for a in base_angles:
                points.append([radii[k] * np.cos(a), radii[k] * np.sin(a)])
                
    x = np.array(points, dtype=np.float64)
    
    # Ensure exactly N points (safety)
    if len(x) < N:
        # Pad with random points
        diff = N - len(x)
        rand_pts = np.random.randn(diff, 2) * 0.5
        x = np.vstack([x, rand_pts])
    elif len(x) > N:
        x = x[:N]
        
    # 2. Optimization Loop
    # We will optimize:
    # A) Ring radii (if we keep the ring structure)
    # B) Angular offsets (if we keep the ring structure)
    # C) Individual point positions (local search)
    
    # Given the time constraint and the "ring-based" directive, 
    # a pure ring optimization might be too constrained to beat a free hex lattice 
    # unless the local search is very effective.
    # However, the "structural change" is the representation.
    # Let's stick to the ring structure for the first phase, then allow free movement 
    # but initialize from rings.
    
    # Actually, the prompt says: "Use a ring-based representation... allowing independent tuning... 
    # This avoids the fixed lattice structure..."
    # So I should optimize the ring parameters (radii, offsets) and possibly the counts?
    # Changing counts is discrete and hard.
    # Let's optimize radii and angular offsets.
    
    # We need a fast score function.
    # Since we are moving points, we can't just recalculate O(N^2) score every step if we do many steps.
    # But N=300, O(N^2) is 90,000 ops. That's fast in numpy.
    # 10,000 steps * 90k ops = 900M ops. Might be tight in 115s in Python/NumPy if not optimized.
    # Let's use a local check for acceptance.
    
    def calc_score(x):
        # Fast exact score
        # Using cdist is faster than manual broadcasting for small N?
        # Or just manual.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)

    best_x = x.copy()
    best_r = calc_score(best_x)
    
    # Phase 1: Optimize Ring Parameters (Radii and Angles)
    # We group points by their original ring index.
    # We need to track which point belongs to which ring.
    # Let's create an array `ring_ids` for the current x.
    ring_ids = np.zeros(N, dtype=int)
    idx = 0
    for k in range(K):
        n_k = counts[k]
        for _ in range(n_k):
            ring_ids[idx] = k
            idx += 1
    # If we padded/trimmed, ring_ids might be wrong for the last few.
    # Let's just re-assign ring_ids based on current radius?
    # No, let's stick to the initial assignment for the parameter optimization.
    # If we move points freely later, we lose the ring structure.
    
    # Let's do a hybrid:
    # 1. Optimize ring radii/angles while points are constrained to rings?
    #    This is hard to implement efficiently with individual point moves.
    # 2. Instead, treat the ring parameters as the "initialization" and then
    #    perform a standard local search (free points) but with a very good start.
    #    The "structural change" is the *initialization* and the *idea* that rings are a good manifold.
    #    But the prompt asks to *use* the representation.
    
    # Let's try to optimize the ring parameters directly.
    # Parameters: radii (K), angles (K).
    # We can use gradient ascent on the smooth min score.
    # But calculating gradients for ring parameters requires mapping back to points.
    
    # Alternative: 
    # Just do a local search on the points, but initialized from the rings.
    # The "ring-based" aspect is the initialization and potentially the constraint that 
    # points stay on rings? 
    # "Place the centres... in a ring-based representation... allowing independent tuning of radial and tangential spacing"
    # This implies the variables ARE the ring radii and offsets.
    
    # Let's implement an optimizer for the ring parameters.
    # We will fix the counts.
    # Variables: R (K,), A (K,)
    # Point i on ring k:
    #   theta_i = 2*pi*i/n_k + A_k
    #   x_i = R_k * [cos(theta_i), sin(theta_i)]
    
    # We will perform gradient ascent on the smooth min distance.
    
    def get_points_from_params(R, A):
        pts = np.zeros((N, 2))
        idx = 0
        for k in range(K):
            n_k = counts[k]
            if n_k == 1 and k == 0:
                pts[idx] = [0.0, 0.0]
                idx += 1
            else:
                base_angles = np.linspace(0, 2*np.pi, n_k, endpoint=False) + A[k]
                cs = np.cos(base_angles)
                sn = np.sin(base_angles)
                pts[idx:idx+n_k, 0] = R[k] * cs
                pts[idx:idx+n_k, 1] = R[k] * sn
                idx += n_k
        return pts

    # Initial params
    R = radii.copy()
    A = angles.copy()
    
    # Smooth min function for parameters
    def smooth_score_params(R, A, beta=50.0):
        pts = get_points_from_params(R, A)
        # Check validity
        if np.any(np.linalg.norm(pts, axis=1) > 1.0):
            return -1e10
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Get lower triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        pair_vals = pair_dists / 2.0
        
        norms = np.linalg.norm(pts, axis=1)
        bound_vals = 1.0 - norms
        
        vals = np.concatenate([pair_vals, bound_vals])
        
        # Log-sum-exp for min
        # min(vals) approx -1/beta * log(sum(exp(-beta * vals)))
        vals_scaled = -beta * vals
        max_val = np.max(vals_scaled)
        if not np.isfinite(max_val):
            return -1e10
        vals_scaled = vals_scaled - max_val
        logsumexp = np.log(np.sum(np.exp(vals_scaled)))
        smooth_min = -(max_val + logsumexp) / beta
        
        return smooth_min

    # Gradient of smooth score w.r.t parameters
    # This is complex. 
    # Alternative: Use a simple coordinate descent or random hill climbing on the parameters.
    # Given the time limit, let's try a few steps of coordinate ascent on R and A.
    
    current_params = (R.copy(), A.copy())
    current_score = smooth_score_params(R, A)
    best_params = (R.copy(), A.copy())
    best_score = current_score
    
    # Time budget for phase 1: 20s
    phase1_end = start_time + 20.0
    
    while time.time() < phase1_end:
        # Try perturbing one ring radius
        k = np.random.randint(K)
        old_R = R[k]
        # Random step
        step = np.random.uniform(-0.01, 0.01)
        R[k] = old_R + step
        
        # Try perturbing one ring angle
        k2 = np.random.randint(K)
        old_A = A[k2]
        step_a = np.random.uniform(-0.1, 0.1)
        A[k2] = old_A + step_a
        
        new_score = smooth_score_params(R, A)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_params = (R.copy(), A.copy())
        else:
            R[k] = old_R
            A[k2] = old_A
            # Decay step size? We are using fixed uniform for simplicity.
            
    # Update x with best params
    x = get_points_from_params(best_params[0], best_params[1])
    best_r = calc_score(x)
    best_x = x.copy()
    
    # Phase 2: Local Search on Points (Free movement)
    # This allows escaping the ring constraints if they are suboptimal.
    # We use a fast local check.
    
    # Precompute neighbors? No, just check all pairs for the moved point.
    # Moving point i:
    # Check dist to all j != i.
    # Check dist to boundary.
    
    alpha = 0.005
    best_r = calc_score(x)
    best_x = x.copy()
    
    # Use a time guard
    while time.time() < start_time + max_time - 2.0:
        # Pick a random point
        i = np.random.randint(N)
        
        # Generate a random move
        dx = np.random.randn(2) * alpha
        new_x = x.copy()
        new_x[i] += dx
        
        # Project to boundary if needed
        norm_new = np.linalg.norm(new_x[i])
        if norm_new > 1.0:
            new_x[i] /= norm_new
            
        # Check validity and score change locally
        # Only check distances involving point i
        dists_to_i = np.linalg.norm(new_x - new_x[i], axis=1)
        dists_to_i[i] = np.inf
        min_pair_new = np.min(dists_to_i) / 2.0
        
        bound_new = 1.0 - norm_new
        
        # We need the current min pair and bound to compare.
        # But we don't have them stored.
        # We can estimate if the move improves the *local* minimum.
        # If min_pair_new > best_r and bound_new > best_r, it's a candidate.
        # But we need to ensure we didn't break a constraint that was *not* the minimum.
        # Actually, if the new local min is > best_r, it doesn't guarantee the global min is > best_r.
        # Example: best_r was determined by pair (a,b). We move i. 
        # If dist(i,j) > best_r for all j, then the new global min is still >= best_r (determined by a,b).
        # So if min_pair_new > best_r AND bound_new > best_r, the move is SAFE (doesn't decrease score below best_r).
        # It might increase the score if best_r was determined by i.
        
        if min_pair_new > best_r and bound_new > best_r:
            # Safe move. Accept.
            x = new_x
            # Check if it improved the score
            # We can update best_r if the move was the bottleneck.
            # But calculating global score is expensive.
            # Let's just accept safe moves and periodically check global score.
            pass
        else:
            # Rejected. Maybe shrink alpha?
            alpha *= 0.95
            if alpha < 0.0001:
                alpha = 0.005
                # Random restart? No, just reset alpha.
                
    # Final polish: Calculate exact score and return
    # We might have accepted moves that are safe but didn't improve best_r.
    # Let's do a few exact checks and accept improvements.
    
    # Actually, the "safe move" logic ensures we never go below best_r.
    # So x is always valid with radius >= best_r.
    # To find if best_r can be increased, we need to see if the *limiting* constraint has moved.
    # The local search will naturally push points apart.
    
    # Let's do a final exact evaluation and a few gradient steps if time permits?
    # No, just return the best valid x found.
    
    final_r = calc_score(x)
    if final_r > best_r:
        best_r = final_r
        best_x = x.copy()
        
    return best_x

# Test
if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
