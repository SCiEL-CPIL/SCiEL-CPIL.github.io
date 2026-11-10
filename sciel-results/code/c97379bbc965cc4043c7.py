import numpy as np
import time
import math

def hexagon_packing_12():
    """
    Finds a packing of 12 unit regular hexagons inside a regular hexagon 
    by searching a discrete hexagonal lattice.
    
    Strategy:
    1. Generate a fine grid of candidate centers on a hexagonal lattice.
    2. Use a greedy algorithm to find an initial valid packing of 12 hexagons.
    3. Perform a local search: iterate through all pairs (current_hex, candidate_hex) 
       and if replacing current_hex with candidate_hex maintains validity and 
       reduces the bounding hexagon side length s, accept the move.
    4. Repeat until no improvement is found or time limit reached.
    
    This discrete approach avoids local minima common in continuous optimization 
    and leverages the symmetry of the problem.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    sqrt3 = math.sqrt(3)
    N = 12
    
    # Lattice basis vectors for hexagonal packing of unit hexagons
    # e1 = (1.5, sqrt(3)/2)
    # e2 = (0.5, sqrt(3)/2)
    e1 = np.array([1.5, sqrt3/2])
    e2 = np.array([0.5, sqrt3/2])
    
    # Generate candidate points
    # We need a region large enough to contain the optimal packing.
    # Optimal s is around 3.6-3.8. Radius ~ 4.
    # Let's use a square region in lattice coordinates [-5, 5]
    max_coord = 6
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            p = i * e1 + j * e2
            # Limit to a circular region to reduce candidates
            if np.linalg.norm(p) < 5.5:
                candidates.append(p)
    
    candidates = np.array(candidates)
    num_candidates = len(candidates)
    
    # Pre-compute vertices offsets for a unit hexagon (angle 0)
    v_offsets = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    # Pre-compute pairwise distances between all candidates
    # dist_sq[i, j] = ||candidates[i] - candidates[j]||^2
    # Two unit hexagons overlap if center distance < sqrt(3) - epsilon
    # Actually, min distance for non-overlap is sqrt(3).
    # Let's verify: Hex A at (0,0), Hex B at (sqrt(3), 0).
    # Rightmost point of A is (1,0). Leftmost point of B is (sqrt(3)-1, 0).
    # sqrt(3) ~ 1.732. sqrt(3)-1 ~ 0.732.
    # Wait, the vertices are at distance 1 from center.
    # If centers are distance d apart, do they overlap?
    # The "width" of a hexagon in any direction is between 1 and 2?
    # No, the distance between parallel sides is sqrt(3).
    # If we align them, the minimum center distance to touch is sqrt(3).
    # If we rotate, it can be smaller?
    # For regular hexagons, the minimum distance between centers for non-overlap 
    # is indeed sqrt(3) (when they share an edge).
    # If they touch at a vertex, distance is 2.
    # So we require dist >= sqrt(3).
    
    min_dist_sq = 3.0 - 1e-9 # Allow tiny numerical error
    
    # Compute distance matrix
    # This is O(K^2) where K is number of candidates (~200). 40k ops. Fast.
    diffs = candidates[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    dist_sq = np.sum(diffs**2, axis=-1)
    
    # valid_pair[i, j] is True if candidates i and j can coexist
    valid_pair = dist_sq > min_dist_sq
    np.fill_diagonal(valid_pair, False)
    
    def calc_s(centers):
        # centers: (N, 2)
        # Compute max projection of all vertices onto the 6 normal directions of the outer hexagon
        # Outer hexagon is assumed to be at origin, angle 0.
        # Normals are at angles 30, 90, 150, 210, 270, 330 degrees.
        # Normal vectors:
        # n0 = (cos(30), sin(30)) = (sqrt3/2, 1/2)
        # n1 = (cos(90), sin(90)) = (0, 1)
        # n2 = (cos(150), sin(150)) = (-sqrt3/2, 1/2)
        # n3 = -n0
        # n4 = -n1
        # n5 = -n2
        # Max projection P = max( v . n_k ) for all vertices v and normals n_k.
        # s = P * 2 / sqrt(3)
        
        # Vertices for all centers: (N, 6, 2)
        verts = centers[:, np.newaxis, :] + v_offsets[np.newaxis, :, :]
        
        # Normals
        n0 = np.array([sqrt3/2, 0.5])
        n1 = np.array([0.0, 1.0])
        n2 = np.array([-sqrt3/2, 0.5])
        normals = np.array([n0, n1, n2, -n0, -n1, -n2])
        
        # Project all vertices onto all normals
        # verts: (N, 6, 2), normals: (6, 2)
        # projs: (N, 6, 6) -> proj[i, j, k] = verts[i, j] . normals[k]
        projs = np.einsum('ijk,kl->ijl', verts, normals)
        
        max_proj = np.max(projs)
        return max_proj * 2.0 / sqrt3

    def is_valid_set(indices):
        # indices: list of 12 candidate indices
        # Check pairwise validity
        for i in range(N):
            for j in range(i+1, N):
                if not valid_pair[indices[i], indices[j]]:
                    return False
        return True

    # --- Phase 1: Greedy Initialization ---
    # We want to pick 12 points that are "central" to minimize s.
    # Sort candidates by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    
    selected = []
    for idx in sorted_idx:
        if len(selected) == N:
            break
        # Check if idx is compatible with all selected
        compatible = True
        for s_idx in selected:
            if not valid_pair[idx, s_idx]:
                compatible = False
                break
        if compatible:
            selected.append(idx)
            
    if len(selected) < N:
        # Fallback: If greedy failed (unlikely with this radius), use a fixed known pattern
        # Pattern: 3x4 block centered
        # Generate specific lattice points
        fallback_indices = []
        # We need to map coordinates back to indices in candidates
        # This is tricky. Let's just try a denser grid or different order if this fails.
        # For now, if it fails, we might return an invalid result or a very large s.
        # But with radius 5.5, it should definitely find 12.
        pass

    current_indices = np.array(selected)
    current_s = calc_s(candidates[current_indices])
    best_indices = current_indices.copy()
    best_s = current_s
    
    # --- Phase 2: Local Search ---
    # Try to improve by swapping one hexagon.
    # For each i in 0..11, for each candidate k not in selected:
    #   if valid_pair[k, selected[j]] for all j != i:
    #      new_set = selected with i replaced by k
    #      new_s = calc_s(new_set)
    #      if new_s < best_s - 1e-9:
    #          accept
    
    # Pre-compute which candidates are compatible with which selected sets?
    # That's too expensive. Just check on the fly.
    
    # Optimization: 
    # calc_s is expensive. We can compute it faster if we maintain the max projections.
    # But N=12 is small. calc_s takes ~1ms.
    # 12 * 200 swaps = 2400 checks per pass. 2400 * 1ms = 2.4s per pass.
    # We can do ~200 passes in 500s. That's plenty.
    
    while time.time() < deadline - 10:
        improved = False
        current_indices_list = list(current_indices)
        
        # Shuffle candidates to avoid bias? Or just iterate.
        # Iterating in sorted order (central first) is good for finding improvements that pull inwards.
        
        for i in range(N):
            idx_to_remove = current_indices_list[i]
            
            # Check compatibility of all other selected with candidate k
            others = [current_indices_list[j] for j in range(N) if j != i]
            
            for k in sorted_idx:
                if k in current_indices_list:
                    continue
                
                # Check if k is compatible with all others
                compatible = True
                for o in others:
                    if not valid_pair[k, o]:
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                # Try the swap
                new_list = current_indices_list.copy()
                new_list[i] = k
                new_indices = np.array(new_list)
                
                new_s = calc_s(candidates[new_indices])
                
                if new_s < best_s - 1e-12:
                    best_s = new_s
                    best_indices = new_indices.copy()
                    current_indices = new_indices.copy()
                    current_indices_list = new_list.copy()
                    improved = True
                    # Don't break inner loop, keep looking for more improvements in this pass
                    # Or break to restart? Restarting is safer for convergence.
                    # Let's break inner and outer to restart the pass with new state.
                    break
            
            if improved:
                break
        
        if not improved:
            # Local minimum found. 
            # To escape, we could do a random perturbation or accept a slightly worse move (SA).
            # Given the discrete nature and "convex" objective (mostly), local min is likely global 
            # or very close.
            # Let's try a few random restarts if time permits.
            # Actually, let's just stop if we hit a local minimum.
            # Or try "kick" moves: remove one, add a random one even if s increases slightly, 
            # then re-optimize.
            
            # Simple kick: pick a random i, pick a random compatible k, swap, even if s increases.
            # Then run local search again.
            # But we need to track best.
            
            # Let's do 50 random kicks if time allows.
            kicks_remaining = 50
            while kicks_remaining > 0 and time.time() < deadline - 20:
                kicks_remaining -= 1
                i = random.randint(0, N-1)
                idx_to_remove = current_indices_list[i]
                others = [current_indices_list[j] for j in range(N) if j != i]
                
                # Pick a random candidate k compatible with others
                valid_ks = []
                for k in sorted_idx:
                    if k in current_indices_list:
                        continue
                    compatible = True
                    for o in others:
                        if not valid_pair[k, o]:
                            compatible = False
                            break
                    if compatible:
                        valid_ks.append(k)
                
                if not valid_ks:
                    continue
                    
                k = random.choice(valid_ks)
                
                new_list = current_indices_list.copy()
                new_list[i] = k
                current_indices = np.array(new_list)
                current_indices_list = new_list.copy()
                
                # Re-run local search from this new state
                # (This is a bit expensive but we have budget)
                # Actually, just let the main loop continue. The main loop will try to improve from here.
                # We need to set a flag to continue the outer while loop.
                # The current structure breaks if improved. If not improved, we do a kick and continue.
                # So we just continue the outer while loop.
                break # Break out of kick loop to go back to main while

    # Final Result Construction
    final_centers = candidates[best_indices]
    final_s = calc_s(final_centers)
    
    # Shift centers so that the "center of mass" or bounding box center is at (0,0)?
    # The problem allows the outer hex to be anywhere.
    # The objective function calc_s assumes outer hex is at (0,0).
    # So we should shift the inner hexes so that their "center" aligns with the outer hex center (0,0).
    # The "center" that minimizes s is the center of the bounding hexagon.
    # For a set of points, the minimal enclosing hexagon (aligned with axes) center is 
    # determined by the midpoints of the max/min projections.
    
    # Let's compute the optimal center for the outer hexagon for the given set of inner hexes.
    # Let P be the set of all vertices of all inner hexes.
    # We want to find (cx, cy) such that max( (v - c) . n ) is minimized for all n.
    # This is equivalent to minimizing the width in each normal direction.
    # For a hexagon, the width in direction n is max(v.n) - min(v.n).
    # If we shift by c, v.n becomes (v-c).n = v.n - c.n.
    # The range is invariant to shift! 
    # Wait. The outer hexagon is defined by its center and size s.
    # The condition is that all inner vertices v must satisfy v . n <= s * sqrt(3)/2 for all normals n,
    # AND v . (-n) <= s * sqrt(3)/2.
    # i.e., |v . n| <= apothem.
    # This assumes the outer hex is centered at origin.
    # If we can move the outer hex, we can move the inner hexes relative to it.
    # The problem asks for (inner, outer, s).
    # We can return inner hexes at their calculated positions, and outer hex at (0,0,0).
    # But we must ensure they fit.
    # If we calculated s assuming outer at (0,0), we must ensure the inner hexes are positioned 
    # such that they fit in a hexagon at (0,0) of size s.
    # My calc_s function computes s = max_proj * 2 / sqrt(3).
    # This s is the minimum size of a hexagon centered at (0,0) that contains the points.
    # So if I return inner hexes at their current positions and outer at (0,0), 
    # and s = calc_s(...), it is valid by definition of calc_s.
    
    # However, we might be able to get a smaller s by shifting the inner hexes?
    # No, shifting the inner hexes changes their positions relative to the outer (0,0).
    # The "shape" of the cluster is fixed. The minimal enclosing hexagon size is an intrinsic property 
    # of the cluster (up to rotation, but we fixed rotation to 0).
    # Actually, we can rotate the cluster? No, the lattice is fixed.
    # Can we translate the cluster? 
    # If I translate the cluster, the max/min projections change.
    # Let's find the optimal translation.
    # We want to minimize s such that for all vertices v, and all normals n:
    # (v + t) . n <= A
    # (v + t) . -n <= A
    # where A = s * sqrt(3)/2.
    # This implies:
    # max_v ((v+t).n) - min_v ((v+t).n) <= 2A ? No.
    # The condition is that the entire set is contained in the hexagon.
    # The hexagon is the intersection of 6 strips: -A <= x . n_k <= A.
    # So we need max_v ((v+t).n_k) <= A and min_v ((v+t).n_k) >= -A for all k.
    # This is equivalent to:
    # max_v (v.n_k) + t.n_k <= A
    # min_v (v.n_k) + t.n_k >= -A
    # Let M_k = max_v (v.n_k) and m_k = min_v (v.n_k).
    # We need:
    # t.n_k <= A - M_k
    # t.n_k >= -A - m_k
    # So we need: -A - m_k <= t.n_k <= A - M_k
    # This requires: -A - m_k <= A - M_k  =>  M_k - m_k <= 2A  =>  A >= (M_k - m_k)/2.
    # So A must be at least half the width in every direction.
    # Let A_min = max_k (M_k - m_k)/2.
    # If we set A = A_min, is there a t that satisfies the inequalities?
    # We need to check if the intersection of the strips for t is non-empty.
    # The constraints on t are linear.
    # If we set A slightly larger than A_min, we have a feasible region for t.
    # The minimum s corresponds to the minimum A such that a feasible t exists.
    # This minimum A is exactly A_min?
    # Let's check. If A = A_min, the constraints become tight for the direction k* that achieves the max width.
    # For that direction, t.n_k* must be exactly (M_k* - m_k*)/2 - M_k* = -m_k* - M_k* / 2?
    # Range for t.n_k is [ -A - m_k, A - M_k ].
    # Width of this interval is 2A - (M_k - m_k).
    # If A = A_min, the width is 0 for the critical direction. So t.n_k* is fixed.
    # For other directions, the width is positive.
    # We need to ensure that the fixed value for t.n_k* is consistent with the other constraints.
    # This is a linear feasibility problem.
    # However, due to symmetry of the hexagon, usually the "center" of the cluster (centroid) is a good guess for t.
    # Or, we can just compute the bounding box of the vertices and center the hexagon on that.
    
    # Let's compute the optimal center t.
    # We can use scipy.optimize to find t that minimizes s(t) = max_k ( max_v((v+t).n_k) - min_v((v+t).n_k) )?
    # No, s is determined by the max projection from origin.
    # Let's just shift the centers so that the centroid of all vertices is at (0,0).
    # This is a good heuristic.
    
    all_verts = []
    for c in final_centers:
        all_verts.append(c + v_offsets)
    all_verts = np.vstack(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    # Shift centers by -centroid
    final_centers = final_centers - centroid
    final_s = calc_s(final_centers)
    
    # It's possible that shifting to centroid increases s if the shape is asymmetric?
    # No, s is the minimal enclosing hexagon size. Shifting doesn't change the shape.
    # But calc_s assumes the hexagon is at (0,0).
    # If I shift the points, the max projection from (0,0) changes.
    # The minimal enclosing hexagon centered at (0,0) might be larger than the minimal enclosing hexagon 
    # centered at the optimal t.
    # So I should find the optimal t.
    
    # Let's find optimal t by minimizing s(t).
    # s(t) = max_k ( max_v ((v+t).n_k) * 2/sqrt3 )? No.
    # s(t) is defined by the constraint that all vertices are inside.
    # For a given t, the required s is max_k ( max_v ((v+t).n_k) ) * 2/sqrt3 ?
    # No, it's max( max_v ((v+t).n_k), max_v ((v+t).(-n_k)) ) ...
    # Actually, the hexagon is symmetric.
    # s(t) = (2/sqrt3) * max( max_k ( max_v ((v+t).n_k) ), max_k ( max_v ((v+t).(-n_k)) ) )
    # Since -n_k is also in the set of normals, this is just:
    # s(t) = (2/sqrt3) * max_k ( max_v ((v+t).n_k) )
    
    # We want to minimize this over t.
    # This is a convex optimization problem (min of max of linear functions).
    # We can use scipy.optimize.minimize.
    
    from scipy.optimize import minimize
    
    def s_of_t(t):
        t = np.array(t)
        max_proj = 0
        for c in final_centers:
            verts = c + v_offsets
            shifted_verts = verts + t
            projs = np.einsum('ij,k->ik', shifted_verts, normals)
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        return max_proj * 2.0 / sqrt3

    # Initial guess: 0
    res = minimize(s_of_t, np.array([0.0, 0.0]), method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    
    if res.success:
        optimal_t = res.x
        final_centers = final_centers + optimal_t
        final_s = res.fun
    else:
        # Fallback to centroid
        final_centers = final_centers - centroid
        final_s = calc_s(final_centers)

    inner_ret = np.zeros((12, 3))
    for i in range(12):
        inner_ret[i, 0] = final_centers[i, 0]
        inner_ret[i, 1] = final_centers[i, 1]
        inner_ret[i, 2] = 0.0
        
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, final_s

# EVOLVE-BLOCK-END
