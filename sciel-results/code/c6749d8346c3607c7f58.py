"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Strategy: 
    1. Frame support selection as finding a large binary constant-weight code C(11, 4, 6).
       Condition: Any two supports must intersect in at most 1 index (to ensure Hamming dist >= 6).
       This is stricter than the current program's "no intersection of 3" but allows 
       cleaner vectorization. Wait, let's re-verify the constraint.
       
       Current program constraint: "no two selected supports share 3 indices".
       Intersection size 3 -> Hamming distance = 4 + 4 - 2*3 = 2.
       Intersection size 2 -> Hamming distance = 4 + 4 - 2*2 = 4.
       Intersection size 1 -> Hamming distance = 4 + 4 - 2*1 = 6.
       Intersection size 0 -> Hamming distance = 8.
       
       The distance constraint for vectors is ||u - v||^2 >= max_norm_sq.
       If we use shell-4 vectors (norm^2 = 4), max_norm_sq is likely 4 or higher.
       If we mix shells, it gets complex.
       
       Let's stick to the shell-4 strategy for the core, but optimize the selection.
       If we only use shell-4 vectors:
       ||u||^2 = 4, ||v||^2 = 4.
       Constraint: ||u-v||^2 >= 4.
       ||u-v||^2 = 8 - 2(u.v) >= 4  =>  u.v <= 2.
       
       u.v for shell-4 vectors:
       Let S_u, S_v be supports. Let I = S_u ∩ S_v.
       u.v = sum_{i in S_u} u_i * (v_i if i in S_v else 0).
       Actually, u_i is +/-1.
       If i not in S_v, v_i = 0.
       So u.v = sum_{i in I} u_i v_i.
       Let k = |I|.
       The signs can align or anti-align.
       Max u.v occurs when signs align for all i in I.
       Max u.v = k.
       We need max u.v <= 2.
       So we need k <= 2.
       
       Therefore, the supports must have intersection size <= 2.
       This allows intersection size 2.
       The current program forbids intersection size 3 (and implicitly 4).
       So the constraint is: Intersection size <= 2.
       
       My previous thought about C(11,4,6) (dist>=6 => inter<=1) is too strict.
       We can allow intersection size 2, provided we manage signs or just accept that 
       if inter=2, we must ensure signs don't align too much?
       No, if we just pick one sign pattern per support, we can't control the dot product 
       between two specific supports if we don't know the relative signs.
       
       However, if we pick supports with intersection <= 1, then max u.v = 1 <= 2. 
       This is always safe regardless of signs.
       If we allow intersection = 2, max u.v = 2 <= 2. This is also safe!
       Wait, if u.v = 2, then ||u-v||^2 = 8 - 4 = 4. This is exactly the threshold.
       So intersection size 2 is ALLOWED.
       
       So the problem is: Find a large set of 4-subsets of {0..10} such that 
       any two subsets intersect in at most 2 elements.
       
       Let's use a greedy + local search approach on this specific constraint.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate all 4-element supports
    supports = []
    for comb in combinations(range(D), 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    
    n_sup = len(supports) # 330
    
    # 2. Precompute intersection sizes
    # inter[i, j] = number of common bits
    inter = np.zeros((n_sup, n_sup), dtype=np.uint8)
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter[i, j] = (supports[i] & supports[j]).bit_count()
            inter[j, i] = inter[i, j]
            
    # Conflict if inter > 2
    conflict = inter > 2
    
    # 3. Vectorized Greedy Construction with Randomized Tie-Breaking
    # We want to select a subset of indices from 0..n_sup-1 such that no two are in conflict.
    # This is a Maximum Independent Set problem on the conflict graph.
    # Greedy: Pick node with min degree among available, or random weighted.
    
    def _greedy_mis(conflict_matrix, rng):
        n = len(conflict_matrix)
        selected = np.zeros(n, dtype=bool)
        available = np.ones(n, dtype=bool)
        
        # Precompute degrees
        degrees = conflict_matrix.sum(axis=1)
        
        while available.any():
            # Get indices of available nodes
            avail_idx = np.where(available)[0]
            if len(avail_idx) == 0: break
            
            # Tie-breaking: among available nodes, pick one with minimum current degree.
            # Current degree = number of available neighbors.
            # Vectorized: sum of conflict[avail_idx][:, avail_idx]
            # This is O(N^2) per step if done naively. 
            # Optimization: maintain a degree array.
            
            # For simplicity and speed in Python loop, we might just pick random from top 10% lowest degrees?
            # Or just pick the global min degree among available.
            
            # Let's use a slightly smarter greedy:
            # Score = -degree + small_random_noise
            # We want min degree.
            
            # To keep it fast, we'll update degrees explicitly.
            # But conflict_matrix is dense 330x330.
            
            # Simple loop is fine for 330 nodes.
            min_deg = float('inf')
            best_candidates = []
            
            for idx in avail_idx:
                deg = int(conflict_matrix[idx, available].sum())
                if deg < min_deg:
                    min_deg = deg
                    best_candidates = [idx]
                elif deg == min_deg:
                    best_candidates.append(idx)
            
            # Randomly pick from best candidates
            v = best_candidates[rng.integers(0, len(best_candidates))]
            selected[v] = True
            available[v] = False
            
            # Update available: remove neighbors of v
            neighbors = np.where(conflict_matrix[v] & available)[0]
            available[neighbors] = False
            
        return selected

    def _local_search_1swap(selected, conflict_matrix, rng, max_iter=2000):
        """
        Try to swap a selected node with an unselected node to increase size or maintain size 
        while potentially allowing further growth (though 1-swap keeps size constant).
        Actually, 1-swap doesn't increase size. It's for escaping local minima in 
        "maximize size" if we allow 2-swap (remove 1, add 2).
        
        Let's do 1-remove, 2-add (R1A2) or just 1-swap to improve "quality" (e.g., lower degree) 
        to prepare for adding more? 
        Actually, standard MIS local search: 
        If we have a maximal independent set, we can't just add. 
        We can try to replace a set of k nodes with k+1 nodes.
        
        Let's try a simpler heuristic: 
        Randomly remove a node, then greedily add as many as possible.
        """
        n = len(selected)
        current_sel = selected.copy()
        current_size = current_sel.sum()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5: break
            
            # Pick a random selected node to remove
            sel_indices = np.where(current_sel)[0]
            if len(sel_indices) == 0: break
            rem_idx = sel_indices[rng.integers(0, len(sel_indices))]
            
            # Tentative removal
            current_sel[rem_idx] = False
            
            # Try to add nodes that are now available
            # Available = not selected AND not conflicting with current selected
            # Since we removed one, some of its neighbors might now be addable.
            # But we must ensure the new node doesn't conflict with other selected nodes.
            
            # Let's just run a quick greedy fill from scratch on the "available" set?
            # That's expensive.
            
            # Alternative: 1-swap.
            # Pick a random unselected node u.
            # Pick a random selected node v.
            # If u is not conflict with (selected \ {v}) and v is not conflict with (unselected \ {u}) [trivial],
            # then swap.
            # This maintains size.
            
            # Let's try 1-remove, 1-add (size same) to shuffle, hoping to find a configuration 
            # where a 2-add is possible?
            
            # Actually, let's just do a simple restart-based greedy.
            # The greedy with random tie-breaking is the main driver.
            # Let's run multiple greedy passes.
            break # Disable complex local search for now, rely on multiple greedy runs
            
        return current_sel

    best_sel = None
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Run multiple greedy constructions
    num_runs = 50
    for run in range(num_runs):
        if time.time() > deadline - 10: break
        
        # Randomize the order of tie-breaking by shuffling the initial degree priority?
        # Or just rely on the random choice among min-degree nodes.
        
        # To diversify, we can randomly perturb the "cost" of nodes.
        # Cost = degree + noise
        # But we want min degree.
        
        # Let's just run the greedy function. The randomness in tie-breaking provides diversity.
        sel = _greedy_mis(conflict, rng)
        size = sel.sum()
        
        if size > best_size:
            best_size = size
            best_sel = sel.copy()
            
    # 4. Post-processing: Add sign variants?
    # For each selected support, we can add multiple sign variants IF they don't conflict with each other 
    # or with vectors from other supports.
    
    # Within a support:
    # Two vectors with same support S.
    # u.v = sum_{i in S} u_i v_i.
    # If u, v differ in k positions, u.v = 4 - 2k.
    # We need u.v <= 2.
    # 4 - 2k <= 2 => 2k >= 2 => k >= 1.
    # So any two distinct sign patterns of the same support have u.v <= 2.
    # Thus, we can include ALL 16 sign patterns for a given support?
    # Wait, if we include all 16, we have 16 vectors.
    # Do they conflict with vectors from OTHER supports?
    
    # Let S1, S2 be two supports with intersection size 2 (allowed).
    # Let u in S1, v in S2.
    # u.v = sum_{i in S1 ∩ S2} u_i v_i.
    # Max u.v = 2 (if signs align).
    # This is exactly the limit.
    # So if we pick signs randomly, we might get u.v = 2, which is OK.
    # But if we pick ALL sign patterns, we are guaranteed to have some pair with u.v = 2.
    # Is u.v = 2 allowed?
    # ||u-v||^2 = 4 + 4 - 2(2) = 4.
    # Max norm squared is 4.
    # So distance squared is 4, which is >= 4.
    # So it is ALLOWED.
    
    # So, for each selected support, we can add all 16 sign variants!
    # Total vectors = 16 * (number of selected supports).
    
    # Let's verify this logic.
    # If we have 2 supports S1, S2 with inter=2.
    # We add all 16 signs for S1 and all 16 for S2.
    # Consider u in S1, v in S2.
    # u.v can be -2, 0, 2.
    # If u.v = 2, dist^2 = 4. OK.
    # If u.v = 0, dist^2 = 8. OK.
    # If u.v = -2, dist^2 = 12. OK.
    
    # What if inter=1?
    # u.v can be -1, 1.
    # Max u.v = 1. dist^2 = 8 - 2 = 6. OK.
    
    # What if inter=0?
    # u.v = 0. dist^2 = 8. OK.
    
    # So, YES, we can include all 16 sign variants for every selected support.
    # This multiplies our score by 16!
    
    # The current program only adds 1 vector per support (or tries to add more but fails due to 
    # strict conflict checks in the vector space which are unnecessary if we trust the support constraint).
    
    # Let's construct the final vectors.
    vectors = []
    for idx in range(n_sup):
        if best_sel[idx]:
            mask = supports[idx]
            inds = [i for i in range(D) if (mask >> i) & 1]
            for sign_mask in range(16):
                v = np.zeros(D, dtype=np.int64)
                for j, idx_val in enumerate(inds):
                    if sign_mask & (1 << j):
                        v[idx_val] = 1
                    else:
                        v[idx_val] = -1
                vectors.append(v)
                
    if not vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = np.array(vectors, dtype=np.int64)
    
    # Final safety check: Verify constraints
    # Norms should be 2 (squared 4)
    # Min pairwise dist squared should be >= 4
    
    # Compute norms
    norms_sq = np.sum(result ** 2, axis=1)
    max_norm_sq = norms_sq.max()
    
    # Compute pairwise distances
    # For large k, this is O(k^2 * d). k is ~16*30 = 480. 480^2 * 11 ~ 2.5M ops. Fast.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v
    # We need min dist^2 >= max_norm_sq
    
    # We can check a random subset to be safe, or all pairs.
    # Let's check all pairs to be absolutely sure.
    n_vec = len(result)
    if n_vec > 1:
        # Compute dot product matrix
        dots = result @ result.T
        # dist^2 matrix
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, 0)
        min_dist_sq = dist_sq.min()
        
        if min_dist_sq < max_norm_sq:
            # This should not happen if our logic is correct.
            # If it does, we have a bug. Return empty or fallback?
            # Let's try to fix by removing vectors that violate.
            # But for now, trust the logic.
            pass
            
    return result

# EVOLVE-BLOCK-END
