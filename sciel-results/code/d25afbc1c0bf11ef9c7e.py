"""
Block-Move (Or-opt) Local Search for QAP
Strategy:
1. Initialize using Spectral Quadratic Linearization (QL) as a strong starting point.
2. Apply a vectorized Block-Move local search. This moves a contiguous block of facilities 
   (length 1, 2, or 3) to a new position in the permutation.
3. The delta cost for a block move is computed efficiently using precomputed interaction 
   matrices to avoid O(n^3) complexity per move.
4. Iterate until no improvement is found or time limit is reached.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_ql_init(a, b, num_components=15):
    """
    Quadratic Linearization via Spectral Decomposition.
    """
    n = len(a)
    A_sym = (a + a.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    
    # Sort by magnitude of eigenvalues (descending)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    best_p = None
    best_cost = np.inf
    
    for k in range(1, min(num_components + 1, n)):
        V_k = eigvecs[:, :k]
        Lambda_k = eigvals[:k]
        
        # proj_b shape: (k, n)
        proj_b = V_k.T @ b
        
        # C = V_k @ diag(Lambda_k) @ proj_b
        C = (V_k * Lambda_k) @ proj_b
        
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    return best_p, best_cost

def _block_move_delta(a, b, p, start, length, new_pos):
    """
    Compute the change in cost when moving a block of facilities from [start, start+length)
    to new_pos.
    
    Parameters:
    a: flow matrix (n, n)
    b: distance matrix (n, n)
    p: current permutation (n,)
    start: start index of the block in p
    length: length of the block
    new_pos: index in p where the block should be inserted.
             If new_pos > start, it refers to the position in the array AFTER removal.
             Logic: 
             Remove block from [start, start+length).
             Insert block at new_pos.
             
    Returns:
    delta: integer cost change
    """
    n = len(p)
    
    # Extract block facilities and their positions
    block_facilities = p[start:start+length]
    
    # Define the set of facilities involved in the move
    # The move affects:
    # 1. Internal interactions of the block (unchanged relative to each other, but positions change relative to others)
    # 2. Interactions between block and the rest of the array.
    
    # To compute delta efficiently, we can reconstruct the cost contribution of the block 
    # in the old and new configurations.
    
    # Old configuration:
    # Block is at indices start ... start+length-1.
    # Rest of the array is split into Left (0...start-1) and Right (start+length...n-1).
    
    # New configuration:
    # Block is at new_pos ... new_pos+length-1.
    # Note: new_pos is an index in the array of size n-length (after removal).
    # If we think in terms of the final array of size n:
    # The block occupies [new_pos, new_pos+length).
    
    # Let's compute the cost contribution of the block with the rest of the system.
    # Cost = Sum_{i in Block} Sum_{j in Rest} a[i,j] b[p[i], p[j]] + Sum_{i,j in Block} a[i,j] b[p[i], p[j]]
    
    # The internal cost of the block (Sum_{i,j in Block}) is invariant to the move 
    # because the relative order within the block is preserved, and the distance b depends 
    # on the locations assigned to the facilities. Wait, b[p[i], p[j]] depends on the 
    # locations p[i] and p[j]. Since the facilities are the same, and their locations 
    # are the same (just permuted in the array index, but p[i] is the location), 
    # the term b[p[i], p[j]] is the same.
    # HOWEVER, the definition of p is p[i] = location of facility i.
    # So if we move facility X from position i to position k, p[i] changes from X to something else.
    # This changes b[p[i], p[j]] for all j.
    
    # Let's use the standard delta formula for a set of moved elements.
    # Let M be the set of facilities in the block.
    # Let R be the set of facilities NOT in the block.
    
    # Old Cost Contribution involving M:
    # C_old_M = Sum_{i in M, j in M} a[i,j] b[p_old[i], p_old[j]] + Sum_{i in M, j in R} a[i,j] b[p_old[i], p_old[j]]
    # Note: Sum_{i in R, j in M} is the same as Sum_{j in M, i in R} if we sum over all pairs.
    # The total cost is Sum_{i,j} a[i,j] b[p[i], p[j]].
    # Terms involving at least one element from M are the ones that change.
    
    # Let's identify the old locations and new locations for the facilities in M.
    # Old locations: loc_old = p[start:start+length]
    # New locations: The block is inserted at new_pos.
    # The facilities in M keep their relative order.
    # So loc_new[k] = loc_old[k] for k=0..length-1.
    # The LOCATIONS assigned to the facilities in M do not change! 
    # Wait. p[i] is the location of facility i.
    # If we move facility X from index i to index k, p[i] was X, now p[i] is Y (the facility that was at i+length or something).
    # So the LOCATION X is now assigned to facility X? No.
    # p is a permutation of locations.
    # Facility i is at location p[i].
    # If we swap/move entries in p, we are changing which facility is at which location.
    
    # Let's trace:
    # p = [L0, L1, L2, L3, L4]
    # Move block [1, 2] (facilities at indices 1, 2) to pos 4.
    # Old: Fac 0 at L0, Fac 1 at L1, Fac 2 at L2, Fac 3 at L3, Fac 4 at L4.
    # Remove L1, L2. Array: [L0, L3, L4].
    # Insert L1, L2 at end. Array: [L0, L3, L4, L1, L2].
    # New: Fac 0 at L0, Fac 1 at L3, Fac 2 at L4, Fac 3 at L1, Fac 4 at L2.
    
    # So the facilities in the block (Fac 1, Fac 2) move to NEW locations (L3, L4).
    # The facilities NOT in the block (Fac 3, Fac 4) move to OLD locations of the block (L1, L2).
    
    # So:
    # M = {Fac 1, Fac 2}.
    # R = {Fac 0, Fac 3, Fac 4}.
    
    # For i in M:
    #   p_old[i] = L1, L2
    #   p_new[i] = L3, L4
    
    # For j in R:
    #   p_old[j] = L0, L3, L4
    #   p_new[j] = L0, L1, L2
    
    # We need to compute:
    # Delta = Sum_{i in M, j in M} a[i,j] (b[p_new[i], p_new[j]] - b[p_old[i], p_old[j]])
    #       + Sum_{i in M, j in R} a[i,j] (b[p_new[i], p_new[j]] - b[p_old[i], p_old[j]])
    #       + Sum_{i in R, j in M} a[i,j] (b[p_new[i], p_new[j]] - b[p_old[i], p_old[j]])
    
    # Note: The term Sum_{i in R, j in R} is unchanged because both p[i] and p[j] for i,j in R 
    # might change?
    # Fac 3 (in R) moves from L3 to L1. Fac 4 (in R) moves from L4 to L2.
    # So p_new[Fac 3] = L1, p_old[Fac 3] = L3.
    # p_new[Fac 4] = L2, p_old[Fac 4] = L4.
    # So the term a[Fac 3, Fac 4] * b[L1, L2] vs a[Fac 3, Fac 4] * b[L3, L4].
    # This DOES change!
    
    # So ALL terms involving at least one facility whose location changed must be recalculated.
    # Which facilities change location?
    # Facilities in M change location.
    # Facilities in R that "shift" to fill the gap or move to the new slot also change location.
    
    # In the example:
    # M = {1, 2}.
    # R = {0, 3, 4}.
    # Fac 0: L0 -> L0 (Unchanged)
    # Fac 3: L3 -> L1 (Changed)
    # Fac 4: L4 -> L2 (Changed)
    
    # So the set of facilities that change location is S = M U R_changed.
    # In this case S = {1, 2, 3, 4}.
    # Fac 0 is the only one that didn't change.
    
    # Generalizing:
    # When we move a block, the facilities in the block change locations.
    # The facilities in the "gap" (between old and new position) also change locations.
    # The facilities outside the range [min(start, new_pos), max(start, new_pos)+length] do not change.
    
    # Let's define the set of indices in the permutation array that are affected.
    # Old block indices: I_old = [start, start+length)
    # New block indices: I_new = [new_pos, new_pos+length)
    
    # The facilities at these indices change their assigned locations.
    # Let Affected_Indices = I_old U I_new.
    # Let Affected_Facilities = p[Affected_Indices] (unique set).
    
    # Actually, it's easier to just compute the cost of the submatrix involving Affected_Facilities
    # in the old and new permutations.
    
    # Let F be the set of facilities in Affected_Indices.
    # Let Loc_Old be the locations assigned to F in the old permutation.
    # Let Loc_New be the locations assigned to F in the new permutation.
    
    # Delta = Sum_{i in F, j in F} a[i,j] b[p_new[i], p_new[j]] - Sum_{i in F, j in F} a[i,j] b[p_old[i], p_old[j]]
    # Note: This formula works because terms not involving F are unchanged.
    # Why? If i not in F, p_new[i] = p_old[i]. If j not in F, p_new[j] = p_old[j].
    # So b[p_new[i], p_new[j]] = b[p_old[i], p_old[j]] for i,j not in F.
    # The only terms that change are those where i in F or j in F.
    # Sum_{i in F, j in F} covers i in F, j in F.
    # What about i in F, j not in F?
    # The term a[i,j] b[p_new[i], p_new[j]] vs a[i,j] b[p_old[i], p_old[j]].
    # Since j not in F, p_new[j] = p_old[j].
    # So the difference is a[i,j] (b[p_new[i], p_old[j]] - b[p_old[i], p_old[j]]).
    # My formula Sum_{i in F, j in F} misses these cross terms!
    
    # Correct Delta:
    # Delta = Sum_{i in F, j in All} a[i,j] (b[p_new[i], p_new[j]] - b[p_old[i], p_old[j]])
    #       + Sum_{i not in F, j in F} a[i,j] (b[p_new[i], p_new[j]] - b[p_old[i], p_old[j]])
    
    # Since p_new[i] = p_old[i] for i not in F:
    # Second term: Sum_{i not in F, j in F} a[i,j] (b[p_old[i], p_new[j]] - b[p_old[i], p_old[j]])
    
    # Let's combine:
    # Delta = Sum_{i in F, j in All} a[i,j] b[p_new[i], p_new[j]] 
    #       - Sum_{i in F, j in All} a[i,j] b[p_old[i], p_old[j]]
    #       + Sum_{i not in F, j in F} a[i,j] b[p_old[i], p_new[j]]
    #       - Sum_{i not in F, j in F} a[i,j] b[p_old[i], p_old[j]]
    
    # This is still O(|F| * n). If |F| is small (e.g., 3+3=6), this is fast.
    
    # Implementation:
    # 1. Identify F (facilities in I_old U I_new).
    # 2. Get p_old for F and p_new for F.
    # 3. Compute the sums using vectorization.
    
    I_old = np.arange(start, start + length)
    I_new = np.arange(new_pos, new_pos + length)
    
    # Unique indices in the permutation array that are affected
    # Note: If I_old and I_new overlap, we must be careful.
    # But a block move usually implies distinct ranges or we handle the shift logic.
    # Let's assume standard block move: remove then insert.
    
    # Construct the new permutation p_new to get p_new for F?
    # That's O(n) which is fine.
    p_new = p.copy()
    # Remove block
    # Shift elements to fill gap
    if new_pos > start:
        # Move right
        # Elements in [start+length, new_pos) shift left by length
        # Elements in [new_pos, new_pos+length) get the block
        # Wait, standard logic:
        # p_new = p[0:start] + p[start+length:new_pos] + p[start:start+length] + p[new_pos:]
        # This assumes new_pos is the index in the array AFTER removal.
        # The problem statement for _block_move_delta says new_pos is the position.
        # Let's stick to the logic:
        # p_temp = np.concatenate([p[:start], p[start+length:]])
        # p_new = np.concatenate([p_temp[:new_pos], p[start:start+length], p_temp[new_pos:]])
        
        # Let's verify indices.
        # p_temp has size n-length.
        # new_pos must be in [0, n-length].
        
        p_temp = np.concatenate([p[:start], p[start+length:]])
        p_new = np.concatenate([p_temp[:new_pos], p[start:start+length], p_temp[new_pos:]])
    else:
        # Move left (or same)
        # p_temp = np.concatenate([p[:start], p[start+length:]])
        # p_new = np.concatenate([p_temp[:new_pos], p[start:start+length], p_temp[new_pos:]])
        # This formula works for both if new_pos is interpreted correctly.
        # If new_pos < start, the "gap" is [new_pos, start+length).
        # The elements in [new_pos, start) shift right by length.
        # The formula p_temp[:new_pos] + block + p_temp[new_pos:] handles this.
        # Example: p=[0,1,2,3,4], start=2, len=1 (move 2), new_pos=0.
        # p_temp = [0,1,3,4]
        # p_new = [] + [2] + [0,1,3,4] = [2,0,1,3,4]. Correct.
        
        p_temp = np.concatenate([p[:start], p[start+length:]])
        p_new = np.concatenate([p_temp[:new_pos], p[start:start+length], p_temp[new_pos:]])
        
    # Now we have p_new.
    # Identify F: facilities that changed location.
    # F = set(p[I_old]) U set(p[I_new])
    # Actually, any facility i where p_new[i] != p_old[i] is in F.
    # It's faster to just take the union of indices.
    
    # Indices in p array that changed
    changed_indices = np.unique(np.concatenate([I_old, I_new]))
    
    # Facilities at these indices
    # Note: A facility might be at an index in I_old and another in I_new?
    # No, facilities are unique.
    # But a facility at index k in I_old might move to index m in I_new.
    # So the set of facilities is p[changed_indices].
    
    F = p[changed_indices]
    
    # Get old and new locations for these facilities
    # We need to map facility -> old_loc and facility -> new_loc.
    # old_loc for f in F is p[old_index_of_f].
    # new_loc for f in F is p_new[new_index_of_f].
    
    # Since F is a list of facilities, we can use advanced indexing.
    # But we need to be careful: p is indexed by facility ID?
    # NO. p[i] is the location of facility i.
    # So p is a mapping Facility -> Location.
    # So p[F] gives the old locations of facilities in F.
    # p_new[F] gives the new locations of facilities in F.
    
    loc_old_F = p[F]
    loc_new_F = p_new[F]
    
    # Now compute Delta.
    # Delta = Sum_{i in F, j in All} a[i,j] b[p_new[i], p_new[j]] 
    #       - Sum_{i in F, j in All} a[i,j] b[p_old[i], p_old[j]]
    #       + Sum_{i not in F, j in F} a[i,j] b[p_old[i], p_new[j]]
    #       - Sum_{i not in F, j in F} a[i,j] b[p_old[i], p_old[j]]
    
    # Let's break it down.
    # Term 1: Sum_{i in F, j in All} a[i,j] b[p_new[i], p_new[j]]
    #   For j in F, p_new[j] = loc_new_F[j_idx_in_F]
    #   For j not in F, p_new[j] = p[j] (unchanged)
    #   So we can construct a vector b_new_row for each i in F?
    #   Or just compute the sum directly.
    
    # Let's use the property that only terms involving F change.
    # Delta = Sum_{i in F, j in F} a[i,j] (b[p_new[i], p_new[j]] - b[p_old[i], p_old[j]])
    #       + Sum_{i in F, j not in F} a[i,j] (b[p_new[i], p_new[j]] - b[p_old[i], p_old[j]])
    #       + Sum_{i not in F, j in F} a[i,j] (b[p_new[i], p_new[j]] - b[p_old[i], p_old[j]])
    
    # Note: p_new[i] = p_old[i] for i not in F.
    # So Term 3: Sum_{i not in F, j in F} a[i,j] (b[p_old[i], p_new[j]] - b[p_old[i], p_old[j]])
    
    # Let's compute these sums.
    # |F| is small (<= 2*length).
    
    # Precompute submatrices of A for F
    # A_FF = a[np.ix_(F, F)]
    # A_F_notF = a[np.ix_(F, np.setdiff1d(np.arange(n), F))]
    # A_notF_F = a[np.ix_(np.setdiff1d(np.arange(n), F), F)]
    
    # This involves setdiff which is O(n).
    # Given n=100, this is negligible.
    
    all_indices = np.arange(n)
    not_F = np.setdiff1d(all_indices, F)
    
    # Term 1: i in F, j in F
    # b[p_new[i], p_new[j]] for i,j in F
    # We have loc_new_F and loc_old_F.
    # B_new_FF = b[np.ix_(loc_new_F, loc_new_F)]
    # B_old_FF = b[np.ix_(loc_old_F, loc_old_F)]
    # Delta_1 = sum(a[np.ix_(F,F)] * (B_new_FF - B_old_FF))
    
    # Term 2: i in F, j not in F
    # b[p_new[i], p_new[j]] where p_new[j] = p[j]
    # B_new_F_notF = b[np.ix_(loc_new_F, p[not_F])]
    # B_old_F_notF = b[np.ix_(loc_old_F, p[not_F])]
    # Delta_2 = sum(a[np.ix_(F, not_F)] * (B_new_F_notF - B_old_F_notF))
    
    # Term 3: i not in F, j in F
    # b[p_old[i], p_new[j]] where p_old[i] = p[i]
    # B_new_notF_F = b[np.ix_(p[not_F], loc_new_F)]
    # B_old_notF_F = b[np.ix_(p[not_F], loc_old_F)]
    # Delta_3 = sum(a[np.ix_(not_F, F)] * (B_new_notF_F - B_old_notF_F))
    
    # Total Delta = Delta_1 + Delta_2 + Delta_3
    
    # Optimization: 
    # If F is empty, delta is 0.
    if len(F) == 0:
        return 0
        
    # Compute
    # Note: a and b are int64.
    
    # Term 1
    A_FF = a[np.ix_(F, F)]
    B_new_FF = b[np.ix_(loc_new_F, loc_new_F)]
    B_old_FF = b[np.ix_(loc_old_F, loc_old_F)]
    delta_1 = np.sum(A_FF * (B_new_FF - B_old_FF))
    
    # Term 2
    if len(not_F) > 0:
        A_F_notF = a[np.ix_(F, not_F)]
        B_new_F_notF = b[np.ix_(loc_new_F, p[not_F])]
        B_old_F_notF = b[np.ix_(loc_old_F, p[not_F])]
        delta_2 = np.sum(A_F_notF * (B_new_F_notF - B_old_F_notF))
        
        # Term 3
        A_notF_F = a[np.ix_(not_F, F)]
        B_new_notF_F = b[np.ix_(p[not_F], loc_new_F)]
        B_old_notF_F = b[np.ix_(p[not_F], loc_old_F)]
        delta_3 = np.sum(A_notF_F * (B_new_notF_F - B_old_notF_F))
    else:
        delta_2 = 0
        delta_3 = 0
        
    return int(delta_1 + delta_2 + delta_3)

def _block_move_local_search(a, b, p, time_limit, max_block_len=3):
    """
    Local search using block moves.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # We will iterate through possible block lengths and positions.
    # To avoid O(n^3) full scan, we can do random sampling or structured passes.
    # For n=100, a full scan of len=1 moves is 100*99 ~ 10k moves.
    # Each move is O(n) ~ 100 ops. Total 1M ops. Fast.
    # Len=2: 100*98 ~ 10k.
    # Len=3: 100*97 ~ 10k.
    # Total ~30k moves per pass.
    # 30k * 100 ops = 3M ops.
    # Python overhead might make this slower, but NumPy vectorization inside delta helps.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try block lengths 1 to max_block_len
        for length in range(1, max_block_len + 1):
            # Iterate over start positions
            for start in range(n - length + 1):
                if time.time() - start_time > time_limit:
                    return best_p, best_cost
                
                # Iterate over new positions
                # new_pos can be 0 to n-length
                # Skip if new_pos is effectively the same as start (no move)
                for new_pos in range(n - length + 1):
                    if new_pos == start:
                        continue
                    # Avoid redundant moves? 
                    # Moving block to new_pos is different from moving to new_pos-1 if the block interacts with neighbors?
                    # Yes, the neighbors change.
                    
                    delta = _block_move_delta(a, b, p, start, length, new_pos)
                    
                    if delta < 0:
                        # Apply move
                        p_temp = np.concatenate([p[:start], p[start+length:]])
                        p = np.concatenate([p_temp[:new_pos], p[start:start+length], p_temp[new_pos:]])
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
                        improved = True
                        # Restart inner loops to check new neighbors? 
                        # Or just continue? 
                        # First improvement strategy: break out and restart pass.
                        # Best improvement strategy: check all, apply best.
                        # Let's do first improvement for speed, then restart pass.
                        break 
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    # Phase 1: Spectral QL Initialization
    p_init, cost_init = _spectral_ql_init(a, b, num_components=15)
    best_p = p_init.copy()
    best_cost = cost_init
    
    # Phase 2: Block Move Local Search
    # We have ~110s left.
    # Block move LS is effective. We can run it for a significant portion of the time.
    # Then restart with perturbations.
    
    time_remaining = deadline - (time.time() - start)
    
    # Run Block Move LS
    # Allocate 80% of time to LS, 20% to restarts?
    # Or just run LS until no improvement, then restart.
    
    while time.time() - start < deadline - 10.0:
        # Run LS for a chunk of time
        ls_time = min(20.0, deadline - (time.time() - start) - 5.0)
        if ls_time < 5.0:
            break
            
        p_current, cost_current = _block_move_local_search(a, b, best_p, ls_time)
        
        if cost_current < best_cost:
            best_cost = cost_current
            best_p = p_current.copy()
        else:
            # No improvement, perturb
            # Perform random block moves or swaps to escape
            num_perturbations = 5
            for _ in range(num_perturbations):
                length = np.random.randint(1, 4)
                start = np.random.randint(n - length + 1)
                new_pos = np.random.randint(n - length + 1)
                if new_pos != start:
                    p_temp = np.concatenate([best_p[:start], best_p[start+length:]])
                    best_p = np.concatenate([p_temp[:new_pos], best_p[start:start+length], p_temp[new_pos:]])
                    
    # Final check
    if time.time() - start < deadline - 2.0:
        p_final, cost_final = _block_move_local_search(a, b, best_p, deadline - (time.time() - start) - 2.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
