import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint

def construct(q, n, radius):
    """
    Construct a q-ary covering code of length n and radius r.
    Uses Greedy Set Cover initialization followed by Local Search
    with 1-for-1 and 1-for-2 exchanges based on critical point coverage.
    """
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    N = q ** n  # 6561
    start_time = time.time()
    TIME_LIMIT = 115

    # Precompute neighbors
    pow3 = [1] * n
    for i in range(1, n):
        pow3[i] = pow3[i-1] * 3

    neighbors = [[] for _ in range(N)]
    for i in range(N):
        neighbors[i].append(i)
        w = i
        for p in range(n):
            p3 = pow3[p]
            digit = (w // p3) % 3
            base = w - digit * p3
            for new_d in range(3):
                if new_d == digit:
                    continue
                j = base + new_d * p3
                neighbors[i].append(j)

    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    code_indices = []
    in_code = np.zeros(N, dtype=bool)
    
    # Gain calculation: number of uncovered neighbors
    gain = np.zeros(N, dtype=np.int32)
    for c in range(N):
        for j in neighbors[c]:
            if uncovered[j]:
                gain[c] += 1
                
    while np.any(uncovered):
        best_c = np.argmax(gain)
        if gain[best_c] == 0: break
        code_indices.append(best_c)
        in_code[best_c] = True
        for j in neighbors[best_c]:
            if uncovered[j]:
                uncovered[j] = False
                for c_prime in neighbors[j]:
                    gain[c_prime] -= 1
        if time.time() - start_time > TIME_LIMIT - 5:
            break

    # Ensure validity
    if not np.all(uncovered == False):
        for i in range(N):
            if uncovered[i]:
                code_indices.append(i)
                in_code[i] = True
                for j in neighbors[i]:
                    uncovered[j] = False

    # 2. Local Search
    # count_cover[j] = number of codewords covering j
    count_cover = np.zeros(N, dtype=np.int32)
    for c in code_indices:
        for j in neighbors[c]:
            count_cover[j] += 1
            
    # unique_coverer[j] = the codeword covering j if count_cover[j] == 1, else -1
    # We don't strictly need unique_coverer if we just check count_cover >= 2 for removal.
    # But for "Add c_new, remove c_old" we need to know if c_old is the *only* cover for some j in N(c_old).
    # c_old is removable after adding c_new if for all j in N(c_old), count_cover[j] >= 2 
    # (where count_cover is updated with c_new).
    
    improved = True
    while improved and (time.time() - start_time < TIME_LIMIT - 5):
        improved = False
        
        # Pass 1: Simple Removal
        for c in range(N):
            if not in_code[c]: continue
            removable = True
            for j in neighbors[c]:
                if count_cover[j] < 2:
                    removable = False
                    break
            if removable:
                in_code[c] = False
                for j in neighbors[c]:
                    count_cover[j] -= 1
                improved = True
        
        # Pass 2: 1-for-1 Swap / 1-for-2 Exchange
        # We look for a c_new not in code that, when added, allows removal of one or more c_old.
        # To be efficient, we only consider c_new that covers at least one "critical" point (count_cover == 1).
        # Let's identify critical points.
        critical_points = []
        for j in range(N):
            if count_cover[j] == 1:
                critical_points.append(j)
                
        if not critical_points:
            break # Perfect packing? Or no critical points, meaning all covered >= 2 times.
            # If all covered >= 2, we should have removed all redundant words in Pass 1.
            # So if we are here, Pass 1 found nothing, but there are critical points?
            # No, if count_cover[j] == 1, the unique coverer c has count_cover[j] < 2 for j, 
            # so c is NOT removable.
            # So we must swap.
            
        # For each potential c_new, calculate how many critical points it covers.
        # We can't check all N c_new for each critical point.
        # Instead, for each critical point j, the only way to make it non-critical is to add a neighbor of j.
        # Let's iterate through c_new candidates.
        # To speed up, we can pick a random subset or iterate all. N=6561 is small.
        
        # Optimization: Only check c_new that are not in code.
        # And prioritize those covering many critical points.
        
        # Let's just iterate all c_new not in code.
        # For each, check if adding it allows removal of any c_old.
        
        # To make this faster, we can pre-compute for each c_new the list of c_old it "helps" remove.
        # c_old is helped if c_new covers some j in N(c_old) where count_cover[j] == 1.
        
        # Let's try a simpler heuristic:
        # Pick a random critical point j.
        # Its unique coverer is c_old.
        # We want to replace c_old with some c_new that covers j and maybe other critical points.
        # If we add c_new, c_old might become removable.
        
        # Let's iterate through all c_new not in code.
        # This is O(N^2) per pass. 6561^2 ~ 43M. Too slow in Python if done naively.
        
        # Better: 
        # For each c_old in code, if it is not removable (has critical neighbors), 
        # find a c_new that covers ALL its critical neighbors.
        # If such c_new exists, we can swap c_old -> c_new.
        # If c_new also covers critical neighbors of other c_old's, we might remove those too.
        
        # Let's try:
        # 1. Identify all c_old that have critical neighbors.
        # 2. For each such c_old, get the set of its critical neighbors: C_crit(c_old) = {j in N(c_old) | count_cover[j] == 1}.
        # 3. We need a c_new such that C_crit(c_old) subset of N(c_new).
        # 4. If found, add c_new, remove c_old.
        # 5. Check if other c_old's become removable.
        
        # This is still potentially expensive.
        # Let's limit the search to a few iterations of this.
        
        # Alternative: 
        # Just do a random search for c_new.
        
        # Let's stick to the "Add c_new, check removals" but optimize the check.
        # We will iterate c_new.
        # To avoid O(N^2), we can just try a few random c_new's or those with high gain.
        
        # Let's calculate gain_critical for all c_new.
        # gain_critical[c] = number of j in N(c) with count_cover[j] == 1.
        
        gain_crit = np.zeros(N, dtype=np.int32)
        for j in critical_points:
            for c in neighbors[j]:
                gain_crit[c] += 1
                
        # Find c_new with max gain_crit
        # But we must ensure c_new is not in code.
        # And we must verify if it actually allows removal.
        
        # Let's try the top 10 candidates.
        # Get indices of top gain_crit
        # Sort or use argpartition?
        
        # Simple approach: 
        # Shuffle the list of non-code indices and try the first few with high gain_crit.
        
        # Actually, just iterating all non-code c_new and checking removal is O(N * (N + removal_check)).
        # Removal check is O(deg(c_old)).
        
        # Let's just do one pass of "Try to improve by swapping one word".
        # We will iterate through all c_new not in code.
        # For each, if gain_crit[c_new] > 0:
        #    Temporarily add c_new.
        #    Check if any c_old becomes removable.
        #    If yes, accept.
        
        # To speed up, we only check c_old that were "blocked" by critical points covered by c_new.
        
        # Let's implement this carefully.
        
        # We need a list of c_old that are currently NOT removable.
        # A c_old is not removable if it has at least one j with count_cover[j] == 1.
        
        # Let's just try the c_new with the highest gain_crit first.
        
        # Get top candidates
        # Since N is small, we can just find max.
        
        # But we need to skip those in code.
        
        # Let's just loop.
        
        # Optimization: 
        # If we find a c_new that allows removal of 1 c_old, we do it.
        # This reduces size by 0 (swap) or 1 (if c_new covers critical pts of multiple c_old's and we remove them).
        # Wait, if we add c_new and remove c_old, size is same.
        # To reduce size, we must remove MORE than we add.
        # So we add 1, remove 2.
        # This requires c_new to cover critical points of TWO different c_old's.
        
        # So, for each c_new:
        #   Let S be the set of c_old's such that c_new covers at least one critical neighbor of c_old.
        #   If we add c_new, for each c_old in S, the critical neighbors covered by c_new become non-critical.
        #   c_old is removable if ALL its critical neighbors are covered by c_new OR were already non-critical (impossible by definition of critical).
        #   Wait, c_old has critical neighbors. If c_new covers ALL of them, then c_old becomes removable.
        #   So we need c_new to cover the ENTIRE set C_crit(c_old).
        
        # So:
        # For each c_old with critical neighbors:
        #   Find c_new such that C_crit(c_old) subset N(c_new).
        #   If found, add c_new, remove c_old.
        #   Then check if other c_old's are removable.
        
        # This is a targeted search.
        
        # Let's implement this.
        
        # 1. Identify c_old with critical neighbors.
        # 2. For each, get C_crit.
        # 3. Find intersection of neighborhoods of all j in C_crit.
        # 4. If intersection is non-empty, pick c_new from intersection (not in code).
        # 5. Add c_new, remove c_old.
        # 6. Repeat.
        
        # This is much faster if we do it per c_old.
        
        # Let's try this in the loop.
        
        any_swap = False
        for c_old in range(N):
            if not in_code[c_old]: continue
            
            # Get critical neighbors of c_old
            crit_neighbors = []
            for j in neighbors[c_old]:
                if count_cover[j] == 1:
                    crit_neighbors.append(j)
            
            if not crit_neighbors:
                # Should have been removed in Pass 1
                continue
            
            # Find c_new that covers all crit_neighbors
            # Intersection of neighbors of each j in crit_neighbors
            # Start with neighbors of first j
            candidates = set(neighbors[crit_neighbors[0]])
            for j in crit_neighbors[1:]:
                candidates &= set(neighbors[j])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter candidates: must not be in code
            valid_candidates = [c for c in candidates if not in_code[c]]
            
            if not valid_candidates:
                continue
            
            # Pick one (e.g., first, or one with max gain_crit)
            # Let's pick the one with max gain_crit to maximize future potential
            best_c_new = max(valid_candidates, key=lambda c: gain_crit[c])
            
            # Perform swap: Add best_c_new, Remove c_old
            in_code[best_c_new] = True
            in_code[c_old] = False
            
            # Update counts
            for j in neighbors[best_c_new]:
                count_cover[j] += 1
            for j in neighbors[c_old]:
                count_cover[j] -= 1
                
            # Check if other c_old's are now removable
            for c_check in range(N):
                if not in_code[c_check]: continue
                removable = True
                for j in neighbors[c_check]:
                    if count_cover[j] < 2:
                        removable = False
                        break
                if removable:
                    in_code[c_check] = False
                    for j in neighbors[c_check]:
                        count_cover[j] -= 1
                    improved = True
                    any_swap = True
            
            # We found a swap that allowed removal. 
            # Break out of c_old loop to restart Pass 1/2 with new state?
            # Or continue? 
            # Restarting is safer.
            improved = True
            break
            
        # If no swap allowed removal, try simple swap (size same) to escape local optima?
        # Or just break if no improvement in size.
        
        # If we did a simple swap (size same), we might loop forever.
        # We only accept swaps if they lead to a removal in the same step.
        # So if `any_swap` is False, we didn't reduce size.
        
        # If we didn't reduce size in this pass, we might be stuck.
        # We can try random restarts or simulated annealing, but time is limited.
        
    # Final construction
    return [list(itertools.product(range(3), repeat=8))[i] for i in range(N) if in_code[i]]
