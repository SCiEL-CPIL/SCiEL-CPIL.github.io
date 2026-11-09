"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 50.0  # safety margin before 60s kill (90s wall-clock)
    
    def elapsed():
        return time.time() - start_time
        
    N_SMALL_END = 128
    N_LARGE_END = 256
    
    small_nums = list(range(2, N_SMALL_END + 1))
    large_nums = list(range(N_SMALL_END + 1, N_LARGE_END + 1))
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Weights
    small_weights = [1.0 / x for x in small_nums]
    large_weights = [1.0 / x for x in large_nums]
    
    # Precompute conflict masks
    # small_conflict_mask[i] : bitmask of small indices conflicting with i
    small_conflict_mask = [0] * n_small
    for i in range(n_small):
        for j in range(i + 1, n_small):
            a, b = small_nums[i], small_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                small_conflict_mask[i] |= (1 << j)
                small_conflict_mask[j] |= (1 << i)
                
    # large_conflict_mask[i] : bitmask of large indices conflicting with small_nums[i]
    small_blocks_large = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(n_large):
            b = large_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                small_blocks_large[i] |= (1 << j)
    
    def sum_large_weights_in_mask(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += large_weights[idx]
            m ^= lsb
        return s

    def get_score(small_mask, blocked_large_mask):
        s_small = 0.0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_small += small_weights[idx]
            m ^= lsb
        s_blocked = sum_large_weights_in_mask(blocked_large_mask)
        s_large = sum(large_weights) - s_blocked
        return s_small + s_large

    def get_blocked_large_mask(small_mask):
        blocked = 0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocked |= small_blocks_large[idx]
            m ^= lsb
        return blocked

    best_score = 0.0
    best_small_mask = 0
    best_blocked_large_mask = 0
    
    def update_best(sm, blm, score):
        nonlocal best_score, best_small_mask, best_blocked_large_mask
        if score > best_score:
            best_score = score
            best_small_mask = sm
            best_blocked_large_mask = blm

    # Bron-Kerbosch with greedy coloring for maximum weight independent set
    # We model this as finding a maximum weight clique in the complement graph
    # where the complement graph has edges between non-conflicting small numbers
    
    # Build complement adjacency (non-conflict) masks for small numbers
    # comp_adj[i] = bitmask of indices j != i where i and j do NOT conflict
    comp_adj = [0] * n_small
    for i in range(n_small):
        for j in range(n_small):
            if i == j:
                continue
            if not (small_conflict_mask[i] & (1 << j)):
                comp_adj[i] |= (1 << j)
    
    # Weight of selecting a subset of small numbers = sum of their weights
    # But we also need to account for blocked large numbers
    # The total score = sum_small_weights - sum_blocked_large_weights + total_large_weight
    # So we want to maximize: sum_small_weights - sum_blocked_large_weights
    # This is not simply the sum of individual weights due to the blocked large terms
    
    # For branch and bound, we need an upper bound on the additional score
    # given a partial selection. The upper bound for adding more small numbers
    # from candidate set P is: sum of (w_i - expected_blocked_contribution) for i in P
    # But blocked contributions overlap, so a simple sum is an overestimate (valid upper bound)
    
    # Actually, let's think differently. We can precompute for each small number i:
    # net_weight[i] = small_weights[i] - sum(large_weights[j] for j in small_blocks_large[i])
    # But this double-counts blocked large numbers when multiple smalls are selected.
    # So net_weight sum is NOT a valid upper bound.
    
    # Better approach: use a simpler upper bound. For any subset S of smalls,
    # score(S) = sum_{i in S} w_i + total_large - sum_{j in blocked(S)} w_large[j]
    # = total_large + sum_{i in S} w_i - sum_{j in blocked(S)} w_large[j]
    # 
    # An upper bound: total_large + sum_{i in S ∪ P} w_i (ignoring blocked penalty)
    # This is valid but loose.
    
    # Tighter: for each candidate i in P, the maximum additional score from adding i
    # is w_i - 0 (if no new large numbers get blocked). But that's optimistic.
    # 
    # Let's use: upper_bound = current_score + sum_{i in P} max(0, w_i)
    # This is valid since blocked penalties can only reduce score.
    
    # Actually, the cleanest approach: we'll do branch and bound on the small numbers,
    # computing exact scores for complete solutions and using the simple upper bound
    # sum of remaining weights as the pruning criterion.
    
    # Bron-Kerbosch with pivot for maximum weight independent set
    # We'll use a recursive branch and bound approach
    
    def bron_kerbosch(P, X, R, current_score, current_blocked):
        """
        P: bitmask of candidate small numbers that can still be added
        X: bitmask of small numbers that were previously removed (for symmetry breaking)
        R: bitmask of selected small numbers
        current_score: score of current selection R
        current_blocked: blocked large mask from R
        """
        if elapsed() > DEADLINE:
            return
        
        # If no candidates, we have a complete solution
        if P == 0:
            update_best(R, current_blocked, current_score)
            return
        
        # Upper bound: current_score + sum of weights of all in P
        # (ignoring blocked penalties, which can only decrease score)
        ub = current_score
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ub += small_weights[idx]
            m ^= lsb
        
        if ub <= best_score + 1e-12:
            return  # Prune
        
        # Greedy coloring bound for tighter pruning
        # Color the subgraph induced by P using greedy coloring
        # The maximum clique size <= number of colors
        # For weighted case, we use a weighted coloring bound
        
        # Simple greedy coloring: assign colors greedily
        # We'll use a simple approach: sort by weight descending, assign smallest available color
        # The bound is: sum of top k weights where k = number of colors needed
        # But this is complex. Let's use a simpler bound for now.
        
        # Pivot selection: choose vertex u from P ∪ X with maximum degree in P
        # This helps reduce the number of recursive calls
        if P | X != 0:
            # Choose pivot with max degree in P
            pivot = -1
            max_deg = -1
            m = P | X
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                deg = bin(comp_adj[idx] & P).count('1')
                if deg > max_deg:
                    max_deg = deg
                    pivot = idx
                m ^= lsb
            
            # Recurse on P \ N(pivot)
            if pivot >= 0:
                P_reduced = P & ~comp_adj[pivot]
            else:
                P_reduced = P
        else:
            P_reduced = P
        
        # Process vertices in P_reduced
        m = P_reduced
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # Add v to R
            new_R = R | lsb
            new_blocked = current_blocked | small_blocks_large[v]
            
            # Compute new score
            new_score = current_score + small_weights[v]
            # Subtract newly blocked large weights
            newly_blocked = new_blocked & ~current_blocked
            new_score -= sum_large_weights_in_mask(newly_blocked)
            
            # Recurse
            new_P = P & comp_adj[v]  # Only candidates compatible with v
            new_X = X & comp_adj[v]
            bron_kerbosch(new_P, new_X, new_R, new_score, new_blocked)
            
            # Move v from P to X
            P = P & ~lsb
            X = X | lsb
            
            if elapsed() > DEADLINE:
                return
    
    # Start branch and bound
    # P = all small numbers, X = empty, R = empty
    bron_kerbosch((1 << n_small) - 1, 0, 0, 0.0, 0)
    
    # Also run local search from the best found solution as a polish
    if elapsed() < DEADLINE - 5:
        current_mask = best_small_mask
        current_blocked = best_blocked_large_mask
        current_score = best_score
        
        improved = True
        iterations = 0
        max_iterations = 10000
        
        while improved and elapsed() < DEADLINE and iterations < max_iterations:
            improved = False
            iterations += 1
            
            best_move_score = current_score
            best_move_new_mask = current_mask
            best_move_new_blocked = current_blocked
            
            # Try Additions
            for i in range(n_small):
                if current_mask & (1 << i):
                    continue
                if small_conflict_mask[i] & current_mask:
                    continue
                
                new_mask = current_mask | (1 << i)
                new_blocked = current_blocked | small_blocks_large[i]
                newly_blocked = new_blocked & ~current_blocked
                new_score = current_score + small_weights[i] - sum_large_weights_in_mask(newly_blocked)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_new_mask = new_mask
                    best_move_new_blocked = new_blocked
            
            # Try Removals
            for i in range(n_small):
                if not (current_mask & (1 << i)):
                    continue
                
                new_mask = current_mask & ~(1 << i)
                new_blocked = get_blocked_large_mask(new_mask)
                newly_unblocked = current_blocked & ~new_blocked
                new_score = current_score - small_weights[i] + sum_large_weights_in_mask(newly_unblocked)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_new_mask = new_mask
                    best_move_new_blocked = new_blocked
            
            # Try Swaps
            if elapsed() < DEADLINE - 2.0:
                for i in range(n_small):
                    if not (current_mask & (1 << i)):
                        continue
                    mask_without_i = current_mask & ~(1 << i)
                    blocked_without_i = get_blocked_large_mask(mask_without_i)
                    score_without_i = current_score - small_weights[i] + sum_large_weights_in_mask(current_blocked & ~blocked_without_i)
                    
                    for j in range(n_small):
                        if mask_without_i & (1 << j):
                            continue
                        if small_conflict_mask[j] & mask_without_i:
                            continue
                        
                        new_mask = mask_without_i | (1 << j)
                        new_blocked = blocked_without_i | small_blocks_large[j]
                        newly_blocked = new_blocked & ~blocked_without_i
                        new_score = score_without_i + small_weights[j] - sum_large_weights_in_mask(newly_blocked)
                        
                        if new_score > best_move_score:
                            best_move_score = new_score
                            best_move_new_mask = new_mask
                            best_move_new_blocked = new_blocked

            if best_move_new_mask != current_mask:
                current_mask = best_move_new_mask
                current_blocked = best_move_new_blocked
                current_score = best_move_score
                improved = True
                update_best(current_mask, current_blocked, current_score)

    # Construct final result
    result = []
    m = best_small_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(small_nums[idx])
        m ^= lsb
        
    blocked = best_blocked_large_mask
    for i in range(n_large):
        if not (blocked & (1 << i)):
            result.append(large_nums[i])
            
    result.sort()
    return result
# EVOLVE-BLOCK-END
