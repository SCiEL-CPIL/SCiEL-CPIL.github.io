import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 55.0
    MAX_VAL = 4096
    
    best_set = []
    best_len = 0

    def compute_sums(S):
        """Compute all unordered triple sums for set S."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(S[i] + S[j] + S[k])
        return sums

    def can_add(S, used_sums, x):
        """Check if x can be added to S without creating duplicate triple sums."""
        k = len(S)
        # 1. x + x + x
        s1 = 3 * x
        if s1 in used_sums:
            return False
        # 2. x + x + s
        for s in S:
            if (2 * x + s) in used_sums:
                return False
        # 3. x + s_i + s_j
        for i in range(k):
            for j in range(i, k):
                if (x + S[i] + S[j]) in used_sums:
                    return False
        return True

    def add_to_used(used_sums, S, x):
        """Add all new triple sums involving x to used_sums."""
        used_sums.add(3 * x)
        for s in S:
            used_sums.add(2 * x + s)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                used_sums.add(x + S[i] + S[j])

    def remove_from_used(used_sums, S, x):
        """Remove all triple sums involving x from used_sums."""
        used_sums.discard(3 * x)
        for s in S:
            used_sums.discard(2 * x + s)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                used_sums.discard(x + S[i] + S[j])

    # Phase 1: Greedy with different strategies to get a good initial set
    def greedy(start_offset=0, skip_prob=0.0, max_candidates=None, order=None):
        S = []
        used_sums = set()
        
        if order is None:
            candidates = list(range(1 + start_offset, MAX_VAL + 1))
            if skip_prob > 0:
                # Simple filter based on random probability
                # This is deterministic per call if we don't seed, but random.shuffle is expensive
                # We'll just iterate and skip with probability
                pass
            # To make skip_prob effective without full shuffle cost, we just iterate normally
            # and skip randomly. But we need to ensure we don't miss good candidates early.
            # Actually, for greedy, order matters. 
            # Let's stick to simple range for speed, or pre-shuffled if needed.
            # For now, standard range.
            iter_source = range(1 + start_offset, MAX_VAL + 1)
        else:
            iter_source = order

        count = 0
        for x in iter_source:
            if max_candidates is not None and len(S) >= max_candidates:
                break
            if skip_prob > 0 and random.random() < skip_prob:
                continue
            if can_add(S, used_sums, x):
                S.append(x)
                add_to_used(used_sums, S[:-1], x)
        return S

    # Standard greedy (increasing)
    S1 = greedy()
    best_set = S1
    best_len = len(S1)

    # Standard greedy (decreasing)
    S2 = greedy(order=range(MAX_VAL, 0, -1))
    if len(S2) > best_len:
        best_set = S2
        best_len = len(S2)

    # Randomized greedy restarts
    while time.time() < DEADLINE - 10.0:
        order = list(range(1, MAX_VAL + 1))
        random.shuffle(order)
        S_rand = greedy(order=order)
        if len(S_rand) > best_len:
            best_set = S_rand
            best_len = len(S_rand)

    # Phase 2: Local Search / Hill Climbing
    # We try to improve the best set found so far
    # Strategy: 
    # 1. Try to add any candidate.
    # 2. If stuck, try 1-1 swap: remove a random element, then try to add a new one.
    #    If the new one fits, keep it. If not, revert.
    
    def local_search(S_init, max_iterations=1000):
        S = list(S_init)
        S.sort()
        used_sums = compute_sums(S)
        current_len = len(S)
        no_improve_count = 0
        max_no_improve = 50
        
        for iteration in range(max_iterations):
            if time.time() > DEADLINE - 2.0:
                break
            
            improved = False
            
            # Try to add a new element
            S_set = set(S)
            # Generate candidates efficiently? 
            # Just iterate through 1..4096
            # To speed up, we can skip numbers that are "far" from existing sums?
            # But simple iteration is O(4096 * K^2). K is small (~20-40). 
            # 4096 * 1600 = 6.5M ops. In Python this is ~0.5s. 
            # We need to be faster.
            
            # Optimization: Pre-compute a list of "conflict" sums for each candidate? No.
            # Let's just try a subset of candidates first.
            
            # Strategy: Try candidates in increasing order first (greedy fill)
            # If that fails, try random.
            
            # 1. Try adding any candidate in increasing order
            for x in range(1, MAX_VAL + 1):
                if x in S_set:
                    continue
                if can_add(S, used_sums, x):
                    S.append(x)
                    S.sort()
                    add_to_used(used_sums, S[:-1], x)
                    S_set.add(x)
                    current_len += 1
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
                
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            
            # 2. Try 1-1 swap
            # Pick a random element to remove
            if len(S) > 1:
                idx_to_remove = random.randint(0, len(S) - 1)
                elem_to_remove = S[idx_to_remove]
                
                # Temporarily remove
                remove_from_used(used_sums, S, elem_to_remove)
                S.pop(idx_to_remove)
                S_set.discard(elem_to_remove)
                
                # Try to add a new element
                # We want to add something that frees up space or allows a larger set
                # Just try to add any valid candidate
                added = False
                # Try candidates in random order for diversity
                candidates = [x for x in range(1, MAX_VAL + 1) if x not in S_set]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add(S, used_sums, x):
                        S.append(x)
                        S.sort()
                        add_to_used(used_sums, S[:-1], x)
                        S_set.add(x)
                        added = True
                        break
                
                if added:
                    # We swapped one for one, so length is same. 
                    # But maybe this configuration allows adding MORE later.
                    # So we keep it and reset no_improve counter slightly?
                    # Actually, if we can't add immediately, we should check if we can add more.
                    # The loop continues, so it will try to add again.
                    # To avoid oscillating, we accept the swap.
                    # Reset no_improve to 0 to encourage further search
                    no_improve_count = 0
                else:
                    # Revert
                    S.append(elem_to_remove)
                    S.sort()
                    add_to_used(used_sums, S[:-1], elem_to_remove)
                    S_set.add(elem_to_remove)

        return S

    # Run local search on the best set found
    if time.time() < DEADLINE - 5.0:
        S_improved = local_search(best_set)
        if len(S_improved) > best_len:
            best_set = S_improved
            best_len = len(S_improved)

    # If we have time, try local search on a few other good sets?
    # Or restart local search with different random seeds?
    # The current best is likely near local optimum.
    
    # One more pass: Try to add elements to the final best set
    S_final = list(best_set)
    S_final.sort()
    used_sums = compute_sums(S_final)
    S_set = set(S_final)
    
    for x in range(1, MAX_VAL + 1):
        if x in S_set:
            continue
        if can_add(S_final, used_sums, x):
            S_final.append(x)
            S_final.sort()
            add_to_used(used_sums, S_final[:-1], x)
            S_set.add(x)
    
    if len(S_final) > best_len:
        best_set = S_final
        best_len = len(S_final)

    return best_set

# Fallback if construct fails or times out?
# The function itself handles the deadline.
