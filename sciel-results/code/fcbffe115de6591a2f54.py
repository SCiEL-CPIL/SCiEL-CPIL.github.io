import time
import numpy as np
import random

def construct():
    """
    Construct a set of distinct positive integers <= 4096 such that all 
    unordered triple sums (with repetition) are unique. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    n_max = 4096
    max_sum = 3 * n_max
    
    # Pre-allocate arrays for speed
    # sums_arr[s] is True if s is a triple sum in the current set
    # pair_sums is a list of all pair sums s_i + s_j with i <= j
    
    def can_add(x, S, sums_arr, pair_sums_np):
        """
        Check if x can be added to S without creating duplicate triple sums.
        S: sorted list of current elements
        sums_arr: boolean numpy array of size max_sum+1
        pair_sums_np: numpy array of current pair sums
        """
        # 1. Check 3x
        if sums_arr[3 * x]:
            return False
        
        # 2. Check 2x + s for s in S
        if S:
            # Vectorized check
            candidates = 2 * x + np.array(S)
            if np.any(sums_arr[candidates]):
                return False
        
        # 3. Check x + p for p in pair_sums
        if len(pair_sums_np) > 0:
            candidates = x + pair_sums_np
            if np.any(sums_arr[candidates]):
                return False
        
        return True

    def add_element(x, S, sums_arr, pair_sums_np):
        """
        Add x to S and update sums_arr and pair_sums_np.
        Returns updated S, sums_arr, pair_sums_np.
        """
        S.append(x)
        
        # Update sums_arr
        sums_arr[3 * x] = True
        
        # Update pair_sums_np and sums_arr for 2x+s
        new_pair_sums = [2 * x]
        if S[:-1]:
            s_arr = np.array(S[:-1])
            # 2x + s
            sums = 2 * x + s_arr
            sums_arr[sums] = True
            new_pair_sums.extend((x + s_arr).tolist())
            
            # x + s + t for s <= t in S[:-1]
            # We need all pair sums of S[:-1]
            # Since S is sorted, we can generate them
            # But we already have pair_sums_np which contains pairs from OLD S.
            # We need pairs from S[:-1] (which is OLD S).
            # So we can just use pair_sums_np?
            # Wait, pair_sums_np contains pairs from the set BEFORE adding x.
            # The new triple sums involving x and two old elements are x + (s_i + s_j).
            # These are exactly x + pair_sums_np.
            if len(pair_sums_np) > 0:
                triple_sums = x + pair_sums_np
                sums_arr[triple_sums] = True
        
        # Update pair_sums_np
        if new_pair_sums:
            new_pair_sums_np = np.array(new_pair_sums)
            pair_sums_np = np.concatenate([pair_sums_np, new_pair_sums_np])
        
        return S, sums_arr, pair_sums_np

    def remove_element(x, S, sums_arr, pair_sums_np):
        """
        Remove x from S. Rebuild sums_arr and pair_sums_np from scratch for simplicity 
        and correctness, as incremental removal is complex.
        """
        S.remove(x)
        S.sort()
        
        # Rebuild sums_arr
        sums_arr[:] = False
        pair_sums = []
        
        k = len(S)
        if k == 0:
            return S, sums_arr, np.array([], dtype=int)
        
        # Generate all pair sums
        pair_sums_list = []
        for i in range(k):
            for j in range(i, k):
                p = S[i] + S[j]
                pair_sums_list.append(p)
                # Triple sums:
                # 3x case: i==j==k
                # 2x+y case: i==j != k
                # x+y+z case: i<j<k
                # It's easier to just iterate all triples for small k
                # But k is small (<20), so O(k^3) is fine.
        
        # Actually, let's just rebuild by iterating triples, it's safer.
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s = S[i] + S[j] + S[l]
                    sums_arr[s] = True
        
        pair_sums_np = np.array(pair_sums_list, dtype=int)
        return S, sums_arr, pair_sums_np

    def greedy_fill(S, sums_arr, pair_sums_np, order):
        """
        Greedily add elements from 'order' to S if they fit.
        """
        for x in order:
            if time.time() > deadline:
                break
            if x in S:
                continue
            if can_add(x, S, sums_arr, pair_sums_np):
                S, sums_arr, pair_sums_np = add_element(x, S, sums_arr, pair_sums_np)
        return S, sums_arr, pair_sums_np

    def local_search(S, sums_arr, pair_sums_np):
        """
        Local search: try to swap an element with a smaller unused one, or insert a new one.
        """
        S = S[:]
        # We need to keep sums_arr and pair_sums_np consistent.
        # Since removal is expensive, we will try moves, and if accepted, rebuild.
        # If rejected, do nothing.
        
        used = set(S)
        all_nums = set(range(1, n_max + 1))
        available = all_nums - used
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
            
            # 1. Try Insertion
            # Sort available by value? Or random?
            # Let's try inserting the smallest available that fits.
            # But checking all is expensive.
            # Let's pick a few random candidates or the smallest few.
            # Since we want to maximize size, any valid insertion is good.
            # Let's try the smallest available first to keep numbers small.
            sorted_avail = sorted(available)
            for x in sorted_avail[:100]: # Check first 100 smallest
                if can_add(x, S, sums_arr, pair_sums_np):
                    S, sums_arr, pair_sums_np = add_element(x, S, sums_arr, pair_sums_np)
                    used.add(x)
                    available.discard(x)
                    improved = True
                    break
            
            if improved:
                continue # Restart loop as set changed

            # 2. Try Swap
            # For each element s in S, try to swap with a smaller available u < s.
            # This reduces the sum, potentially allowing more elements.
            S_sorted = sorted(S)
            for idx, s in enumerate(S_sorted):
                if time.time() > deadline - 5:
                    break
                
                # Find available u < s
                # We can iterate through available
                # To be efficient, we can check if any u < s can replace s.
                # Replacing s with u:
                # New set S' = S \ {s} U {u}
                # We need to check if S' is valid.
                # This is equivalent to: 
                # 1. Remove s from S -> S_rem
                # 2. Check if u can be added to S_rem.
                
                # To avoid full rebuild on every check, we can do a "dry run" check?
                # Or just try the swap and rebuild if it works.
                # Given k is small, rebuild is O(k^3) ~ 8000 ops.
                # We can afford a few hundred rebuilds.
                
                # Let's try swapping s with the smallest available u < s.
                # Or random u < s.
                # Let's try the smallest u < s.
                
                # Find smallest u in available such that u < s
                u = None
                # Since available is a set, we can't sort it easily every time.
                # Let's maintain a sorted list of available? 
                # Or just iterate.
                # For simplicity, iterate through available.
                # Optimization: Only try if there is an available u < s.
                # min_avail = min(available)
                # if min_avail >= s: continue
                
                # Let's pick a random u < s to avoid bias
                # But smallest is usually better for "packing".
                
                candidates = [u for u in available if u < s]
                if not candidates:
                    continue
                
                # Try the smallest candidate
                u = min(candidates)
                
                # Perform swap
                S_test = S[:]
                S_test.remove(s)
                S_test.append(u)
                S_test.sort()
                
                # Check validity of S_test
                # We can use can_add on u against S_test \ {u}
                # But we don't have sums_arr for S_test \ {u}.
                # So we must rebuild or do a full check.
                # Full check O(k^3)
                
                valid = True
                k = len(S_test)
                seen_sums = set()
                for i in range(k):
                    for j in range(i, k):
                        for l in range(j, k):
                            s_val = S_test[i] + S_test[j] + S_test[l]
                            if s_val in seen_sums:
                                valid = False
                                break
                            seen_sums.add(s_val)
                    if not valid:
                        break
                    if not valid:
                        break
                    if not valid:
                        break
                
                if valid:
                    # Accept swap
                    S = S_test
                    used.remove(s)
                    used.add(u)
                    available.discard(u)
                    available.add(s)
                    # Rebuild sums_arr and pair_sums_np
                    S, sums_arr, pair_sums_np = remove_element(s, S, sums_arr, pair_sums_np) # This removes s, but we need to add u?
                    # Wait, remove_element rebuilds from S. 
                    # So we should just call a rebuild function.
                    
                    # Let's refactor: remove_element currently removes x from S passed to it.
                    # So:
                    # S_current = S (which still has s)
                    # We want S_new = S \ {s} U {u}
                    # Let's just rebuild from scratch using a helper.
                    
                    def rebuild(S_in):
                        sums_arr_local = np.zeros(max_sum + 1, dtype=bool)
                        pair_sums_list = []
                        k = len(S_in)
                        for i in range(k):
                            for j in range(i, k):
                                p = S_in[i] + S_in[j]
                                pair_sums_list.append(p)
                        pair_sums_np_local = np.array(pair_sums_list, dtype=int)
                        
                        for i in range(k):
                            for j in range(i, k):
                                for l in range(j, k):
                                    s_val = S_in[i] + S_in[j] + S_in[l]
                                    sums_arr_local[s_val] = True
                        return S_in, sums_arr_local, pair_sums_np_local
                    
                    S, sums_arr, pair_sums_np = rebuild(S)
                    improved = True
                    break # Restart loop
        
        return S, sums_arr, pair_sums_np

    best_set = []
    
    # Strategy 1: Greedy from small numbers
    S = []
    sums_arr = np.zeros(max_sum + 1, dtype=bool)
    pair_sums_np = np.array([], dtype=int)
    
    # Initialize with a few small numbers to get a base
    # Or just start empty and greedily add smallest available.
    order = list(range(1, n_max + 1))
    S, sums_arr, pair_sums_np = greedy_fill(S, sums_arr, pair_sums_np, order)
    
    if len(S) > len(best_set):
        best_set = S[:]
    
    if time.time() > deadline - 10:
        return best_set

    # Local search on the greedy result
    S, sums_arr, pair_sums_np = local_search(S, sums_arr, pair_sums_np)
    if len(S) > len(best_set):
        best_set = S[:]

    # Strategy 2: Multiple random restarts
    for _ in range(20):
        if time.time() > deadline - 2:
            break
        
        # Random start
        S = []
        sums_arr = np.zeros(max_sum + 1, dtype=bool)
        pair_sums_np = np.array([], dtype=int)
        
        # Add a few random numbers
        # Ensure they don't collide
        for _ in range(3):
            x = random.randint(1, n_max)
            if can_add(x, S, sums_arr, pair_sums_np):
                S, sums_arr, pair_sums_np = add_element(x, S, sums_arr, pair_sums_np)
        
        # Greedy fill with random order
        order = list(range(1, n_max + 1))
        random.shuffle(order)
        S, sums_arr, pair_sums_np = greedy_fill(S, sums_arr, pair_sums_np, order)
        
        # Local search
        S, sums_arr, pair_sums_np = local_search(S, sums_arr, pair_sums_np)
        
        if len(S) > len(best_set):
            best_set = S[:]

    # Final verification and cleanup
    best_set.sort()
    return best_set

# EVOLVE-BLOCK-END
