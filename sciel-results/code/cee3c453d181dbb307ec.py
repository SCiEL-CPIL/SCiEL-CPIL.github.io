import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    MAX_VAL = 256
    MAX_SUM = 60 * MAX_VAL
    # Bitset size needs to be MAX_SUM + 1
    # Python integers can handle arbitrary length, so we use int as bitset.
    
    def check_validity(A):
        """
        Verify A using bitsets.
        S[k] is a bitset where bit s is 1 if sum s is achievable by a subset of size k.
        Condition: For i != j, S[i] & S[j] == 0.
        """
        n = len(A)
        S = [0] * (n + 1)
        S[0] = 1 # sum 0 is achievable with size 0
        
        for x in A:
            # Update S from high to low
            for k in range(n, 0, -1):
                # S[k] gets new sums from S[k-1] shifted by x
                # We must be careful not to use the updated S[k-1]
                # Since we iterate k down, S[k-1] is from previous iteration (or original if k-1 > current max index processed?)
                # Wait, standard 0/1 knapsack DP:
                # S[k] |= S[k-1] << x
                # This works if we update k in descending order.
                S[k] |= (S[k-1] << x)
        
        # Check disjointness
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if S[i] & S[j]:
                    return False
        return True

    def solve_backtrack(candidates, current_A, S, current_size, best_size_ref):
        """
        Recursive backtracking.
        candidates: list of remaining numbers to consider
        current_A: list of numbers in current set
        S: list of bitsets for current set
        current_size: len(current_A)
        best_size_ref: list holding the best size found (mutable)
        """
        if time.time() - start_time > deadline:
            return
        
        # Update best
        if current_size > best_size_ref[0]:
            best_size_ref[0] = current_size
            # We could store the best set here, but returning it up the stack is complex.
            # We'll store it in a nonlocal variable or return it.
            # For simplicity in this edit, we'll just track size and reconstruct if needed, 
            # or pass the set. Passing the set is safer.
            
        # Pruning: if even adding all remaining candidates doesn't beat best, stop
        if current_size + len(candidates) <= best_size_ref[0]:
            return

        # Heuristic: Try candidates in a specific order
        # Descending order often works well for packing problems? Or ascending?
        # Let's stick to the order provided in candidates.
        
        for i, x in enumerate(candidates):
            if time.time() - start_time > deadline:
                return
            
            # Check if x can be added
            new_size = current_size + 1
            valid = True
            
            # We need to check if adding x creates a conflict.
            # New sums for size j are (S[j-1] << x).
            # These must not intersect with S[i] for i != j.
            # And they must not intersect with each other? No, same size is OK.
            # But wait, the new sums for size j are added to S[j].
            # The condition is that the FINAL sets are disjoint.
            # Since existing S[i] are disjoint, we just need to ensure:
            # 1. New sums for size j do not intersect with OLD S[i] for i != j.
            # 2. New sums for size j do not intersect with OLD S[i] for i == j? No, S[j] already exists.
            #    Wait, S[j] is updated to S[j] | (S[j-1] << x).
            #    The conflict check is between DIFFERENT cardinalities.
            #    So for the new elements $E_j = S[j-1] \ll x$, we must check:
            #    $E_j \cap S[i] = \emptyset$ for all $i \neq j$.
            #    Note: $E_j$ might overlap with $S[j]$? Yes, that's allowed (same cardinality).
            #    But $E_j$ might overlap with $S[i]$ for $i \neq j$.
            
            # Optimization:
            # For each j in 1..new_size:
            #   E_j = S[j-1] << x
            #   Forbidden = Union(S[i] for i in 0..current_size, i != j)
            #   If E_j & Forbidden: invalid.
            
            # We can precompute Forbidden for each j?
            # Forbidden_j = (TotalUnion) ^ S[j]  (if S[j] is subset of TotalUnion, which it is)
            # TotalUnion = OR of all S[i].
            
            total_union = 0
            for k in range(current_size + 1):
                total_union |= S[k]
            
            for j in range(1, new_size + 1):
                # S[j-1] is the source
                # Note: if j-1 > current_size, S[j-1] is 0, so E_j is 0.
                if j - 1 > current_size:
                    continue
                
                E_j = S[j-1] << x
                
                # Check intersection with S[i] for i != j
                # We can check: (E_j & total_union) must be a subset of S[j]
                # Because any bit in E_j that is in total_union must be in S[j] (since S[j] is the only set that can contain it, as others are disjoint from S[j] and each other).
                # Wait, S[i] for i != j are disjoint from S[j].
                # So if a bit is in E_j and in total_union, it MUST be in S[j] OR in some S[i] (i!=j).
                # If it's in S[i] (i!=j), it's a conflict.
                # So condition: (E_j & total_union) == (E_j & S[j])
                
                intersection = E_j & total_union
                allowed = E_j & S[j]
                
                if intersection != allowed:
                    valid = False
                    break
            
            if valid:
                # Add x
                new_S = S[:] # Copy
                for k in range(new_size, 0, -1):
                    new_S[k] |= (new_S[k-1] << x)
                
                # Recurse
                remaining = candidates[:i] + candidates[i+1:]
                # To save memory/time, we can pass slices, but slicing creates lists.
                # For small N (256), it's fine.
                solve_backtrack(remaining, current_A + [x], new_S, new_size, best_size_ref)
                
                if time.time() - start_time > deadline:
                    return

    # Initial setup
    # We want to find the largest set.
    # Start with empty set.
    # Candidates: 1..256
    
    # To improve performance, we should order candidates intelligently.
    # Randomized restarts with different orders.
    
    best_set = []
    best_size = 0
    
    # Deterministic orders first
    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        # Interleaved
        [i for i in range(1, 257, 2)] + [i for i in range(2, 257, 2)]
    ]
    
    for order in orders:
        if time.time() - start_time > deadline:
            break
        
        best_size_ref = [best_size]
        # We need to capture the actual set, not just size.
        # Let's modify solve_backtrack to return the best set found in that branch?
        # Or use a nonlocal variable.
        
        found_set = [None]
        
        def solve_bt(candidates, current_A, S, current_size):
            nonlocal found_set
            if time.time() - start_time > deadline:
                return
            
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
                found_set[0] = current_A[:]
            
            if current_size + len(candidates) <= best_size_ref[0]:
                return

            for i, x in enumerate(candidates):
                if time.time() - start_time > deadline:
                    return
                
                new_size = current_size + 1
                valid = True
                
                total_union = 0
                for k in range(current_size + 1):
                    total_union |= S[k]
                
                for j in range(1, new_size + 1):
                    if j - 1 > current_size:
                        continue
                    E_j = S[j-1] << x
                    intersection = E_j & total_union
                    allowed = E_j & S[j]
                    if intersection != allowed:
                        valid = False
                        break
                
                if valid:
                    new_S = S[:]
                    for k in range(new_size, 0, -1):
                        new_S[k] |= (new_S[k-1] << x)
                    
                    remaining = candidates[:i] + candidates[i+1:]
                    solve_bt(remaining, current_A + [x], new_S, new_size)
                    
                    if time.time() - start_time > deadline:
                        return

        # Initial S
        S_init = [0] * 61
        S_init[0] = 1
        
        solve_bt(order, [], S_init, 0)
        
        if found_set[0] and len(found_set[0]) > len(best_set):
            best_set = found_set[0]
            best_size = len(best_set)

    # If we still have time, do random restarts
    while time.time() - start_time < deadline:
        order = list(range(1, 257))
        random.shuffle(order)
        
        best_size_ref = [best_size]
        found_set = [None]
        
        def solve_bt_rand(candidates, current_A, S, current_size):
            nonlocal found_set
            if time.time() - start_time > deadline:
                return
            
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
                found_set[0] = current_A[:]
            
            if current_size + len(candidates) <= best_size_ref[0]:
                return

            for i, x in enumerate(candidates):
                if time.time() - start_time > deadline:
                    return
                
                new_size = current_size + 1
                valid = True
                
                total_union = 0
                for k in range(current_size + 1):
                    total_union |= S[k]
                
                for j in range(1, new_size + 1):
                    if j - 1 > current_size:
                        continue
                    E_j = S[j-1] << x
                    intersection = E_j & total_union
                    allowed = E_j & S[j]
                    if intersection != allowed:
                        valid = False
                        break
                
                if valid:
                    new_S = S[:]
                    for k in range(new_size, 0, -1):
                        new_S[k] |= (new_S[k-1] << x)
                    
                    remaining = candidates[:i] + candidates[i+1:]
                    solve_bt_rand(remaining, current_A + [x], new_S, new_size)
                    
                    if time.time() - start_time > deadline:
                        return

        S_init = [0] * 61
        S_init[0] = 1
        solve_bt_rand(order, [], S_init, 0)
        
        if found_set[0] and len(found_set[0]) > len(best_set):
            best_set = found_set[0]
            best_size = len(best_set)

    # Final verification
    if best_set and not check_validity(best_set):
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]

    return [int(x) for x in best_set]
