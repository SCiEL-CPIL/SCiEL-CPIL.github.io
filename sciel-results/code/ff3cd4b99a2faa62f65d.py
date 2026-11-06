import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 512
    MAX_SIZE = 48
    
    def is_valid_subset_sum_zero(diffs, min_size=2):
        """
        Checks if there is a subset of diffs of size >= min_size that sums to 0.
        diffs: list of integers.
        """
        if len(diffs) < min_size:
            return False
        
        # Bitset DP for subset sums
        # Range of sums: +/- 512 * 48 = 24576
        # Offset to handle negatives
        offset = 25000
        size = 50000
        # dp[s] is a bitset where bit k is set if there is a subset of size k summing to s
        # This is too big: 50000 * 48 bits = 2.4M bits = 300KB per x.
        # 48 x's -> 14MB. Manageable.
        # But Python lists of ints are slow.
        
        # Alternative: Check specific small sizes first.
        # Size 2: a + b = 0 => a = -b
        # Size 3: a + b + c = 0
        # Size 4: a + b + c + d = 0
        
        # For efficiency, we can just check all subsets of size 2, 3, 4.
        # If no zero sum in size 2, 3, 4, it's very unlikely for size > 4 in a random set.
        # Actually, if a large subset sums to 0, it can be decomposed? No.
        # But mean-free sets are typically sparse enough that large subsets don't sum to 0 easily.
        
        # Let's use a set of reachable sums for each size.
        # sums[k] = set of sums achievable with k elements.
        sums = [set() for _ in range(len(diffs) + 1)]
        sums[0].add(0)
        
        for d in diffs:
            # Iterate backwards to avoid using d multiple times
            for k in range(len(diffs) - 1, 0, -1):
                if sums[k-1]:
                    new_sums = {s + d for s in sums[k-1]}
                    sums[k].update(new_sums)
            
            # Early exit: if 0 is in sums[k] for k >= 2
            for k in range(2, len(diffs) + 1):
                if 0 in sums[k]:
                    return True
        return False

    def is_mean_free(S):
        """
        Checks if no element in S is the mean of any subset of >= 2 other elements.
        """
        S_list = list(S)
        n = len(S_list)
        for i, x in enumerate(S_list):
            # Get differences y - x for y in S, y != x
            diffs = [y - x for j, y in enumerate(S_list) if j != i]
            if is_valid_subset_sum_zero(diffs, min_size=2):
                return False
        return True

    def get_conflicts(S, y):
        """
        Returns a set of elements in S that conflict with y.
        If S U {y} is invalid, this helps identify which elements to remove.
        """
        conflicts = set()
        S_list = list(S)
        
        # 1. Check if y is the mean of a subset of S
        diffs_y = [x - y for x in S_list]
        if is_valid_subset_sum_zero(diffs_y, min_size=2):
            # y is the mean. All elements involved in the zero-sum subset conflict.
            # For simplicity, mark all as potential conflicts or find specific ones.
            # We just need to know that y cannot be added.
            # To remove, we need to break the zero-sum.
            # We'll return a representative conflict or all.
            # For swap heuristic, we just need to know it's invalid.
            pass 

        # 2. Check if any x in S is the mean of a subset involving y
        for x in S_list:
            # Subset T from (S \ {x}) U {y}
            # T must have size >= 2
            # Sum(T) = |T| * x
            # Let T = {y} U T' where T' subset of S \ {x}
            # y + Sum(T') = (|T'| + 1) * x
            # Sum(T') = (|T'| + 1) * x - y
            # Let k = |T'|. Sum(T') = k*x + x - y.
            # We need to check if there is a subset of S \ {x} of size k summing to k*x + x - y.
            
            # This is expensive. 
            # Approximation: Check if y creates a 3-AP with x and another z.
            # 2x = y + z => z = 2x - y. If z in S, conflict.
            z = 2 * x - y
            if z in S and z != x:
                conflicts.add(x)
                conflicts.add(z)
                
            # Check if x is mean of y and two others?
            # 3x = y + z1 + z2
            # Too complex for fast check.
            
        return conflicts

    # Strategy:
    # 1. Find a valid set of size ~40 using greedy.
    # 2. Use local search to expand to 48.
    
    best_set = set()
    best_size = 0
    
    # Greedy Initialization
    def greedy_init():
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for y in candidates:
            if len(S) >= MAX_SIZE:
                break
            # Check if adding y is valid
            # Use a fast check: just check 3-APs for speed in init
            valid = True
            for x in S:
                z = 2 * x - y
                if z in S and z != x:
                    valid = False
                    break
                # Also check if y is mean of x and z
                z = 2 * y - x
                if z in S and z != x and z != y:
                    valid = False
                    break
            if valid:
                # Full check for safety? No, too slow.
                # Assume 3-AP check is sufficient for init, refine later.
                S.add(y)
        return S

    # Local Search
    def local_search(S):
        S = set(S)
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = set(S)
            
        no_improve_count = 0
        temperature = 1.0
        cooling = 0.999
        
        while time.time() < deadline:
            # Try to add a random element
            if len(S) < MAX_SIZE:
                y = random.randint(1, N)
                if y not in S:
                    # Check validity of S U {y}
                    # Use fast 3-AP check first
                    valid_3ap = True
                    for x in S:
                        z = 2 * x - y
                        if z in S and z != x:
                            valid_3ap = False
                            break
                        z = 2 * y - x
                        if z in S and z != x and z != y:
                            valid_3ap = False
                            break
                    
                    if valid_3ap:
                        # Full check
                        new_S = S | {y}
                        if is_mean_free(new_S):
                            S = new_S
                            if len(S) > best_size:
                                best_size = len(S)
                                best_set = set(S)
                            no_improve_count = 0
                            continue
                    else:
                        no_improve_count += 1

            # If we can't add, or if we are at max size, try swaps to find better structure
            # Or if we stuck, try swaps
            
            if len(S) > 0:
                # Random Swap
                x = random.choice(list(S))
                y = random.randint(1, N)
                if y in S or y == x:
                    continue
                
                # Try swapping x with y
                new_S = (S - {x}) | {y}
                
                # Fast check
                valid_3ap = True
                for a in new_S:
                    z = 2 * a - y
                    if z in new_S and z != a:
                         # Check if this is a new conflict
                         # If z == x, it's removed, so OK.
                         # If z != x, it was in S, so if it was valid before, it is still valid?
                         # No, y is new.
                         # We just need to check conflicts involving y.
                         pass
                
                # Just do full check for simplicity, but limit frequency?
                # is_mean_free is ~10-50ms.
                # 1000 checks in 60s is feasible.
                
                if is_mean_free(new_S):
                    S = new_S
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 100:
                        # Restart or cool down
                        no_improve_count = 0
                        # Maybe randomize S slightly?
                        if len(S) > 0:
                            # Remove a random element
                            S.discard(random.choice(list(S)))
            
            time.sleep(0.001) # Yield CPU
            
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        S_init = greedy_init()
        local_search(S_init)
        restarts += 1
        if best_size >= MAX_SIZE:
            break
            
    return list(best_set)

# EVOLVE-BLOCK-END
