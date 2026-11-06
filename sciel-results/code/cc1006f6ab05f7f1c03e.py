import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (evaluator kills at 90s)
    
    TARGET = 200
    MAX_ELEM = 100
    
    # Bitset-based subset sum check: use Python int as bitset
    # Bit i set means sum i is reachable
    def is_valid_bitset(S):
        """Check if any subset of S sums to TARGET using bitset DP."""
        bits = 1  # bit 0 set: sum 0 reachable
        for x in S:
            if x > TARGET:
                continue
            # Shift left by x and OR
            bits |= (bits << x)
            # Early exit if TARGET bit is set
            if (bits >> TARGET) & 1:
                return False
        return True
    
    def find_conflict_bitset(S):
        """Find a subset of S that sums to TARGET. Returns list of elements or None."""
        # DP with parent tracking
        # dp[s] = list of elements used to achieve sum s, or None
        # Use dictionary for sparsity
        dp = {0: []}
        for x in S:
            if x > TARGET:
                continue
            new_dp = {}
            for s, elems in dp.items():
                ns = s + x
                if ns <= TARGET and ns not in dp and ns not in new_dp:
                    new_dp[ns] = elems + [x]
            for ns, elems in new_dp.items():
                dp[ns] = elems
            if TARGET in dp:
                return dp[TARGET]
        return None
    
    def greedy_build(order):
        """Build a valid set by greedily adding elements in given order."""
        S = []
        bits = 1  # bitset for reachable sums
        for x in order:
            if x > TARGET:
                S.append(x)
                continue
            # Check if adding x creates TARGET
            if (bits >> (TARGET - x)) & 1:
                # Conflict: some subset sums to TARGET-x, adding x gives TARGET
                continue
            # Safe to add: update bitset
            bits |= (bits << x)
            S.append(x)
        return S
    
    def local_search_improve(S):
        """Try to add elements from complement to S while maintaining validity."""
        S_set = set(S)
        all_elements = set(range(1, MAX_ELEM + 1))
        complement = list(all_elements - S_set)
        improved = True
        current_S = list(S)
        
        # Maintain bitset for fast checking
        bits = 1
        for x in current_S:
            if x <= TARGET:
                bits |= (bits << x)
        
        while improved:
            improved = False
            random.shuffle(complement)
            for x in complement:
                if x > TARGET:
                    # Always safe to add
                    current_S.append(x)
                    S_set.add(x)
                    complement.remove(x)
                    improved = True
                else:
                    # Check if adding x creates TARGET
                    if not ((bits >> (TARGET - x)) & 1):
                        # Safe to add
                        bits |= (bits << x)
                        current_S.append(x)
                        S_set.add(x)
                        complement.remove(x)
                        improved = True
        return current_S
    
    best_set = []
    best_size = -1
    
    def try_update(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Strategy 1: Greedy descending (large elements first)
    order = list(range(MAX_ELEM, 0, -1))
    S = greedy_build(order)
    S = local_search_improve(S)
    try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 2: Greedy ascending
    order = list(range(1, MAX_ELEM + 1))
    S = greedy_build(order)
    S = local_search_improve(S)
    try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 3: Randomized greedy with multiple restarts
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = list(range(1, MAX_ELEM + 1))
        random.shuffle(order)
        S = greedy_build(order)
        S = local_search_improve(S)
        try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 4: Start from full set, remove minimal conflicts
    # This tends to give larger sets
    num_removal_tries = 500
    for i in range(num_removal_tries):
        if time.time() > deadline:
            break
        
        S = list(range(1, MAX_ELEM + 1))
        random.shuffle(S)
        
        # Remove elements until valid
        max_removals = 50
        for _ in range(max_removals):
            conflict = find_conflict_bitset(S)
            if conflict is None:
                break
            # Remove a random element from the conflict
            elem_to_remove = random.choice(conflict)
            S.remove(elem_to_remove)
        
        # Now S should be valid (or we ran out of removals)
        if is_valid_bitset(S):
            S = local_search_improve(S)
            try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 5: Smart removal - remove elements that appear in most conflicts
    # First, identify "critical" elements
    num_smart_removal = 200
    for i in range(num_smart_removal):
        if time.time() > deadline:
            break
        
        S = list(range(1, MAX_ELEM + 1))
        
        # Count how often each element appears in conflicts
        conflict_count = {x: 0 for x in S}
        for _ in range(20):
            conflict = find_conflict_bitset(S)
            if conflict is None:
                break
            for elem in conflict:
                conflict_count[elem] += 1
        
        # Remove elements in order of highest conflict count
        for elem in sorted(S, key=lambda x: conflict_count[x], reverse=True):
            if not is_valid_bitset(S):
                S.remove(elem)
            else:
                break
        
        if is_valid_bitset(S):
            S = local_search_improve(S)
            try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 6: Hill climbing on the current best set
    # Try swapping elements or removing and re-adding
    if best_set:
        S = best_set.copy()
        S_set = set(S)
        complement = list(set(range(1, MAX_ELEM + 1)) - S_set)
        
        bits = 1
        for x in S:
            if x <= TARGET:
                bits |= (bits << x)
        
        # Try to add more elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(complement)
            for x in complement:
                if x > TARGET:
                    S.append(x)
                    S_set.add(x)
                    complement.remove(x)
                    improved = True
                else:
                    if not ((bits >> (TARGET - x)) & 1):
                        bits |= (bits << x)
                        S.append(x)
                        S_set.add(x)
                        complement.remove(x)
                        improved = True
        
        try_update(S)
    
    # Final check
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    if not best_set:
        best_set = [100]
    
    return best_set
# EVOLVE-BLOCK-END
