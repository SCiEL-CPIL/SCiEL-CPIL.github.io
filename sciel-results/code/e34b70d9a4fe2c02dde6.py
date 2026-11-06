import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (evaluator kills at 90s)
    
    TARGET = 200
    MAX_ELEM = 100
    
    def is_valid_bitset(S):
        """Check if any subset of S sums to TARGET using bitset DP."""
        bits = 1  # bit 0 set: sum 0 reachable
        for x in S:
            if x > TARGET:
                continue
            bits |= (bits << x)
            if (bits >> TARGET) & 1:
                return False
        return True
    
    def find_conflict_fast(S):
        """Find a subset of S that sums to TARGET. Returns list of elements or None."""
        parent = [None] * (TARGET + 1)
        parent[0] = (0, -1)
        reachable = [0] * (TARGET + 1)
        reachable[0] = 1
        
        for idx, x in enumerate(S):
            if x > TARGET:
                continue
            for s in range(TARGET - x, -1, -1):
                if reachable[s] and not reachable[s + x]:
                    reachable[s + x] = 1
                    parent[s + x] = (s, idx)
            if reachable[TARGET]:
                elems = []
                cur = TARGET
                while cur > 0:
                    prev_s, elem_idx = parent[cur]
                    elems.append(S[elem_idx])
                    cur = prev_s
                return elems
        return None
    
    def greedy_build(order, bits=None):
        """Build a valid set by greedily adding elements in given order."""
        S = []
        if bits is None:
            bits = 1
        for x in order:
            if x > TARGET:
                S.append(x)
                continue
            if (bits >> (TARGET - x)) & 1:
                continue
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
        
        bits = 1
        for x in current_S:
            if x <= TARGET:
                bits |= (bits << x)
        
        while improved:
            improved = False
            random.shuffle(complement)
            for x in complement:
                if x > TARGET:
                    current_S.append(x)
                    S_set.add(x)
                    complement.remove(x)
                    improved = True
                else:
                    if not ((bits >> (TARGET - x)) & 1):
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
    
    # Strategy 0: Known good structural sets
    # {67,...,100}: 34 elements, provably valid (no pair sums to 200, min triple > 200)
    # Try to extend with small elements
    structural_sets = [
        list(range(67, 101)),  # 34 elements
        list(range(66, 101)),  # 35 elements - check if valid
        list(range(1, 51)),    # 50 elements - check
        list(range(51, 101)),  # 50 elements - check
    ]
    
    for es in structural_sets:
        if is_valid_bitset(es):
            es = local_search_improve(es)
            try_update(es)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 1: Greedy descending
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
    
    # Strategy 3: Weighted greedy - prefer elements that are "harder" to use in conflicts
    # Elements near TARGET/2 are most dangerous; elements far from TARGET/2 are safer
    # Actually, large elements > TARGET/2 can only pair with small ones
    # Sort by "danger": elements that create more conflicts
    order = list(range(1, MAX_ELEM + 1))
    order.sort(key=lambda x: abs(x - TARGET/2))  # Most dangerous first? Or least?
    # Try least dangerous first (far from 100)
    S = greedy_build(order)
    S = local_search_improve(S)
    try_update(S)
    
    # Most dangerous first
    order.sort(key=lambda x: abs(x - TARGET/2), reverse=True)
    S = greedy_build(order)
    S = local_search_improve(S)
    try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 4: Randomized greedy with many restarts
    num_restarts = 5000
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
    
    # Strategy 5: Start from full set, remove minimal conflicts (hitting set approach)
    num_removal_tries = 1000
    for i in range(num_removal_tries):
        if time.time() > deadline:
            break
        
        S = list(range(1, MAX_ELEM + 1))
        random.shuffle(S)
        
        max_removals = 60
        for _ in range(max_removals):
            conflict = find_conflict_fast(S)
            if conflict is None:
                break
            elem_to_remove = random.choice(conflict)
            S.remove(elem_to_remove)
        
        if is_valid_bitset(S):
            S = local_search_improve(S)
            try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 6: Smart removal - remove elements that appear in most conflicts
    num_smart_removal = 500
    for i in range(num_smart_removal):
        if time.time() > deadline:
            break
        
        S = list(range(1, MAX_ELEM + 1))
        
        conflict_count = {x: 0 for x in S}
        for _ in range(50):
            conflict = find_conflict_fast(S)
            if conflict is None:
                break
            for elem in conflict:
                conflict_count[elem] += 1
        
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
    
    # Strategy 7: Hill climbing on current best
    if best_set:
        S = best_set.copy()
        S_set = set(S)
        complement = list(set(range(1, MAX_ELEM + 1)) - S_set)
        
        bits = 1
        for x in S:
            if x <= TARGET:
                bits |= (bits << x)
        
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
    
    # Strategy 8: Swap-based local search - try to swap elements to improve
    if best_set and time.time() < deadline:
        S = best_set.copy()
        S_set = set(S)
        complement = list(set(range(1, MAX_ELEM + 1)) - S_set)
        
        bits = 1
        for x in S:
            if x <= TARGET:
                bits |= (bits << x)
        
        # Try swapping: remove one from S, add one from complement
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            random.shuffle(complement)
            for x_in in complement:
                if x_in > TARGET:
                    S.append(x_in)
                    S_set.add(x_in)
                    complement.remove(x_in)
                    improved = True
                    continue
                if (bits >> (TARGET - x_in)) & 1:
                    continue
                # Can we add x_in?
                new_bits = bits | (bits << x_in)
                if not ((new_bits >> TARGET) & 1):
                    S.append(x_in)
                    S_set.add(x_in)
                    complement.remove(x_in)
                    bits = new_bits
                    improved = True
            if not improved:
                break
        try_update(S)
    
    # Final check
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    if not best_set:
        best_set = [100]
    
    return best_set
# EVOLVE-BLOCK-END
