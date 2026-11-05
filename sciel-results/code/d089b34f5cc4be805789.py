# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    TARGET = 200
    UNIVERSE = list(range(1, 101))
    
    # Bitset representation for reachable sums
    # bit i set means sum i is reachable
    mask = (1 << (TARGET + 1)) - 1
    
    def can_add_bitset(reach_bits, x):
        """Check if adding x to a set with reachable sums 'reach_bits' creates sum 200."""
        if x > TARGET:
            return False
        return not (reach_bits & (1 << (TARGET - x)))
    
    def update_bitset(reach_bits, x):
        """Update reachable sums after adding x."""
        if x > TARGET:
            return reach_bits
        return reach_bits | ((reach_bits << x) & mask)
    
    def recompute_reach(S):
        """Recompute reachable sums for a set S."""
        reach_bits = 1  # bit 0 set
        for x in S:
            reach_bits = update_bitset(reach_bits, x)
        return reach_bits
    
    def greedy_build(order):
        """Greedy: add elements in given order if it doesn't create sum=TARGET."""
        reach_bits = 1  # bit 0 set
        selected = set()
        for x in order:
            if x in selected:
                continue
            if can_add_bitset(reach_bits, x):
                selected.add(x)
                reach_bits = update_bitset(reach_bits, x)
        return selected, reach_bits
    
    def local_search_improve(S, reach_bits):
        """Try removing one element and re-greeding to improve."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for y in S_list:
                if time.time() > deadline:
                    return S, reach_bits
                
                S_temp = set(S)
                S_temp.remove(y)
                
                # Recompute reach for S_temp
                reach_temp = recompute_reach(S_temp)
                
                # Try to add back valid elements
                candidates = [x for x in UNIVERSE if x not in S_temp]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add_bitset(reach_temp, x):
                        S_temp.add(x)
                        reach_temp = update_bitset(reach_temp, x)
                
                if len(S_temp) > len(S):
                    S = S_temp
                    reach_bits = reach_temp
                    improved = True
                    break  # Restart the loop with new S
            
        return S, reach_bits
    
    best_S = set()
    best_reach = 1
    
    # 1. Initialize with a known strong candidate: {66, ..., 100}
    # This set is valid: min sum of 3 is 66+67+68=201 > 200.
    # Max sum of 2 is 100+99=199 < 200.
    init_S = set(range(66, 101))
    init_reach = recompute_reach(init_S)
    
    if len(init_S) > len(best_S):
        best_S = init_S
        best_reach = init_reach
        
    # Run local search on this strong start
    S, reach = local_search_improve(best_S, best_reach)
    if len(S) > len(best_S):
        best_S = S
        best_reach = reach

    # 2. Run several random initializations and local searches to see if we can beat 35
    for i in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random order greedy
        order = list(UNIVERSE)
        random.shuffle(order)
        S, reach = greedy_build(order)
        
        if len(S) > len(best_S):
            best_S = S
            best_reach = reach
        
        # Local search on this S
        S, reach = local_search_improve(S, reach)
        
        if len(S) > len(best_S):
            best_S = S
            best_reach = reach
    
    # Final check: try to add any remaining valid elements to best_S
    candidates = [x for x in UNIVERSE if x not in best_S]
    for x in candidates:
        if can_add_bitset(best_reach, x):
            best_S.add(x)
            best_reach = update_bitset(best_reach, x)
    
    return list(best_S)
# EVOLVE-BLOCK-END
