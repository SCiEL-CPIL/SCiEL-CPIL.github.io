"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    target = 200
    mask = (1 << (target + 1)) - 1  # bits 0..200
    
    def try_order(order, initial_set=None, initial_reach=None):
        """Greedy: add elements in given order if it doesn't create sum=target."""
        if initial_reach is None:
            reachable = 1  # bit 0 set
        else:
            reachable = initial_reach
        if initial_set is None:
            selected = []
        else:
            selected = list(initial_set)
        
        selected_set = set(selected)
        for x in order:
            if x in selected_set:
                continue
            new_sums = (reachable << x) & mask
            if not (new_sums & (1 << target)):
                selected.append(x)
                selected_set.add(x)
                reachable |= new_sums
        return selected, reachable
    
    def recompute_reach(S):
        """Compute reachable sums for set S."""
        reachable = 1
        for x in sorted(S):
            reachable |= (reachable << x) & mask
        return reachable
    
    best = []
    best_size = 0
    
    # Strategy 1: Start with guaranteed-valid set {67,...,100}
    S0 = list(range(67, 101))
    reach0 = recompute_reach(S0)
    # Try adding smaller numbers in various orders
    small = list(range(1, 67))
    
    orders = [
        small,
        list(reversed(small)),
        small[::2] + small[1::2],
        small[1::2] + small[::2],
    ]
    random.shuffle(small)
    orders.append(small[:])
    random.shuffle(small)
    orders.append(small[:])
    
    for order in orders:
        S, reach = try_order(order, S0, reach0)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Strategy 2: Pure greedy from scratch with various orders
    all_orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(100, 50, -1)) + list(range(50, 0, -1)),
        list(range(1, 51)) + list(range(100, 50, -1)),
    ]
    # Alternating from ends
    order5 = []
    lo, hi = 1, 100
    while lo <= hi:
        order5.append(hi)
        if lo != hi:
            order5.append(lo)
        lo += 1
        hi -= 1
    all_orders.append(order5)
    # Middle out
    all_orders.append(list(range(50, 101)) + list(range(49, 0, -1)))
    
    for order in all_orders:
        S, reach = try_order(order)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local search: try removing one element and re-adding
    S = list(best)
    reach = recompute_reach(S)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        for y in S_list:
            if time.time() > deadline - 3:
                break
            S_temp = [x for x in S if x != y]
            reach_temp = recompute_reach(S_temp)
            # Try to add back any valid elements
            candidates = [x for x in range(1, 101) if x not in S_temp]
            random.shuffle(candidates)
            S_new, reach_new = try_order(candidates, S_temp, reach_temp)
            if len(S_new) > len(S):
                S = S_new
                reach = reach_new
                if len(S) > best_size:
                    best = list(S)
                    best_size = len(S)
                improved = True
                break
    
    # Final pass: try to add any remaining valid elements
    S = list(best)
    reach = recompute_reach(S)
    S_final, reach_final = try_order(list(range(1, 101)), S, reach)
    if len(S_final) > best_size:
        best = S_final
        best_size = len(S_final)
    
    return best
# EVOLVE-BLOCK-END
