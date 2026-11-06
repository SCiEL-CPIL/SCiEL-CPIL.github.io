"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill (evaluator limit)
    N = 24
    universe = list(range(1, N + 1))
    
    # Precompute LCM of all numbers 1..24 for exact integer arithmetic
    total_lcm = 1
    for i in range(1, N + 1):
        total_lcm = lcm(total_lcm, i)
    
    # weight[i] = total_lcm // i
    weights = {i: total_lcm // i for i in universe}
    
    best_set = set()
    best_size = 0
    
    # Fast validity check using Meet-in-the-Middle
    def is_valid(s):
        s_list = list(s)
        n = len(s_list)
        if n <= 1:
            return True
        
        for i, a in enumerate(s_list):
            target = weights[a]
            others = [x for j, x in enumerate(s_list) if j != i]
            m = len(others)
            
            if m == 0:
                continue
            
            # Quick check: if target > sum of all others, impossible
            if target > sum(weights[x] for x in others):
                continue
                
            # Meet in the middle
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Build left sums
            left_sums = set()
            left_sums.add(0)
            for mask in range(1, 1 << len(left)):
                s_sum = 0
                for k in range(len(left)):
                    if mask & (1 << k):
                        s_sum += weights[left[k]]
                left_sums.add(s_sum)
            
            # Check if target can be formed by left_sum + right_sum
            for mask in range(1 << len(right)):
                r_sum = 0
                for k in range(len(right)):
                    if mask & (1 << k):
                        r_sum += weights[right[k]]
                if (target - r_sum) in left_sums:
                    return False
            
        return True

    def greedy_construct(order, initial_set=None):
        current_set_set = set(initial_set) if initial_set else set()
        current_list = list(current_set_set)
        for elem in order:
            if elem in current_set_set:
                continue
            candidate_set = current_set_set | {elem}
            if is_valid(candidate_set):
                current_list.append(elem)
                current_set_set.add(elem)
        return current_set_set

    def try_extend(base_set, remaining, order):
        """Try to extend a base set by adding elements from remaining in given order."""
        current_set = set(base_set)
        for elem in order:
            if elem in current_set or elem not in remaining:
                continue
            candidate_set = current_set | {elem}
            if is_valid(candidate_set):
                current_set.add(elem)
        return current_set

    # 1. Deterministic Greedy: Descending (large elements are "smaller" reciprocally, safer)
    order_desc = sorted(universe, reverse=True)
    set_desc = greedy_construct(order_desc)
    if len(set_desc) > best_size:
        best_size = len(set_desc)
        best_set = set(set_desc)
        
    # 2. Deterministic Greedy: Ascending
    order_asc = sorted(universe)
    set_asc = greedy_construct(order_asc)
    if len(set_asc) > best_size:
        best_size = len(set_asc)
        best_set = set(set_asc)
    
    # 3. Structured initialization: Start from a known-good core (avoid very small numbers)
    # The insight: small numbers have large weights, making them hard to fit
    # Try starting from {12..24} or similar, then add smaller ones
    core_starts = [
        list(range(12, 25)),  # 12-24
        list(range(10, 25)),  # 10-24
        list(range(8, 25)),   # 8-24
        list(range(7, 25)),   # 7-24
    ]
    
    for core in core_starts:
        if time.time() > deadline - 5:
            break
        # Check if core itself is valid
        core_set = set(core)
        if is_valid(core_set):
            if len(core_set) > best_size:
                best_size = len(core_set)
                best_set = set(core_set)
            # Try to extend by adding smaller numbers
            remaining = [x for x in range(1, min(core)) if x not in core_set]
            # Try adding in descending order first
            for r_order in [sorted(remaining, reverse=True), sorted(remaining), remaining]:
                extended = try_extend(core_set, set(remaining), r_order)
                if len(extended) > best_size:
                    best_size = len(extended)
                    best_set = set(extended)
    
    # 4. Randomized Greedy Restart with diverse strategies
    attempts = 0
    rng = random.Random(42)
    
    while time.time() < deadline - 3:
        strategy = rng.random()
        
        if strategy < 0.3:
            # Pure random shuffle
            order = universe[:]
            rng.shuffle(order)
        elif strategy < 0.6:
            # Descending with random perturbations
            order = sorted(universe, reverse=True)
            # Swap a few random pairs
            for _ in range(rng.randint(1, 5)):
                i, j = rng.sample(range(len(order)), 2)
                order[i], order[j] = order[j], order[i]
        elif strategy < 0.8:
            # Start from a good core, then randomize the rest
            core_size = rng.randint(8, 14)
            core = rng.sample(range(core_size + 1, 25), min(core_size, 24 - core_size))
            rest = [x for x in universe if x not in core]
            rng.shuffle(rest)
            order = core + rest
        else:
            # Weighted random: larger numbers more likely to come first
            order = sorted(universe, key=lambda x: -x + rng.gauss(0, 5))
        
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
        
        attempts += 1
    
    # 5. Local improvement: try to add one more element to best_set
    if time.time() < deadline - 2:
        for elem in universe:
            if elem in best_set:
                continue
            candidate = best_set | {elem}
            if is_valid(candidate):
                best_set = candidate
                best_size = len(best_set)
    
    # 6. Try swapping: remove one element, try to add two others
    if time.time() < deadline - 1 and best_size < 24:
        for remove_elem in list(best_set):
            if time.time() > deadline - 1:
                break
            reduced = best_set - {remove_elem}
            # Try adding any two elements not in reduced
            for add1 in universe:
                if add1 in reduced:
                    continue
                for add2 in universe:
                    if add2 in reduced or add2 == add1:
                        continue
                    candidate = reduced | {add1, add2}
                    if is_valid(candidate) and len(candidate) > best_size:
                        best_set = candidate
                        best_size = len(best_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
