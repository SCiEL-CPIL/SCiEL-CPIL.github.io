"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a // gcd(a, b) * b

def _lcm_list(lst):
    return reduce(_lcm, lst)

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    Strategy: 
    1. Precompute integer weights w_i = LCM(1..22) / i.
    2. Use a hybrid validity check: Python sets for small k, NumPy for large k.
    3. Initialize with multiple greedy passes.
    4. Use Local Search with Add, Swap, and 2-Swap moves to maximize cardinality.
    """
    N = 22
    universe = list(range(1, N + 1))
    
    # Precompute LCM of 1..22
    L = _lcm_list(universe)
    
    # Integer weights: w_i = L // i
    weights = [0] * (N + 1)
    for i in universe:
        weights[i] = L // i

    def is_dissociated(subset):
        """
        Check if the subset of integer weights has all distinct subset sums.
        Returns True if valid, False otherwise.
        """
        k = len(subset)
        if k == 0:
            return True
        
        # For small k, use set for speed in Python
        if k <= 15:
            sums = {0}
            for i in subset:
                w = weights[i]
                new_sums = set()
                for s in sums:
                    ns = s + w
                    if ns in sums or ns in new_sums:
                        return False
                    new_sums.add(ns)
                sums.update(new_sums)
            return True
        else:
            # For larger k, use NumPy
            # Generate all subset sums
            sums = np.array([0], dtype=np.int64)
            for i in subset:
                w = weights[i]
                # sums = np.concatenate([sums, sums + w])
                # Check for duplicates early? 
                # np.unique is O(N log N). 
                # We can check if len(sums) == len(np.unique(sums)) at the end.
                # But intermediate steps might be huge.
                # 2^16 = 65536. 2^20 = 1M. 
                # Memory is fine.
                sums = np.concatenate((sums, sums + w))
            
            # Check uniqueness
            # Using np.unique sorts the array, which is O(N log N)
            # Alternatively, we can use a set on the numpy array? No, that's slow.
            # np.unique is best.
            if len(np.unique(sums)) != len(sums):
                return False
            return True

    # Greedy construction with multiple random orderings
    best_set = []
    best_size = 0
    
    # Try a few deterministic orderings first (sorted by weight, etc.)
    orders = [
        universe[:],
        sorted(universe, key=lambda x: weights[x]),
        sorted(universe, key=lambda x: -weights[x]),
    ]
    
    # Add random orderings
    for _ in range(20):
        random_order = universe[:]
        random.shuffle(random_order)
        orders.append(random_order)
        
    for order in orders:
        S = []
        for i in order:
            candidate = S + [i]
            if is_dissociated(candidate):
                S = candidate
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Local Search
    current_set = best_set[:]
    current_size = len(current_set)
    
    start_time = time.time()
    timeout = 55.0 # Leave buffer
    
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() - start_time < timeout:
        if no_improvement_count > max_no_improvement:
            # Restart from best_set with a perturbation
            current_set = best_set[:]
            current_size = len(current_set)
            # Perturb: remove 2 random elements
            if current_size > 2:
                to_remove = random.sample(current_set, 2)
                current_set = [x for x in current_set if x not in to_remove]
                current_size = len(current_set)
            no_improvement_count = 0
            continue

        move_type = random.random()
        
        if move_type < 0.3:
            # Add move
            candidates_to_add = [i for i in universe if i not in current_set]
            if candidates_to_add:
                i = random.choice(candidates_to_add)
                new_set = current_set + [i]
                if is_dissociated(new_set):
                    current_set = new_set
                    current_size = len(current_set)
                    no_improvement_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

        elif move_type < 0.6:
            # Swap move (remove 1, add 1)
            if current_size > 0:
                i_remove = random.choice(current_set)
                candidates_to_add = [i for i in universe if i not in current_set]
                if candidates_to_add:
                    i_add = random.choice(candidates_to_add)
                    new_set = [x for x in current_set if x != i_remove] + [i_add]
                    if is_dissociated(new_set):
                        current_set = new_set
                        # Size remains same, but we explore
                        no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

        else:
            # 2-Swap move (remove 2, add 2)
            if current_size >= 2:
                to_remove = random.sample(current_set, 2)
                candidates_to_add = [i for i in universe if i not in current_set]
                if len(candidates_to_add) >= 2:
                    to_add = random.sample(candidates_to_add, 2)
                    new_set = [x for x in current_set if x not in to_remove] + to_add
                    if is_dissociated(new_set):
                        current_set = new_set
                        no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
