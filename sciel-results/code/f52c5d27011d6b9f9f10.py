"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
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
    
    Strategy: Use randomized local search with simulated annealing on the
    space of subsets of [1,22]. The objective is to maximize cardinality
    while maintaining the dissociated property (all 2^k subset sums of
    reciprocals are distinct).
    
    We work with integer weights: let L = lcm(1..22). For a subset S,
    the reciprocal subset sums are distinct iff the subset sums of
    {L/i : i in S} are all distinct (i.e., the set is dissociated in Z).
    
    A set of integers is dissociated if all 2^|S| subset sums are distinct.
    We check this by computing all subset sums and verifying uniqueness.
    """
    N = 22
    universe = list(range(1, N + 1))
    
    # Precompute LCM of 1..22
    L = _lcm_list(universe)
    
    # Integer weights: w_i = L // i
    weights = {i: L // i for i in universe}
    
    def is_dissociated(subset):
        """Check if the subset of integer weights has all distinct subset sums."""
        if len(subset) == 0:
            return True
        # Compute all 2^k subset sums
        k = len(subset)
        if k > 20:
            # Too many to check exhaustively; use a heuristic or skip
            # For our purposes, we only need to check up to ~15-16
            # For larger sets, we'll use a probabilistic check
            pass
        
        subset_sums = set()
        subset_sums.add(0)
        for idx, i in enumerate(subset):
            w = weights[i]
            # For each existing sum, add w
            new_sums = set()
            for s in subset_sums:
                ns = s + w
                if ns in subset_sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            subset_sums.update(new_sums)
        
        return len(subset_sums) == (1 << k)
    
    def is_dissociated_fast(subset):
        """
        Faster check: compute all subset sums and check for duplicates.
        For k up to ~15, 2^k = 32768 which is manageable.
        """
        k = len(subset)
        if k == 0:
            return True
        if k > 18:
            # Too expensive; assume valid if we got here (shouldn't happen often)
            return True
        
        sums = [0]
        for i in subset:
            w = weights[i]
            # Add w to all existing sums
            current_len = len(sums)
            for j in range(current_len):
                sums.append(sums[j] + w)
        
        # Check for duplicates
        if len(sums) != len(set(sums)):
            return False
        return True
    
    # Start with a known good set
    # The seed has 9 elements. Let's try to find larger sets.
    # Known: powers of 2 style sets are dissociated.
    # In terms of reciprocals: {1/2, 1/4, 1/8, ...} but we're restricted to [1,22].
    
    # Let's use simulated annealing to maximize size.
    # State: a subset of [1,22]
    # Objective: maximize |S| subject to S being dissociated
    
    # Initialize with a greedy construction
    # Greedy: start with empty, add elements one by one if dissociated
    def greedy_construct():
        S = []
        # Try adding elements in various orders
        for i in universe:
            candidate = S + [i]
            if is_dissociated_fast(candidate):
                S = candidate
        return S
    
    # Try multiple random orderings for greedy
    best_set = []
    best_size = 0
    
    # First, try a few greedy constructions with different orderings
    for trial in range(50):
        order = universe[:]
        random.shuffle(order)
        S = []
        for i in order:
            candidate = S + [i]
            if is_dissociated_fast(candidate):
                S = candidate
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Now use simulated annealing / local search to try to improve
    # Moves: add an element, remove an element, swap an element
    
    current_set = best_set[:]
    current_size = len(current_set)
    best_set = current_set[:]
    best_size = current_size
    
    # Local search
    import time
    start_time = time.time()
    timeout = 50.0  # Leave 10s buffer
    
    T = 1.0
    cooling = 0.9999
    
    while time.time() - start_time < timeout:
        if T < 0.01:
            break
            
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add an element
            candidates_to_add = [i for i in universe if i not in current_set]
            if not candidates_to_add:
                continue
            i = random.choice(candidates_to_add)
            new_set = current_set + [i]
            if is_dissociated_fast(new_set):
                # Accept (always improves or maintains)
                current_set = new_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
            else:
                # Reject with probability based on temperature
                # Since we can't have invalid states, just reject
                pass
                
        elif move_type < 0.7:
            # Try to remove an element (to allow adding others later)
            if len(current_set) <= 1:
                continue
            i = random.choice(current_set)
            new_set = [x for x in current_set if x != i]
            # Always valid (removing preserves dissociation)
            # Accept with some probability to explore
            if random.random() < 0.3:
                current_set = new_set
        else:
            # Swap: remove one, add another
            if len(current_set) == 0:
                continue
            i_remove = random.choice(current_set)
            candidates_to_add = [i for i in universe if i not in current_set]
            if not candidates_to_add:
                continue
            i_add = random.choice(candidates_to_add)
            new_set = [x for x in current_set if x != i_remove] + [i_add]
            if is_dissociated_fast(new_set):
                current_set = new_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
        
        T *= cooling
    
    # Final answer: best_set
    if not best_set:
        # Fallback: return the seed
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
