"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd

# LCM of 1..22
_LCM = 232792560

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit

    def is_valid(subset):
        """Check if all reciprocal subset sums of subset are distinct using integer scaling."""
        if not subset:
            return True
        weights = [_LCM // i for i in subset]
        sums = {0}
        for w in weights:
            new_sums = {s + w for s in sums}
            if new_sums & sums:
                return False
            sums |= new_sums
        return True

    def try_greedy(order):
        """
        Greedily build a set from 'order' using incremental sum tracking.
        """
        current_set = []
        sums = {0}
        
        for x in order:
            if x in current_set:
                continue
            w = _LCM // x
            
            # Check for collisions
            new_sums = {s + w for s in sums}
            if new_sums & sums:
                continue
            
            sums |= new_sums
            current_set.append(x)
            
        return current_set

    best = []
    
    # 1. Deterministic orderings
    base_orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        # Powers of 2 first
        [1, 2, 4, 8, 16] + [x for x in range(1, 23) if x not in [1, 2, 4, 8, 16]],
        # Primes first
        [2, 3, 5, 7, 11, 13, 17, 19] + [x for x in range(1, 23) if x not in [2, 3, 5, 7, 11, 13, 17, 19]],
        # Large numbers first
        list(range(22, 0, -1)),
        # Mix of small and large
        [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 21, 9, 18, 11, 22, 13, 15, 17, 19],
    ]
    
    for order in base_orders:
        if time.time() > deadline:
            break
        res = try_greedy(order)
        if len(res) > len(best):
            best = res

    # 2. Randomized greedy
    random.seed(42)
    powers_of_2 = [1, 2, 4, 8, 16]
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        order = list(range(1, 23))
        random.shuffle(order)
        
        # Sometimes try powers of 2 first
        if random.random() < 0.2:
            rest = [x for x in order if x not in powers_of_2]
            order = powers_of_2 + rest
            
        res = try_greedy(order)
        if len(res) > len(best):
            best = res

    # 3. Local Search: Try to improve best set via swaps
    # This explores the non-convex landscape by swapping elements in and out
    if len(best) >= 1 and time.time() < deadline - 5.0:
        best_set = set(best)
        
        # Try to find a larger set by removing one element and greedily adding others
        # Or by swapping an element for a different one
        improved = False
        for _ in range(1000):  # Limited iterations
            if time.time() > deadline - 2.0:
                break
            
            # Pick a random element from best to remove
            if not best:
                break
            remove_idx = random.randint(0, len(best) - 1)
            removed_elem = best[remove_idx]
            
            # Create a new candidate set without the removed element
            candidate = [x for x in best if x != removed_elem]
            candidate_set = set(candidate)
            
            # Rebuild sums for the candidate set
            sums = {0}
            valid_base = True
            for x in candidate:
                w = _LCM // x
                new_sums = {s + w for s in sums}
                if new_sums & sums:
                    valid_base = False
                    break
                sums |= new_sums
            
            if not valid_base:
                continue
            
            # Try to add elements not in candidate_set
            remaining = [x for x in range(1, 23) if x not in candidate_set]
            random.shuffle(remaining)
            
            # Greedily add from remaining
            new_candidate = candidate[:]
            for x in remaining:
                w = _LCM // x
                new_sums = {s + w for s in sums}
                if not (new_sums & sums):
                    sums |= new_sums
                    new_candidate.append(x)
            
            if len(new_candidate) > len(best):
                best = new_candidate
                improved = True
                # Restart local search with the new best? Or just continue
                # For now, just update best and continue loop

    if best:
        return best
    
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
