"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    """
    deadline = time.time() + 55  # safety margin before 90s limit
    
    # Start with a greedy baseline (from seed)
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    baseline = set(selected)
    best = baseline
    best_size = len(baseline)
    
    if time.time() > deadline:
        return list(baseline)
    
    # Try to improve by removing elements that cause subset-sum-200 conflicts
    # and adding others
    # First, check if the baseline itself is valid (it should be by construction)
    
    # Strategy: Try random swaps - remove one element, add another not in set
    # to see if we can maintain validity while increasing size
    
    # Better strategy: Since we want max cardinality, try to find a larger valid set
    # Use local search: randomly perturb and repair
    
    def is_valid(S):
        """Check if no subset of S sums to exactly 200. S is a set."""
        # Use bitset DP
        reach = 1  # bit i set means sum i is reachable
        for x in S:
            reach |= (reach << x)
            # Early exit if 200 is reachable
            if reach & (1 << 200):
                return False
        return not (reach & (1 << 200))
    
    # Verify baseline
    if not is_valid(baseline):
        # Fallback: use a safe construction
        # Take all numbers > 100? No, we only have [1,100]
        # Take numbers such that no two (or more) sum to 200
        # Simple: take {101, 102, ..., 200}? No, max is 100
        # If all numbers <= 100, max sum of two is 200 (100+100 but distinct)
        # Actually 99+101 > 100, so max pair sum is 99+100=199 < 200
        # So any subset of [1,100] where no subset sums to 200...
        # Wait, 100+99+... could reach 200
        # Safe set: {100, 99, 98, ..., 51} - all > 50
        # Any subset of {51..100}: min sum of two is 51+52=103, max is 100+99=199
        # So no two distinct elements sum to 200. But three or more?
        # 51+52+53=156, 51+52+...+99 could exceed 200
        # Actually sum of 51+52+...+99 is large. We need to be careful.
        # Simplest safe set: all numbers in (100, 200] but we only have up to 100
        # Let's just use the greedy result which is known to be valid
        pass
    
    # Local search with random restarts
    rng = random.Random(42)
    
    def try_local_search(start_set, max_iters=50000):
        """Try to improve the set via local search."""
        current = set(start_set)
        best_local = set(current)
        best_local_size = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding a random element not in current
            available = [x for x in range(1, 101) if x not in current]
            if not available:
                break
            
            # Try removing a random element from current
            to_remove = rng.choice(list(current))
            to_add = rng.choice(available)
            
            new_set = (current - {to_remove}) | {to_add}
            
            if len(new_set) <= len(current):
                continue  # Only consider same or larger sets
            
            if is_valid(new_set):
                current = new_set
                if len(current) > best_local_size:
                    best_local = set(current)
                    best_local_size = len(current)
                    if best_local_size >= 100:
                        break
        
        # Also try pure additions
        for _ in range(max_iters // 2):
            if time.time() > deadline:
                break
            
            available = [x for x in range(1, 101) if x not in current]
            if not available:
                break
            
            to_add = rng.choice(available)
            new_set = current | {to_add}
            
            if is_valid(new_set):
                current = new_set
                if len(current) > best_local_size:
                    best_local = set(current)
                    best_local_size = len(current)
                    if best_local_size >= 100:
                        break
        
        return best_local
    
    # Run local search from baseline
    improved = try_local_search(baseline, max_iters=100000)
    if len(improved) > best_size:
        best = improved
        best_size = len(improved)
    
    if time.time() > deadline:
        return list(best)
    
    # Try more aggressive search: start from random large sets and repair
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start with a random subset of size ~80
        rand_set = set(rng.sample(range(1, 101), 80))
        
        # Repair: remove elements that cause invalidity
        while not is_valid(rand_set) and len(rand_set) > 10:
            # Find a problematic element and remove it
            removed = False
            for x in list(rand_set):
                test_set = rand_set - {x}
                if is_valid(test_set):
                    rand_set = test_set
                    removed = True
                    break
            if not removed:
                break
        
        if len(rand_set) > best_size:
            best = rand_set
            best_size = len(rand_set)
        
        # Local search from this starting point
        improved = try_local_search(rand_set, max_iters=50000)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    return list(best)
# EVOLVE-BLOCK-END
