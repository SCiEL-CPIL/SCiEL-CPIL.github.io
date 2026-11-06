"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    DEADLINE = 140.0  # safety margin before 150s kill
    start_time = time.time()

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_instance(values, time_limit):
        n = len(values)
        # Precompute: for each pair (i, j) with i <= j, check if values[i] + values[j] is in values
        # We'll use a greedy approach with local search
        
        # Strategy: 
        # 1. Start with a good initial solution (e.g., all odds, or greedy)
        # 2. Use local search: try to add elements that don't conflict, or swap to improve
        
        # For efficiency, maintain a set of "forbidden" elements for each candidate
        # Actually, let's use a simpler approach: 
        # - Represent solution as a set of indices
        # - For each element not in solution, check if it can be added
        # - Try random restarts with different initial solutions
        
        # Precompute sum membership
        val_set = set(values)
        
        # For each element, precompute which pairs (a, b) in the set would make a + b = this element
        # Or rather, for each element c, the forbidden pairs are (a, b) where a + b = c
        # When we add c, we must ensure no a, b in current solution with a + b = c
        # Also, if we have a, b in solution, we cannot add c = a + b
        
        # Let's use a bitset approach or just maintain a set
        
        best_solution = set()
        best_size = 0
        
        # Try multiple random restarts
        rng = random.Random(42)
        attempts = 0
        
        while time.time() - start_time < time_limit:
            attempts += 1
            # Random restart: start with a random subset and greedily improve
            # Or start with odds and try to add evens
            
            # Approach: start with empty set, greedily add elements in random order
            # But that's not great. Better: start with a heuristic solution, then local search.
            
            # Heuristic 1: all odd numbers
            sol = set(i for i, v in enumerate(values) if v % 2 != 0)
            
            # Heuristic 2: greedy by some criterion
            # Let's try: sort by |value| descending, add if no conflict
            # Or random order
            
            # Local search: try to add any element that doesn't conflict
            # Then try swaps: remove one, add two?
            
            # First, let's do a greedy improvement: repeatedly add any element that can be added
            changed = True
            while changed:
                changed = False
                # Build forbidden set: elements that would create a conflict
                # For current solution S, an element c can be added if:
                # 1. c not in S
                # 2. No a, b in S with a + b = c (i.e., c - a not in S for any a in S, considering a != b case too)
                # 3. No a in S with 2*a = c
                # Actually, condition is: for all a, b in S, a + b != c
                # Which means: for all a in S, c - a not in S (this covers a=b when 2a=c)
                
                # Precompute S as a set of values for fast lookup
                S_vals = set(values[i] for i in sol)
                
                for i in range(n):
                    if i in sol:
                        continue
                    c = values[i]
                    # Check if c can be added
                    can_add = True
                    for a_val in S_vals:
                        if (c - a_val) in S_vals:
                            can_add = False
                            break
                    if can_add:
                        sol.add(i)
                        S_vals.add(c)
                        changed = True
            
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = set(sol)
            
            # If we found a perfect solution (all elements), break
            if best_size == n:
                break
            
            # Try local search with swaps: remove one element, try to add more
            # Or try a different random restart strategy
            
            # Randomized restart: start with a random subset of size ~n/2 and improve
            if attempts % 5 == 0:
                # Start with random subset
                sol = set(rng.sample(range(n), n // 2))
                # Greedily add
                changed = True
                while changed:
                    changed = False
                    S_vals = set(values[i] for i in sol)
                    for i in range(n):
                        if i in sol:
                            continue
                        c = values[i]
                        can_add = True
                        for a_val in S_vals:
                            if (c - a_val) in S_vals:
                                can_add = False
                                break
                        if can_add:
                            sol.add(i)
                            S_vals.add(c)
                            changed = True
                if len(sol) > best_size:
                    best_size = len(sol)
                    best_solution = set(sol)
        
        return [values[i] for i in sorted(best_solution)]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        time_left = DEADLINE - (time.time() - start_time)
        if time_left <= 0:
            # Fallback: return odds
            results.append([x for x in vals if x % 2 != 0])
        else:
            sol = solve_instance(vals, time_left)
            results.append(sol)
    
    return results
# EVOLVE-BLOCK-END
