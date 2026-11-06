"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflict graph: edge (a,b) if (a+b) | (a*b)
    # conflict[a][b] = True means a and b cannot both be in the set
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True
    
    # Helper: check if x can be added to current set
    def can_add(x, current_set):
        for y in current_set:
            if conflict[x][y]:
                return False
        return True
    
    # Greedy: start with all odds (known valid), then try to add evens
    # Also try different orderings
    
    best_set = set()
    
    # Strategy 1: Greedy by adding numbers in various orders
    # Start with odd numbers
    odds = [x for x in numbers if x % 2 == 1]
    evens = [x for x in numbers if x % 2 == 0]
    
    # Try multiple random orderings and greedy approaches
    best_size = 0
    best_result = odds[:]
    
    # Approach 1: Start with odds, greedily add evens
    current = set(odds)
    for e in evens:
        if can_add(e, current):
            current.add(e)
    if len(current) > best_size:
        best_size = len(current)
        best_result = list(current)
    
    # Approach 2: Randomized greedy - pick random order, add if possible
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = numbers[:]
        random.shuffle(order)
        current = set()
        for x in order:
            if can_add(x, current):
                current.add(x)
        if len(current) > best_size:
            best_size = len(current)
            best_result = list(current)
    
    # Approach 3: Local search - start from best, try swaps
    if time.time() < deadline:
        current = set(best_result)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            in_set = list(current)
            not_in_set = [x for x in numbers if x not in current]
            random.shuffle(not_in_set)
            for out in in_set:
                if time.time() > deadline:
                    break
                for inp in not_in_set:
                    # Try swapping out for inp
                    if conflict[inp][out]:
                        continue
                    # Check inp against all other elements in set
                    ok = True
                    for y in current:
                        if y != out and conflict[inp][y]:
                            ok = False
                            break
                    if ok:
                        # Also check if removing out and adding inp is valid
                        # (it is, since inp doesn't conflict with any remaining element)
                        current.remove(out)
                        current.add(inp)
                        improved = True
                        break
                if improved:
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = list(current)
    
    # Approach 4: Simulated annealing-style local search
    if time.time() < deadline:
        current = set(best_result)
        current_size = len(current)
        T = 1.0
        for step in range(10000):
            if time.time() > deadline:
                break
            if T < 0.01:
                break
            in_set = list(current)
            not_in_set = [x for x in numbers if x not in current]
            if not in_set or not not_in_set:
                break
            out = random.choice(in_set)
            inp = random.choice(not_in_set)
            
            if conflict[inp][out]:
                T *= 0.999
                continue
            
            # Check if inp conflicts with any other element in current (excluding out)
            ok = True
            for y in current:
                if y != out and conflict[inp][y]:
                    ok = False
                    break
            
            if ok:
                # Valid swap
                current.remove(out)
                current.add(inp)
                T *= 0.999
            else:
                # Invalid swap, accept with probability exp(-1/T)
                import math
                if random.random() < math.exp(-1.0 / T):
                    # This would create a conflict, so we can't actually do this
                    # Skip - we can't have conflicts
                    pass
                T *= 0.999
        
        if len(current) > best_size:
            best_size = len(current)
            best_result = list(current)
    
    # Final: try to add any remaining numbers to best_result
    current = set(best_result)
    for x in numbers:
        if x not in current and can_add(x, current):
            current.add(x)
    
    best_result = list(current)
    
    # Ensure nonempty
    if not best_result:
        best_result = [1]
    
    return best_result
# EVOLVE-BLOCK-END
