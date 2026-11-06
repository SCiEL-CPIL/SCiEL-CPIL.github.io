"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave some buffer for 90s limit

    # The task: find up to 80 distinct positive integers <= 4096
    # such that every unordered triple (allowing repetitions) has a unique sum.
    # This is a B_3[1] sequence (Sidon-like for triples).
    
    # Theoretical bound: for |S|=n, number of triples is n(n+1)(n+2)/6.
    # Sums range from 3 to 12288, giving 12286 possible distinct sums.
    # n=40: 11480 <= 12286. OK.
    # n=41: 12411 > 12286. Not OK.
    # So the theoretical maximum is 40.
    
    def run_greedy(order, time_budget, start_time_ref):
        """Greedy algorithm with incremental sum tracking."""
        selected = []
        current_sums = set()
        
        for x in order:
            if time.time() - start_time_ref > time_budget:
                break
            # Check if adding x creates any duplicate triple sum
            # New triples involving x: (x,x,x), (x,x,y), (x,y,y), (x,y,z)
            
            new_sums = set()
            valid = True
            
            # (x, x, x)
            s = 3 * x
            if s in current_sums:
                valid = False
            else:
                new_sums.add(s)
            
            if valid:
                # (x, x, y) for y in selected
                for y in selected:
                    s = 2 * x + y
                    if s in current_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                # (x, y, y) for y in selected
                for y in selected:
                    s = x + 2 * y
                    if s in current_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                # (x, y, z) for y, z in selected, y <= z, y != z
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        y, z = selected[i], selected[j]
                        s = x + y + z
                        if s in current_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                current_sums.update(new_sums)
        
        return selected

    def verify(s):
        """Verify that the set has unique triple sums."""
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = [sum(t) for t in combinations_with_replacement(s, 3)]
        return len(sums) == len(set(sums))

    # Strategy 1: Greedy from 1 upward
    best_set = run_greedy(range(1, 4097), time_limit - (time.time() - start_time), start_time)
    
    # Strategy 2: Greedy from larger numbers downward
    if time.time() - start_time < time_limit - 10:
        s2 = run_greedy(range(4096, 0, -1), time_limit - (time.time() - start_time), start_time)
        if len(s2) > len(best_set):
            best_set = s2
    
    # Strategy 3: Random orderings with different seeds
    random.seed(42)
    for trial in range(20):
        if time.time() - start_time > time_limit - 5:
            break
        order = list(range(1, 4097))
        random.shuffle(order)
        s = run_greedy(order, time_limit - (time.time() - start_time), start_time)
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 4: Biased ordering - favor numbers that are far apart
    if time.time() - start_time < time_limit - 10:
        # Try a sparse sequence with larger gaps
        sparse = list(range(1, 4097, 50))
        s = run_greedy(sparse, time_limit - (time.time() - start_time), start_time)
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 5: Try to extend the best set found so far
    if time.time() - start_time < time_limit - 10:
        base = best_set[:]
        base_set = set(base)
        remaining = [x for x in range(1, 4097) if x not in base_set]
        extended = run_greedy(remaining, time_limit - (time.time() - start_time), start_time)
        if len(extended) > len(base):
            best_set = base + extended
    
    # Verify the result
    if not verify(best_set):
        # Fallback to a simple valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return best_set
# EVOLVE-BLOCK-END
