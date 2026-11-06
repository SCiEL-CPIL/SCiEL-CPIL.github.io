"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    N = 256
    # Build conflict: conflict[a] = set of b >= a where a+b is a perfect square
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict[a] contains all b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Start with all numbers ≡ 1 mod 3 (guaranteed independent set)
    base_set = set(range(1, N + 1, 3))
    
    # Try to greedily add elements from other residue classes
    # Sort candidates by degree (fewest conflicts first) to maximize additions
    candidates = [x for x in range(1, N + 1) if x % 3 != 1]
    
    # Precompute conflict count with current set for each candidate
    def conflicts_with_set(x, current_set):
        return len(conflict[x] & current_set)
    
    # Greedy: repeatedly add the candidate with fewest conflicts
    current_set = set(base_set)
    remaining = list(candidates)
    
    while remaining:
        # Check time
        if time.time() > deadline:
            break
        
        # Find candidate with minimum conflicts
        best_x = None
        best_conf = None
        for x in remaining:
            c = conflicts_with_set(x, current_set)
            if best_conf is None or c < best_conf:
                best_conf = c
                best_x = x
        
        if best_x is None:
            break
        
        if best_conf == 0:
            current_set.add(best_x)
            remaining.remove(best_x)
        else:
            # If minimum conflicts > 0, no more can be added greedily
            break
    
    # Try local search: remove one element, try to add two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = list(current_set)
        for i in range(len(current_list)):
            if time.time() > deadline:
                break
            x = current_list[i]
            new_set = current_set - {x}
            # Try to add candidates
            for y in candidates:
                if y in new_set:
                    continue
                if len(conflict[y] & new_set) == 0:
                    new_set.add(y)
                    # Try to add one more
                    for z in candidates:
                        if z in new_set:
                            continue
                        if len(conflict[z] & new_set) == 0:
                            new_set.add(z)
                            if len(new_set) > len(current_set):
                                current_set = new_set
                                improved = True
                            break
                    break
    
    result = sorted(current_set)
    return result if result else list(range(1, 257, 3))
# EVOLVE-BLOCK-END
