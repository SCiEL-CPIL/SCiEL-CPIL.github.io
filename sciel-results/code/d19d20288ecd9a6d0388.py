"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s safety margin
    
    n = 256
    w = [Fraction(1, i) for i in range(1, n+1)]
    
    # Precompute conflict pairs: for each x, list of (a,b) where a<b and {a,b,x} is a 3-AP
    conflict_pairs = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        pairs = set()
        for d in range(1, min(x-1, n-x)+1):
            a, b = x-d, x+d
            pairs.add((a, b))
        for d in range(1, (n-x)//2 + 1):
            a, b = x+d, x+2*d
            pairs.add((a, b))
        for d in range(1, x//2 + 1):
            a, b = x-2*d, x-d
            pairs.add((a, b))
        conflict_pairs[x] = list(pairs)
    
    def can_add_x(x, S):
        for a, b in conflict_pairs[x]:
            if a in S and b in S:
                return False
        return True
    
    # Greedy initialization
    greedy_set = set()
    for x in range(1, n+1):
        if can_add_x(x, greedy_set):
            greedy_set.add(x)
    
    best_weight = sum(Fraction(1, x) for x in greedy_set)
    best_set = set(greedy_set)
    
    # Local search with multiple improvement passes
    improved = True
    while improved:
        improved = False
        current_set = set(best_set)
        current_weight = best_weight
        
        # Pass 1: Try adding each excluded element
        for x in range(1, n+1):
            if x in current_set:
                continue
            if can_add_x(x, current_set):
                new_weight = current_weight + w[x-1]
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_set = current_set | {x}
                    improved = True
                    break
        
        if improved:
            continue
        
        # Pass 2: Try 1-for-1 swaps
        for x in range(1, n+1):
            if x not in current_set:
                continue
            new_set = current_set - {x}
            for y in range(1, n+1):
                if y in new_set or y == x:
                    continue
                if can_add_x(y, new_set):
                    new_weight = current_weight - w[x-1] + w[y-1]
                    if new_weight > best_weight:
                        best_weight = new_weight
                        best_set = new_set | {y}
                        improved = True
                        break
            if improved:
                break
        if time.time() > deadline:
            break
        
        if improved:
            continue
        
        # Pass 3: Try 1-for-2 swaps
        for x in range(1, n+1):
            if x not in current_set:
                continue
            new_set = current_set - {x}
            candidates = [y for y in range(1, n+1) if y not in new_set and y != x]
            for i, y in enumerate(candidates):
                if can_add_x(y, new_set):
                    set_with_y = new_set | {y}
                    for z in candidates[i+1:]:
                        if z == y:
                            continue
                        if can_add_x(z, set_with_y):
                            new_weight = current_weight - w[x-1] + w[y-1] + w[z-1]
                            if new_weight > best_weight:
                                best_weight = new_weight
                                best_set = set_with_y | {z}
                                improved = True
                                break
                if time.time() > deadline:
                    break
            if improved or time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
        # Pass 4: Try 2-for-2 swaps
        if not improved and time.time() < deadline - 5:
            S_list = sorted(best_set)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    x, y = S_list[i], S_list[j]
                    new_set = best_set - {x, y}
                    candidates = [z for z in range(1, n+1) if z not in new_set and z != x and z != y]
                    for k, a in enumerate(candidates):
                        if can_add_x(a, new_set):
                            set_a = new_set | {a}
                            for b in candidates[k+1:]:
                                if b == a:
                                    continue
                                if can_add_x(b, set_a):
                                    new_weight = best_weight - w[x-1] - w[y-1] + w[a-1] + w[b-1]
                                    if new_weight > best_weight:
                                        best_weight = new_weight
                                        best_set = set_a | {b}
                                        improved = True
                                        break
                        if time.time() > deadline:
                            break
                    if improved or time.time() > deadline:
                        break
                if improved or time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
