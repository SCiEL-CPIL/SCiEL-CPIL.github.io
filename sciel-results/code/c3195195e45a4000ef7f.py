"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    all_elements = list(range(1, N + 1))
    
    # Pre-compute LCM table for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    # Find all bad triples: (a, b, c) with a < b < c where lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, N + 1))
    
    # Build conflict map: for each element v, the set of pairs (u, w) such that {v, u, w} is bad
    conflict_map = defaultdict(set)
    for a, b, c in bad_triples:
        conflict_map[a].add((b, c))
        conflict_map[b].add((a, c))
        conflict_map[c].add((a, b))
    
    # Helper: check if adding element v to selected_set creates a bad triple
    def can_add(v, selected_set):
        for u, w in conflict_map[v]:
            if u in selected_set and w in selected_set:
                return False
        return True
    
    # Strategy: Start with prime powers (numbers of form p^k), then greedily add others
    prime_powers = set()
    for i in range(1, N + 1):
        x = i
        valid = True
        while x > 1:
            p = None
            for d in range(2, int(x ** 0.5) + 1):
                if x % d == 0:
                    p = d
                    break
            if p is None:
                p = x
            if x % p != 0:
                valid = False
                break
            x //= p
        if valid:
            prime_powers.add(i)
    
    prime_powers.add(1)
    
    selected_set = set(prime_powers)
    
    # Sort remaining elements by number of conflicts (ascending)
    remaining = [x for x in all_elements if x not in selected_set]
    remaining.sort(key=lambda x: len(conflict_map[x]))
    
    for v in remaining:
        if time.time() > deadline:
            break
        if can_add(v, selected_set):
            selected_set.add(v)
    
    if time.time() > deadline:
        return list(selected_set)
    
    best_set = set(selected_set)
    best_size = len(best_set)
    
    # Local search: 1-swap (remove one, add one)
    improved = True
    max_iters = 50
    while improved and max_iters > 0 and time.time() < deadline:
        improved = False
        max_iters -= 1
        current_list = list(selected_set)
        
        for v in current_list:
            if time.time() > deadline:
                break
            
            test_set = set(selected_set)
            test_set.discard(v)
            
            for w in all_elements:
                if w in test_set:
                    continue
                if can_add(w, test_set):
                    new_set = test_set | {w}
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_size = len(best_set)
                        improved = True
                        break
            if improved:
                selected_set = best_set
                break
    
    if time.time() > deadline:
        return list(best_set)
    
    # 2-swap local search (remove two, add two) - limited
    for _ in range(30):
        if time.time() > deadline:
            break
        
        current_list = list(selected_set)
        found_improvement = False
        
        for i in range(len(current_list)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(current_list)):
                v1, v2 = current_list[i], current_list[j]
                test_set = set(selected_set)
                test_set.discard(v1)
                test_set.discard(v2)
                
                candidates = [x for x in all_elements if x not in test_set]
                for w1 in candidates:
                    if time.time() > deadline:
                        break
                    if not can_add(w1, test_set):
                        continue
                    test_set2 = test_set | {w1}
                    for w2 in candidates:
                        if w2 == w1 or w2 in test_set2:
                            continue
                        if can_add(w2, test_set2):
                            new_set = test_set2 | {w2}
                            if len(new_set) > best_size:
                                best_set = new_set
                                best_size = len(best_set)
                                found_improvement = True
                                break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            break
        selected_set = best_set
    
    # Ensure non-empty
    if not best_set:
        best_set = {1}
    
    return list(best_set)

# EVOLVE-BLOCK-END
