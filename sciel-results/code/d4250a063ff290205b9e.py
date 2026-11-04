import numpy as np
import time
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety

    def int_to_base_q(num, q, n):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    def base_q_to_int(digits, q):
        num = 0
        for d in reversed(digits):
            num = num * q + d
        return num

    # Handle general case fallback if not the specific task
    if q != 3 or n != 7 or radius != 1:
        total_words = q ** n
        words = [int_to_base_q(i, q, n) for i in range(total_words)]
        neighbors = [[] for _ in range(total_words)]
        for i in range(total_words):
            w = words[i]
            neighbors[i].append(i)
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        new_w = w[:]
                        new_w[pos] = new_val
                        j = base_q_to_int(new_w, q)
                        neighbors[i].append(j)
        uncovered = set(range(total_words))
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total_words):
                count = len(uncovered.intersection(neighbors[c]))
                if count > best_count:
                    best_count = count
                    best_c = c
                    if best_count == len(neighbors[best_c]): break
            if best_c == -1: break
            code.append(best_c)
            for nb in neighbors[best_c]:
                if nb in uncovered: uncovered.remove(nb)
            if time.time() > deadline: break
        return [words[c] for c in code]

    # Specific optimized path for q=3, n=7, radius=1
    total_words = 3 ** 7 # 2187
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        neighbors[i].append(i)
        temp = i
        for pos in range(n):
            digit = temp % q
            temp //= q
            base_val = i - digit * (q ** pos)
            for new_val in range(q):
                if new_val != digit:
                    j = base_val + new_val * (q ** pos)
                    neighbors[i].append(j)
    
    # Convert neighbors to sets for faster intersection
    neighbor_sets = [set(nb) for nb in neighbors]
    
    # Initial greedy solution to get an upper bound
    uncovered = set(range(total_words))
    initial_code = []
    while uncovered:
        best_c = -1
        best_count = -1
        for c in range(total_words):
            count = len(uncovered.intersection(neighbor_sets[c]))
            if count > best_count:
                best_count = count
                best_c = c
                if best_count == len(neighbor_sets[best_c]): break
        if best_c == -1: break
        initial_code.append(best_c)
        for nb in neighbors[best_c]:
            if nb in uncovered: uncovered.remove(nb)
        if time.time() > deadline: break
    
    # Prune initial greedy solution
    coverage = np.zeros(total_words, dtype=int)
    for c in initial_code:
        for nb in neighbors[c]:
            coverage[nb] += 1
    pruned_code = initial_code.copy()
    changed = True
    while changed:
        changed = False
        for i in range(len(pruned_code) - 1, -1, -1):
            c = pruned_code[i]
            removable = True
            for nb in neighbors[c]:
                if coverage[nb] == 1:
                    removable = False
                    break
            if removable:
                pruned_code.pop(i)
                for nb in neighbors[c]:
                    coverage[nb] -= 1
                changed = True
                if time.time() > deadline: break
        if time.time() > deadline: break
    
    best_size = len(pruned_code)
    best_solution = pruned_code.copy()
    
    # Branch and Bound
    sys.setrecursionlimit(5000)
    
    def lower_bound(uncovered_set):
        if not uncovered_set:
            return 0
        return (len(uncovered_set) + 14) // 15

    def search(uncovered_set, current_code):
        nonlocal best_size, best_solution
        
        if time.time() > deadline:
            return
            
        if not uncovered_set:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_solution = current_code.copy()
            return
            
        # Lower bound check
        lb = lower_bound(uncovered_set)
        if len(current_code) + lb >= best_size:
            return
            
        # Generate candidates
        candidates = []
        for c in range(total_words):
            count = len(uncovered_set.intersection(neighbor_sets[c]))
            if count > 0:
                candidates.append((count, c))
                
        # Sort by count descending
        candidates.sort(key=lambda x: -x[0])
        
        for count, c in candidates:
            if time.time() > deadline:
                return
                
            new_uncovered = set(uncovered_set)
            for nb in neighbors[c]:
                if nb in new_uncovered:
                    new_uncovered.remove(nb)
                    
            current_code.append(c)
            search(new_uncovered, current_code)
            current_code.pop()
            
    search(set(range(total_words)), [])
    
    return [int_to_base_q(c, q, n) for c in best_solution]
