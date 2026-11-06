import math
import time
import random

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    all_moduli = list(range(2, 129))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcd_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_matrix[i][j] = math.gcd(all_moduli[i], all_moduli[j])
    
    # Compute degree (number of moduli that share a common factor > 1)
    degree = [0]*n
    for i in range(n):
        for j in range(n):
            if i != j and gcd_matrix[i][j] > 1:
                degree[i] += 1
    
    def greedy_build(order):
        assigned = []
        residues = {}
        for idx in order:
            m = all_moduli[idx]
            found = False
            for r in range(m):
                valid = True
                for (r2, idx2) in assigned:
                    g = gcd_matrix[idx][idx2]
                    if g > 1 and (r % g) == (r2 % g):
                        valid = False
                        break
                if valid:
                    assigned.append((r, idx))
                    residues[idx] = r
                    found = True
                    break
        return [[r, all_moduli[idx]] for (r, idx) in assigned]
    
    best_result = []
    best_count = 0
    
    # Multiple greedy orders
    # Degree-based (most constrained first)
    order_degree = sorted(range(n), key=lambda i: degree[i], reverse=True)
    result = greedy_build(order_degree)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result
    
    # Size descending
    order_size = sorted(range(n), key=lambda i: all_moduli[i], reverse=True)
    result = greedy_build(order_size)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result
    
    # Size ascending
    order_size_asc = sorted(range(n), key=lambda i: all_moduli[i])
    result = greedy_build(order_size_asc)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result
    
    # Random restarts
    for seed in range(100):
        if time.time() > deadline:
            break
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        result = greedy_build(order)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Now do local search / improvement on the best solution
    # Try to add more moduli or swap residues
    def local_search(initial_result):
        # Build state from result
        state = {}
        for (r, m) in initial_result:
            idx = all_moduli.index(m)
            state[idx] = r
        
        current_count = len(state)
        best_local = current_count
        best_state = dict(state)
        
        # Try adding unassigned moduli
        unassigned = [i for i in range(n) if i not in state]
        
        # Try to add each unassigned modulus
        for idx in unassigned:
            if time.time() > deadline:
                break
            m = all_moduli[idx]
            for r in range(m):
                valid = True
                for idx2, r2 in state.items():
                    g = gcd_matrix[idx][idx2]
                    if g > 1 and (r % g) == (r2 % g):
                        valid = False
                        break
                if valid:
                    state[idx] = r
                    current_count += 1
                    break
        
        if current_count > best_local:
            best_local = current_count
            best_state = dict(state)
        
        return [[r, all_moduli[idx]] for idx, r in best_state.items()]
    
    improved = local_search(best_result)
    if len(improved) > best_count:
        best_count = len(improved)
        best_result = improved
    
    # If still have time, do more random restarts with local search
    for seed in range(100, 500):
        if time.time() > deadline:
            break
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        result = greedy_build(order)
        improved = local_search(result)
        if len(improved) > best_count:
            best_count = len(improved)
            best_result = improved
    
    if not best_result:
        best_result = [[1, 2]]
    
    return best_result
