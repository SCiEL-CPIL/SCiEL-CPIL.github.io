import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    items = []
    for i, d in enumerate(denominators):
        w = L // d
        items.append((w, i))
        
    items.sort(key=lambda x: x[0], reverse=True)
    
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items[i][0]
        
    best_deficit = L
    best_indices = []
    
    # Greedy initial solution
    current_sum = 0
    greedy_indices = []
    for w, idx in items:
        if current_sum + w < L:
            current_sum += w
            greedy_indices.append(idx)
    if greedy_indices:
        best_deficit = L - current_sum
        best_indices = greedy_indices[:]
        
    node_count = 0
    last_time_check = time.time()
    
    def search(idx, current_sum, current_indices):
        nonlocal best_deficit, best_indices, node_count, last_time_check
        
        node_count += 1
        if node_count % 10000 == 0:
            if time.time() > deadline:
                return
                
        # Prune 1: Check if we can improve
        max_possible_sum = current_sum + suffix_sums[idx]
        if max_possible_sum <= L - best_deficit:
            return
            
        # Update best if current is better
        current_deficit = L - current_sum
        if current_deficit < best_deficit:
            best_deficit = current_deficit
            best_indices = current_indices[:]
            
        if idx == n:
            return
            
        w, orig_idx = items[idx]
        
        # Branch Include
        if current_sum + w < L:
            current_indices.append(orig_idx)
            search(idx + 1, current_sum + w, current_indices)
            current_indices.pop()
            
        # Branch Exclude
        search(idx + 1, current_sum, current_indices)
        
    search(0, 0, [])
    
    return best_indices
