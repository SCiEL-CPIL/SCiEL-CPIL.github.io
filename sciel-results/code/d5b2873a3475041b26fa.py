import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Build conflict: conflict[i][j] = True if lcm(candidates[i], candidates[j]) <= 256
    # Use bitmasks
    conflict = [0] * n
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if math.lcm(a, b) <= N:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    weights = [1.0 / candidates[i] for i in range(n)]
    
    # Sort by descending weight (ascending value)
    order = sorted(range(n), key=lambda i: -weights[i])
    new_weights = [weights[order[i]] for i in range(n)]
    
    # Rebuild conflict in new order
    new_conflict = [0] * n
    for i in range(n):
        old_i = order[i]
        for j in range(i + 1, n):
            old_j = order[j]
            if conflict[old_i] & (1 << old_j):
                new_conflict[i] |= (1 << j)
                new_conflict[j] |= (1 << i)
    
    # Suffix sums for bounding
    suffix_w = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_w[i] = suffix_w[i + 1] + new_weights[i]
    
    best_weight = 0.0
    best_sel = 0
    
    # Branch and bound with iterative deepening on the bound
    # Use a stack to avoid recursion depth issues
    # State: (node_idx, available_mask, current_weight, current_sel)
    
    # We'll use a priority queue for best-first search to find good solutions early
    import heapq
    
    # Start with greedy to get a good initial bound
    greedy_sel = 0
    greedy_w = 0.0
    available = (1 << n) - 1
    for i in range(n):
        if available & (1 << i):
            greedy_sel |= (1 << i)
            greedy_w += new_weights[i]
            available &= ~new_conflict[i]
            available &= ~(1 << i)
    
    best_weight = greedy_w
    best_sel = greedy_sel
    
    # Now do branch and bound to improve
    # Use DFS with pruning
    
    stack = [(0, (1 << n) - 1, 0.0, 0)]
    nodes_visited = 0
    
    while stack:
        if time.time() > deadline:
            break
        
        node_idx, avail, cur_w, cur_sel = stack.pop()
        nodes_visited += 1
        
        if node_idx >= n:
            if cur_w > best_weight:
                best_weight = cur_w
                best_sel = cur_sel
            continue
        
        # Bound
        if cur_w + suffix_w[node_idx] <= best_weight:
            continue
        
        # Try include
        if avail & (1 << node_idx):
            new_avail = avail & ~(1 << node_idx)
            new_avail &= ~new_conflict[node_idx]
            new_avail &= ~((1 << (node_idx + 1)) - 1)
            stack.append((node_idx + 1, new_avail, cur_w + new_weights[node_idx], 
                         cur_sel | (1 << node_idx)))
        
        # Try exclude
        new_avail = avail & ~(1 << node_idx)
        new_avail &= ~((1 << (node_idx + 1)) - 1)
        stack.append((node_idx + 1, new_avail, cur_w, cur_sel))
    
    # Convert back
    selected_new = []
    m = best_sel
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        selected_new.append(idx)
        m ^= lsb
    
    result = [candidates[order[i]] for i in selected_new]
    
    if not result:
        result = [2]
    
    return result
