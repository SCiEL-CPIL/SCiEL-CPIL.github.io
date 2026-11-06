import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin

    n = 5040
    target = 4979

    # Generate divisors
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort()  # Sort ascending to allow early breaking in loops
    D = len(divisors)
    
    # BFS State: (current_sum, last_index_used)
    # We want to minimize the number of divisors (depth).
    # For a given sum, a smaller last_index is better as it allows more future choices.
    
    # best_last_index[s] stores the minimum last_index seen for sum s.
    # Initialize with a value larger than any index.
    INF = D + 1
    best_last_index = [INF] * (target + 1)
    best_last_index[0] = -1
    
    # To reconstruct the path, we store the parent state for each sum update.
    # parent[s] = (prev_sum, prev_index, current_index)
    parent = {}
    
    queue = deque()
    queue.append((0, -1))
    
    found = False
    final_sum = 0
    
    while queue:
        if time.time() > deadline:
            # Fallback: return empty list or best effort? 
            # The problem implies a solution exists. 
            # If we time out, we might return a partial or invalid result.
            # However, BFS on this small graph is very fast.
            break
            
        s, i = queue.popleft()
        
        if s == target:
            found = True
            final_sum = s
            break
            
        # Try adding divisors with index > i
        # Since divisors are sorted ascending, if s + d[j] > target, we can break.
        for j in range(i + 1, D):
            d = divisors[j]
            new_s = s + d
            if new_s > target:
                break
            
            # If we found a way to reach new_s with a smaller last index, update
            if j < best_last_index[new_s]:
                best_last_index[new_s] = j
                parent[new_s] = (s, i, j)
                queue.append((new_s, j))
                
    if not found:
        # If not found via BFS (should not happen for this instance), 
        # we might need a fallback, but the instance is known to be solvable.
        # For safety, return empty list or raise. The contract says "at most 256...".
        # Usually implies finding *a* set. If none found, maybe return []?
        # But the task says "Minimize the number...". 
        # Let's assume BFS finds it.
        return []
        
    # Reconstruct path
    path_indices = []
    current_s = target
    while current_s != 0:
        prev_s, prev_i, curr_j = parent[current_s]
        path_indices.append(curr_j)
        current_s = prev_s
        
    # path_indices contains indices in reverse order (last added to first added)
    # We just need the divisors.
    selected_divisors = [divisors[idx] for idx in path_indices]
    
    return selected_divisors
