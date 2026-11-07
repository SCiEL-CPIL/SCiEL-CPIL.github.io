import time
from collections import deque

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    n = P['n']
    edges = P['edges']
    full_mask = (1 << n) - 1
    
    # Build adjacency bitsets
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute valid parts (cliques and independent sets)
    valid_parts = []
    # Check all non-empty subsets
    # Optimization: Only check subsets up to a certain size? No, n is small.
    # But 2^14 is 16384. Looping through all is fine.
    
    # To speed up clique/ind checks, we can use recursive generation or just bit checks.
    # For a mask m, it's a clique if for all v in m, (m without v) is subset of adj[v].
    # i.e., (m & ~(1<<v)) & ~adj[v] == 0 for all v in m.
    # Similarly for independent set: (m & ~(1<<v)) & adj[v] == 0.
    
    for m in range(1, 1 << n):
        is_clique = True
        is_ind = True
        # Check clique
        mm = m
        while mm:
            v_bit = mm & -mm
            v = v_bit.bit_length() - 1
            # Check if all other nodes in m are neighbors of v
            others = m & ~v_bit
            if others & ~adj[v]:
                is_clique = False
                break
            mm -= v_bit
        
        if not is_clique:
            # Check independent set
            mm = m
            while mm:
                v_bit = mm & -mm
                v = v_bit.bit_length() - 1
                others = m & ~v_bit
                # Check if no other nodes in m are neighbors of v
                if others & adj[v]:
                    is_ind = False
                    break
                mm -= v_bit
        
        if is_clique or is_ind:
            valid_parts.append(m)
            
    if time.time() > deadline:
        # Fallback to greedy if precomputation took too long (unlikely)
        return greedy_construct(n, edges)

    # BFS to find minimum number of parts
    # visited[mask] = (prev_mask, part_mask)
    visited = {0: None}
    queue = deque([0])
    
    while queue:
        if time.time() > deadline:
            # Return best found so far? 
            # Since we haven't found full_mask, we need a valid partition.
            # We can reconstruct a partition from any visited mask that is "large" or just use greedy.
            # But BFS level by level ensures optimality if we find it.
            # If we time out, we return a valid partition from greedy.
            return greedy_construct(n, edges)
            
        current_level_size = len(queue)
        for _ in range(current_level_size):
            m = queue.popleft()
            for v in valid_parts:
                if m & v == 0:
                    new_m = m | v
                    if new_m not in visited:
                        visited[new_m] = (m, v)
                        if new_m == full_mask:
                            # Reconstruct
                            parts = []
                            curr = full_mask
                            while curr != 0:
                                prev, part_mask = visited[curr]
                                # Convert part_mask to list of vertices
                                verts = [i for i in range(n) if (part_mask >> i) & 1]
                                parts.append(verts)
                                curr = prev
                            return parts
                        queue.append(new_m)
                            
    # If BFS finishes without finding full_mask (should not happen for n>0)
    # or if we run out of states
    return greedy_construct(n, edges)

def greedy_construct(n, edges):
    es = {tuple(e) for e in edges}
    es = es | {(b, a) for a, b in es}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            # Check if clique
            is_c = all((u, w) in es for i, u in enumerate(trial) for w in trial[i+1:])
            # Check if independent
            is_i = all((u, w) not in es for i, u in enumerate(trial) for w in trial[i+1:])
            if is_c or is_i:
                part.append(v)
                break
        else:
            parts.append([v])
    return parts
