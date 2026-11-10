import time
import sys
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure with edge indices
    # Map edge (u, v) to a unique index
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        if u < v:
            edge_map[(u, v)] = i
        else:
            edge_map[(v, u)] = i
            
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Precompute edge indices for fast lookup in the closing step
    # We need to check if edge (current, 0) is available
    # We can just look it up in edge_map
    
    def find_hamiltonian_cycle(used_mask, deadline, rng=None):
        visited = [False] * n
        visited[0] = True
        path = [0]
        
        def dfs(current, remaining_count):
            if time.time() > deadline:
                return None
            
            if remaining_count == 0:
                # Check if we can close the cycle to start (vertex 0)
                # Find edge index for (current, 0)
                u, v = current, 0
                if u < v:
                    ek = (u, v)
                else:
                    ek = (v, u)
                if ek in edge_map:
                    idx = edge_map[ek]
                    if not (used_mask & (1 << idx)):
                        return list(path)
                return None
            
            # Get unvisited neighbors
            candidates = []
            for nb, e_idx in adj[current]:
                if not visited[nb]:
                    if not (used_mask & (1 << e_idx)):
                        # Heuristic: count how many unvisited neighbors nb has
                        # Lower count is preferred (Warnsdorff)
                        deg = 0
                        for n2, e2_idx in adj[nb]:
                            if not visited[n2]:
                                if not (used_mask & (1 << e2_idx)):
                                    deg += 1
                        candidates.append((deg, nb, e_idx))
            
            if not candidates:
                return None
                
            # Sort by degree (fewest options first)
            if rng:
                # Add random tie-breaker
                candidates = [(deg, rng.random(), nb, e_idx) for deg, nb, e_idx in candidates]
                candidates.sort(key=lambda x: (x[0], x[1]))
                iter_candidates = [(nb, e_idx) for _, _, nb, e_idx in candidates]
            else:
                candidates.sort(key=lambda x: x[0])
                iter_candidates = [(nb, e_idx) for _, nb, e_idx in candidates]
            
            for nb, e_idx in iter_candidates:
                path.append(nb)
                visited[nb] = True
                
                result = dfs(nb, remaining_count - 1, used_mask | (1 << e_idx))
                if result is not None:
                    return result
                    
                path.pop()
                visited[nb] = False
                
            return None
        
        return dfs(0, n - 1, 0)

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    used_mask = 0
    cycles = []
    
    max_attempts = 5
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        if len(cycles) >= 3:
            break
            
        # First attempt: Deterministic
        cycle = find_hamiltonian_cycle(used_mask, deadline, rng=None)
        
        if not cycle:
            # If deterministic failed, try randomized restarts
            found = False
            for i in range(10):
                if time.time() > deadline:
                    break
                rng = random.Random(i)
                cycle = find_hamiltonian_cycle(used_mask, deadline, rng=rng)
                if cycle:
                    found = True
                    break
            
            if not found:
                break

        if cycle:
            cycles.append(cycle)
            # Mark edges as used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if u < v:
                    ek = (u, v)
                else:
                    ek = (v, u)
                if ek in edge_map:
                    idx = edge_map[ek]
                    used_mask |= (1 << idx)
        else:
            break

    # Ensure result types are correct
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
        
    return result
# EVOLVE-BLOCK-END
