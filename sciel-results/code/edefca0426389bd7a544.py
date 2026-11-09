"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edge_pos = {}
    for i, (u, v) in enumerate(P['edge_order']):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i
    
    def get_score(cycle):
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            pos = edge_pos[(u, v)]
            if pos > max_pos:
                max_pos = pos
        return max_pos + 1 # 1-based index

    def greedy_build(start):
        path = [start]
        visited = {start}
        while len(path) < n:
            last = path[-1]
            best_next = -1
            best_pos = float('inf')
            for v in range(n):
                if v not in visited:
                    # Avoid closing the cycle too early
                    if len(path) == n - 1 and v == start:
                        continue
                    pos = edge_pos[(last, v)]
                    if pos < best_pos:
                        best_pos = pos
                        best_next = v
            if best_next == -1:
                return None
            path.append(best_next)
            visited.add(best_next)
        
        # Check if it's a valid Hamiltonian cycle (last connects to first)
        if edge_pos[(path[-1], start)] is not None:
            return path
        return None

    def two_opt(cycle):
        best_cycle = cycle[:]
        best_score = get_score(best_cycle)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() - start_time > 85:
                    return best_cycle
                for j in range(i + 1, n):
                    if time.time() - start_time > 85:
                        return best_cycle
                    # Reverse segment i+1 to j
                    # New edges: (i, j) and (i-1, i+1)
                    # Old edges: (i, i+1) and (j, j+1)
                    u1 = best_cycle[i]
                    v1 = best_cycle[(i + 1) % n]
                    u2 = best_cycle[j]
                    v2 = best_cycle[(j + 1) % n]
                    
                    # Check if edges are distinct to avoid trivial swaps
                    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                        continue
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue

                    pos_old1 = edge_pos[(u1, v1)]
                    pos_old2 = edge_pos[(u2, v2)]
                    pos_new1 = edge_pos[(u1, u2)]
                    pos_new2 = edge_pos[(v1, v2)]
                    
                    old_max = max(pos_old1, pos_old2)
                    new_max = max(pos_new1, pos_new2)
                    
                    # This is a heuristic: if the max of the changed edges is lower, it might improve the global max.
                    # A more rigorous check would compute the full score, but that's O(N).
                    # Let's try if the worst of the new edges is better than the worst of the old edges.
                    if new_max < old_max:
                        # Perform swap
                        best_cycle[i+1:j+1] = reversed(best_cycle[i+1:j+1])
                        new_score = get_score(best_cycle)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            # Revert
                            best_cycle[i+1:j+1] = reversed(best_cycle[i+1:j+1])
        return best_cycle

    best_cycle = None
    best_score = float('inf')
    
    # Try multiple starting vertices for greedy build
    for start in range(n):
        if time.time() - start_time > 85:
            break
        cycle = greedy_build(start)
        if cycle:
            score = get_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle[:]
                
    if best_cycle:
        # Refine with 2-opt
        best_cycle = two_opt(best_cycle)
        best_score = get_score(best_cycle)
        
    if best_cycle:
        return best_cycle
    else:
        # Fallback to identity if greedy fails (should not happen for complete graph)
        return list(range(n))

# EVOLVE-BLOCK-END
