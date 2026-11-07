"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    added_edges = []
    
    def get_distances():
        dists = [[-1] * n for _ in range(n)]
        for s in range(n):
            dists[s][s] = 0
            queue = [s]
            for i in range(len(queue)):
                u = queue[i]
                for v in adj[u]:
                    if dists[s][v] == -1:
                        dists[s][v] = dists[s][u] + 1
                        queue.append(v)
        return dists

    def is_valid_candidate(u, v):
        if v in adj[u]:
            return False
        if adj[u] & adj[v]:
            return False
        return True

    def evaluate_candidate(u, v):
        adj[u].add(v)
        adj[v].add(u)
        dists = get_distances()
        max_d = 0
        count_gt4 = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i][j]
                if d > max_d:
                    max_d = d
                if d > 4:
                    count_gt4 += 1
        adj[u].remove(v)
        adj[v].remove(u)
        return (max_d, count_gt4)

    dists = get_distances()
    max_d = max(max(row) for row in dists)
    
    while max_d > 4:
        if time.time() - start_time > 50:
            break
            
        best_score = (float('inf'), float('inf'))
        best_edge = None
        
        for u in range(n):
            for v in range(u + 1, n):
                if not is_valid_candidate(u, v):
                    continue
                score = evaluate_candidate(u, v)
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
        
        if best_edge is None:
            break
            
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([u, v])
        
        dists = get_distances()
        max_d = max(max(row) for row in dists)

    return [[int(u), int(v)] for u, v in added_edges]
# EVOLVE-BLOCK-END
