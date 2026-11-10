import time
import random
from collections import deque

def _has_path_length_5(adj, u, v, n):
    """Check if there's a simple path of length exactly 5 from u to v."""
    # BFS that tracks depth, allowing v to be checked at depth 5
    # even if it was reached earlier
    
    # Use a set of (node, depth) to track explored states
    # But this could be expensive. Instead, let's use the fact that
    # we only care about depth up to 5.
    
    # reachable[d] = set of nodes reachable from u by a simple path of length d
    # We build this iteratively, tracking visited nodes per path is hard.
    
    # For efficiency, let's use a heuristic: check if the shortest path is 5.
    # This is a sufficient condition for a C6 but not necessary.
    
    dist = [-1] * n
    dist[u] = 0
    queue = deque([u])
    
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= 5:
            continue
        for w in adj[node]:
            if dist[w] == -1:
                dist[w] = d + 1
                if w == v and d + 1 == 5:
                    return True
                queue.append(w)
    
    return dist[v] == 5


def construct():
    n = 40
    deadline = time.time() + 110  # 10s safety margin
    random.seed(42)
    
    best_adj = [[0] * n for _ in range(n)]
    best_edge_set = set()
    
    # Initialize with a random graph with density ~0.4
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.4:
                best_edge_set.add([u, v])
                best_adj[u][v] = 1
                best_adj[v][u] = 1
    
    # Greedy addition with perturbation
    while time.time() < deadline:
        added = False
        for u in range(n):
            for v in range(u + 1, n):
                if [u, v] in best_edge_set:
                    continue
                if time.time() > deadline:
                    break
                if not _has_path_length_5(best_adj, u, v, n):
                    best_edge_set.add([u, v])
                    best_adj[u][v] = 1
                    best_adj[v][u] = 1
                    added = True
            if time.time() > deadline:
                break
        
        if not added:
            if len(best_edge_set) >= 2:
                edges_list = list(best_edge_set)
                num_remove = random.randint(2, 4)
                num_remove = min(num_remove, len(edges_list))
                to_remove = random.sample(edges_list, num_remove)
                for e in to_remove:
                    best_edge_set.remove(e)
                    best_adj[e[0]][e[1]] = 0
                    best_adj[e[1]][e[0]] = 0
            else:
                break
    
    return list(best_edge_set)
