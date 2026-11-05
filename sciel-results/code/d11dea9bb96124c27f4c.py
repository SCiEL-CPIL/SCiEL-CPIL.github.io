"""Long paths above the random-graph phase transition."""
# EVOLVE-BLOCK-START
import time
import sys
import random

# Build the pinned graph
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Precompute degrees
degrees = [len(adj[i]) for i in range(N)]

def _longest_path_dfs(start, deadline):
    """Find longest path starting from 'start' using DFS with backtracking."""
    best_path = [start]
    visited = [False] * N
    visited[start] = True
    path = [start]
    
    def dfs(node):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = path[:]
        # Try neighbors in order of increasing remaining unvisited degree (Warnsdorff-like)
        unvisited_neighbors = [n for n in adj[node] if not visited[n]]
        # Sort by: prefer nodes with fewer unvisited neighbors (to avoid trapping)
        unvisited_neighbors.sort(key=lambda x: (sum(1 for y in adj[x] if not visited[y]), x))
        
        for next_node in unvisited_neighbors:
            visited[next_node] = True
            path.append(next_node)
            dfs(next_node)
            path.pop()
            visited[next_node] = False
            if time.time() > deadline:
                return
    
    dfs(start)
    return best_path

def _greedy_path(start):
    """Quick greedy path for initial lower bound."""
    path = [start]
    seen = {start}
    while True:
        unvisited = adj[path[-1]] - seen
        if not unvisited:
            break
        # Pick neighbor with fewest unvisited options
        v = min(unvisited, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # First, get a quick greedy lower bound from every start
    greedy_best = []
    for s in range(N):
        p = _greedy_path(s)
        if len(p) > len(greedy_best):
            greedy_best = p
    
    best_path = greedy_best[:]
    
    # Now do DFS from each start vertex, starting with promising ones
    # Prioritize starts: higher degree first, then by greedy path length
    starts_sorted = sorted(range(N), key=lambda s: (-len(_greedy_path(s)), -degrees[s]))
    
    for s in starts_sorted:
        if time.time() > deadline:
            break
        p = _longest_path_dfs(s, deadline)
        if len(p) > len(best_path):
            best_path = p[:]
        # Early termination: if we found a Hamiltonian path
        if len(best_path) == N:
            break
    
    # Final polish: try to extend the best path from both ends
    if best_path and time.time() < deadline - 2:
        # Try extending from the end
        end = best_path[-1]
        seen = set(best_path)
        for neighbor in adj[end]:
            if neighbor not in seen:
                candidate = best_path[:] + [neighbor]
                if len(candidate) > len(best_path):
                    best_path = candidate
                    break
        # Try extending from the start
        start_v = best_path[0]
        seen = set(best_path)
        for neighbor in adj[start_v]:
            if neighbor not in seen:
                candidate = [neighbor] + best_path[:]
                if len(candidate) > len(best_path):
                    best_path = candidate
                    break
    
    return best_path

# EVOLVE-BLOCK-END
