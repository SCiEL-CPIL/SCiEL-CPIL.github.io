"""Longest simple path in the pinned G(100, 1.8/100) graph."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']
DEADLINE = 80.0  # safety margin before 90s kill

# Build adjacency
adj = [set() for _ in range(N)]
for u, v in EDGES:
    adj[u].add(v)
    adj[v].add(u)

# Convert to lists for speed
adj_list = [list(s) for s in adj]
adj_set = [set(s) for s in adj_list]


def time_remaining():
    return DEADLINE - time.time()


def greedy_path(start, adj_set, adj_list, rng=None):
    """Greedy path extension from start using a heuristic."""
    path = [start]
    seen = {start}
    while True:
        last = path[-1]
        candidates = [v for v in adj_list[last] if v not in seen]
        if not candidates:
            break
        # Choose neighbor with fewest unvisited neighbors (Warnsdorff-like)
        if rng:
            # Randomized tie-breaking
            scores = []
            for v in candidates:
                unvisited = len(adj_set[v] - seen)
                scores.append((unvisited, rng.random(), v))
            scores.sort()
            nxt = scores[0][2]
        else:
            nxt = min(candidates, key=lambda w: (len(adj_set[w] - seen), w))
        path.append(nxt)
        seen.add(nxt)
    return path


def try_extend_endpoints(path, adj_set, adj_list, rng):
    """Try to extend the path from either endpoint."""
    path = list(path)
    seen = set(path)
    improved = True
    while improved:
        improved = False
        # Try extending from the end
        last = path[-1]
        candidates = [v for v in adj_list[last] if v not in seen]
        if candidates:
            # Pick the one with fewest unvisited neighbors
            nxt = min(candidates, key=lambda w: (len(adj_set[w] - seen), w))
            path.append(nxt)
            seen.add(nxt)
            improved = True
        # Try extending from the front
        first = path[0]
        candidates = [v for v in adj_list[first] if v not in seen]
        if candidates:
            nxt = min(candidates, key=lambda w: (len(adj_set[w] - seen), w))
            path.insert(0, nxt)
            seen.add(nxt)
            improved = True
    return path


def segment_reversal(path, adj_set, adj_list, rng):
    """Try reversing a segment to expose new endpoints and extend."""
    n = len(path)
    if n < 3:
        return path
    best_path = list(path)
    best_len = n
    
    # Try all segments of length 2 to n-1 (reversing creates new endpoints)
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Reversing path[i..j]
            # New endpoints: path[0] stays, path[i-1] (if i>0) connects to path[j]
            # Actually: segment reversal of path[i..j] means:
            # path = path[0:i] + path[j:i-1:-1] + path[j+1:n]
            # The new adjacency at the join points:
            # path[i-1] must connect to path[j] (was path[i-1]-path[i])
            # path[j+1] must connect to path[i] (was path[j]-path[j+1])
            # Original: path[i-1]-path[i] and path[j]-path[j+1] are edges
            # After reversal: path[i-1]-path[j] and path[i]-path[j+1] must be edges
            
            if i > 0:
                if path[j] not in adj_set[path[i-1]]:
                    continue
            if j < n - 1:
                if path[i] not in adj_set[path[j+1]]:
                    continue
            
            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
            # Try to extend
            extended = try_extend_endpoints(new_path, adj_set, adj_list, rng)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = extended
    
    return best_path


def local_search(path, adj_set, adj_list, rng, max_iters=200):
    """Local search: try segment reversals and endpoint extensions."""
    path = list(path)
    best = list(path)
    best_len = len(path)
    
    for it in range(max_iters):
        if time_remaining() < 2.0:
            break
        # Try segment reversal
        candidate = segment_reversal(path, adj_set, adj_list, rng)
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
            path = candidate
        else:
            # No improvement, try random perturbation
            if len(path) >= 4 and rng.random() < 0.5:
                i = rng.randint(0, len(path) - 3)
                j = rng.randint(i + 1, len(path) - 1)
                # Check if reversal is valid
                if i > 0 and path[j] not in adj_set[path[i-1]]:
                    continue
                if j < len(path) - 1 and path[i] not in adj_set[path[j+1]]:
                    continue
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                extended = try_extend_endpoints(new_path, adj_set, adj_list, rng)
                if len(extended) > best_len:
                    best = extended
                    best_len = len(extended)
                    path = extended
                else:
                    path = new_path  # accept non-improving move for diversification
            else:
                break
    
    return best


def dfs_longest_path(adj_set, adj_list, timeout=5.0):
    """DFS with backtracking to find a long path."""
    start_time = time.time()
    best_path = []
    best_len = 0
    visited = [False] * N
    
    def dfs(path, seen):
        nonlocal best_path, best_len
        if time.time() - start_time > timeout:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        last = path[-1]
        # Sort neighbors by degree for better heuristic
        candidates = [v for v in adj_list[last] if v not in seen]
        candidates.sort(key=lambda w: len(adj_list[w]))
        
        for v in candidates:
            if time.time() - start_time > timeout:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.discard(v)
    
    # Try from each vertex
    for start in range(N):
        if time.time() - start_time > timeout:
            break
        path = [start]
        seen = {start}
        dfs(path, seen)
    
    return best_path


def random_walk_path(adj_set, adj_list, rng, steps=1000):
    """Generate a path via random walk, then extend greedily."""
    start = rng.randint(0, N - 1)
    path = [start]
    seen = {start}
    
    for _ in range(steps):
        last = path[-1]
        candidates = [v for v in adj_list[last] if v not in seen]
        if not candidates:
            break
        v = rng.choice(candidates)
        path.append(v)
        seen.add(v)
    
    # Extend greedily
    path = try_extend_endpoints(path, adj_set, adj_list, rng)
    return path


def construct():
    start_time = time.time()
    DEADLINE_LOCAL = start_time + DEADLINE
    
    def time_ok():
        return time.time() < DEADLINE_LOCAL - 2.0  # 2s safety margin
    
    rng = random.Random(42)
    
    # Phase 1: Greedy from all starts
    best_path = []
    best_len = 0
    for s in range(N):
        if not time_ok():
            break
        p = greedy_path(s, adj_set, adj_list, None)
        p = try_extend_endpoints(p, adj_set, adj_list, rng)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Phase 2: Local search on best path
    if time_ok():
        best_path = local_search(best_path, adj_set, adj_list, rng, max_iters=500)
        best_len = len(best_path)
    
    # Phase 3: DFS from multiple starts
    if time_ok():
        dfs_timeout = min(3.0, max(0.5, time_remaining() * 0.15))
        dfs_path = dfs_longest_path(adj_set, adj_list, timeout=dfs_timeout)
        if len(dfs_path) > best_len:
            best_len = len(dfs_path)
            best_path = dfs_path
            # Local search on DFS result
            if time_ok():
                best_path = local_search(best_path, adj_set, adj_list, rng, max_iters=300)
                best_len = len(best_path)
    
    # Phase 4: Multiple random restarts with local search
    if time_ok():
        for _ in range(20):
            if not time_ok():
                break
            p = random_walk_path(adj_set, adj_list, rng, steps=rng.randint(50, 200))
            p = local_search(p, adj_set, adj_list, rng, max_iters=100)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
    
    # Phase 5: More local search on the best
    if time_ok():
        remaining = time_remaining()
        if remaining > 3.0:
            best_path = local_search(best_path, adj_set, adj_list, rng, max_iters=int(remaining * 50))
            best_len = len(best_path)
    
    # Phase 6: Final DFS with more time if available
    if time_ok():
        remaining = time_remaining()
        if remaining > 5.0:
            dfs_path = dfs_longest_path(adj_set, adj_list, timeout=remaining * 0.4)
            if len(dfs_path) > best_len:
                best_len = len(dfs_path)
                best_path = dfs_path
                if time_ok():
                    best_path = local_search(best_path, adj_set, adj_list, rng, max_iters=200)
                    best_len = len(best_path)
    
    # Final: ensure we have a valid path
    if not best_path:
        best_path = [0]
    
    # Validate
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in adj_set[best_path[i]]:
            # Truncate to valid prefix
            best_path = best_path[:i+1]
            break
    
    return best_path
# EVOLVE-BLOCK-END
