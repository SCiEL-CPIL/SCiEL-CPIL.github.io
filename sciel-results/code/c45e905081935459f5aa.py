"""Longest simple path in a pinned G(100, 0.018) graph.

Strategy:
  1. Greedy heuristics (Worst-First, Best-First, degree-based) to get a strong initial path.
  2. Iterative local improvement via 2-opt, 3-opt, and move operations.
  3. Random restarts with time management.
"""
import time
import random
from itertools import permutations

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Precompute adjacency as sets and a fast adjacency matrix
N = P['n']
adj_sets = [set() for _ in range(N)]
adj_matrix = [[False] * N for _ in range(N)]
for u, v in P['edges']:
    adj_sets[u].add(v)
    adj_sets[v].add(u)
    adj_matrix[u][v] = True
    adj_matrix[v][u] = True

def is_edge(u, v):
    return adj_matrix[u][v]

def greedy_worst_first(start):
    """Greedy: always pick the neighbor with the fewest unvisited options (WFS)."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = [v for v in adj_sets[cur] if v not in seen]
        if not candidates:
            break
        # Pick the one with minimum remaining degree
        v = min(candidates, key=lambda w: len(adj_sets[w] - seen))
        path.append(v)
        seen.add(v)
    return path

def greedy_best_first(start):
    """Greedy: always pick the neighbor with the most unvisited options."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = [v for v in adj_sets[cur] if v not in seen]
        if not candidates:
            break
        v = max(candidates, key=lambda w: len(adj_sets[w] - seen))
        path.append(v)
        seen.add(v)
    return path

def try_extend(path, direction=1):
    """Try to extend the path at one end. Returns new path if extended, else original."""
    if direction == 1:
        end = path[-1]
    else:
        end = path[0]
    seen = set(path)
    for v in adj_sets[end]:
        if v not in seen:
            if direction == 1:
                return path + [v]
            else:
                return [v] + path
    return path

def local_improve(path, max_iters=50000):
    """Improve path using 2-opt, 3-opt, and move operations."""
    path = list(path)
    n = len(path)
    if n < 3:
        return path
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # 2-opt: reverse a segment to try to extend
        # Try reversing path[i..j] and check if it connects better
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Reverse segment [i..j]
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                # Check validity: need edge between path[i-1] and new_path[i]
                # and between new_path[j] and new_path[j+1] if they exist
                valid = True
                if i > 0 and not is_edge(path[i-1], new_path[i]):
                    valid = False
                if j < n - 1 and not is_edge(new_path[j], new_path[j+1]):
                    valid = False
                if valid and len(set(new_path)) == n:
                    # Check if this actually helps - try to extend
                    old_len = n
                    new_path = try_extend(new_path, 1)
                    new_path = try_extend(new_path, -1)
                    if len(new_path) > old_len:
                        path = new_path
                        n = len(path)
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # 3-opt: try moving a vertex to a different position
        for i in range(1, n - 1):
            v = path[i]
            # Try inserting v at position j
            for j in range(0, n):
                if j == i or j == i + 1:
                    continue
                new_path = path[:i] + path[i+1:]
                if j > i:
                    j -= 1
                new_path = new_path[:j] + [v] + new_path[j:]
                # Validate
                valid = True
                for k in range(len(new_path) - 1):
                    if not is_edge(new_path[k], new_path[k+1]):
                        valid = False
                        break
                if valid:
                    new_path = try_extend(new_path, 1)
                    new_path = try_extend(new_path, -1)
                    if len(new_path) > n:
                        path = new_path
                        n = len(path)
                        improved = True
                        break
            if improved:
                break
    
    return path

def random_greedy(start, rng):
    """Greedy with random tie-breaking."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = [v for v in adj_sets[cur] if v not in seen]
        if not candidates:
            break
        # Sort by remaining degree, then random tie-break
        candidates.sort(key=lambda w: (len(adj_sets[w] - seen), rng.random()))
        v = candidates[0]
        path.append(v)
        seen.add(v)
    return path

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    best_path = []
    
    # Phase 1: Greedy from all starts
    for start in range(N):
        if time.time() > deadline:
            break
        p1 = greedy_worst_first(start)
        p2 = greedy_best_first(start)
        for p in [p1, p2]:
            if len(p) > len(best_path):
                best_path = p
    
    # Phase 2: Local improvement on best
    if len(best_path) > 0:
        best_path = local_improve(best_path)
    
    # Phase 3: Random restarts with improvement
    rng = random.Random(42)
    while time.time() < deadline:
        start = rng.randint(0, N - 1)
        p = random_greedy(start, rng)
        p = local_improve(p, max_iters=10000)
        if len(p) > len(best_path):
            best_path = p
    
    # Final: try to extend best path
    best_path = try_extend(best_path, 1)
    best_path = try_extend(best_path, -1)
    best_path = local_improve(best_path)
    best_path = try_extend(best_path, 1)
    best_path = try_extend(best_path, -1)
    
    if not best_path:
        best_path = [0]
    
    return best_path
