"""Improved path-finder for the pinned G(100, 1.8/100) graph."""
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Build adjacency
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Precompute adjacency as frozensets for faster operations
adj_fs = [frozenset(s) for s in adj]

def greedy_path(start, seen=None, adj=adj):
    """Greedy path extension with Warnsdorff-like heuristic."""
    if seen is None:
        seen = set()
    path = [start]
    seen.add(start)
    while True:
        last = path[-1]
        candidates = adj[last] - seen
        if not candidates:
            break
        # Choose neighbor with fewest unvisited neighbors (Warnsdorff)
        best_v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        path.append(best_v)
        seen.add(best_v)
    return path

def try_extend_both_ends(path, adj=adj):
    """Try to extend the path from both ends."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Try extending the front
        front = path[0]
        candidates = adj[front] - seen
        if candidates:
            # Pick the one with fewest options
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Try extending the back
        back = path[-1]
        candidates = adj[back] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def local_search_2opt(path, adj=adj, max_iters=50):
    """Try 2-opt style improvements: reverse segments to find longer paths."""
    path = list(path)
    n = len(path)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try reversing subpaths
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Check if reversing path[i..j] creates a valid longer path
                # Actually, 2-opt for Hamiltonian path: try swapping edges
                # Simpler: try inserting a new vertex at each position
                pass
        # Try inserting unvisited vertices
        seen = set(path)
        unvisited = set(range(N)) - seen
        for u in unvisited:
            for pos in range(len(path)):
                # Check if u can be inserted at pos
                if pos == 0:
                    if u in adj[path[0]]:
                        new_path = [u] + path
                        if all(adj[new_path[k]] & adj[new_path[k+1]] or new_path[k+1] in adj[new_path[k]] for k in range(len(new_path)-1)):
                            path = new_path
                            seen.add(u)
                            improved = True
                            break
                elif pos == len(path):
                    if u in adj[path[-1]]:
                        new_path = path + [u]
                        if all(new_path[k+1] in adj[new_path[k]] for k in range(len(new_path)-1)):
                            path = new_path
                            seen.add(u)
                            improved = True
                            break
                else:
                    if u in adj[path[pos-1]] and u in adj[path[pos]]:
                        # Check that removing the edge path[pos-1]-path[pos] and adding path[pos-1]-u-path[pos] is valid
                        new_path = path[:pos] + [u] + path[pos:]
                        if all(new_path[k+1] in adj[new_path[k]] for k in range(len(new_path)-1)):
                            path = new_path
                            seen.add(u)
                            improved = True
                            break
            if improved:
                break
    return path

def dfs_longest_path(start, adj, adj_fs, deadline, best_path_ref, max_depth=None):
    """DFS to find a long path from start, with time limit."""
    if max_depth is None:
        max_depth = N
    best_len = best_path_ref[0]
    
    def dfs(path, seen, depth):
        nonlocal best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path_ref[1] = list(path)
        if depth >= max_depth:
            return
        last = path[-1]
        candidates = adj[last] - seen
        if not candidates:
            return
        # Sort by degree (Warnsdorff)
        sorted_cands = sorted(candidates, key=lambda w: (len(adj[w] - seen), w))
        for v in sorted_cands:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen, depth + 1)
            path.pop()
            seen.discard(v)
    
    dfs([start], {start}, 0)
    return best_path_ref[1]

def branch_and_bound(adj, deadline):
    """Branch and bound search for longest path."""
    best_path = []
    
    # Try from each vertex, but with time limit
    for start in range(N):
        if time.time() > deadline:
            break
        # Quick greedy to get a lower bound
        greedy = greedy_path(start)
        greedy = try_extend_both_ends(greedy)
        if len(greedy) > len(best_path):
            best_path = list(greedy)
    
    # Now try DFS from promising starts
    for start in range(N):
        if time.time() > deadline - 5:
            break
        # Only try if greedy from this start was decent
        greedy = greedy_path(start)
        if len(greedy) < len(best_path) - 2:
            continue
        # DFS with tight time limit
        sub_deadline = min(deadline - 2, time.time() + 2)
        result = dfs_longest_path(start, adj, adj_fs, sub_deadline, [len(best_path), best_path], max_depth=N)
        if len(result) > len(best_path):
            best_path = result
    
    return best_path

def construct():
    t0 = time.time()
    deadline = t0 + 75  # Safety margin for 90s limit
    
    # Phase 1: Multi-start greedy with Warnsdorff heuristic
    best = []
    for start in range(N):
        path = greedy_path(start)
        path = try_extend_both_ends(path)
        if len(path) > len(best):
            best = path
    
    # Phase 2: Local search improvements
    best = local_search_2opt(best, adj, max_iters=20)
    
    # Phase 3: More aggressive multi-start with different heuristics
    for start in range(N):
        if time.time() > deadline - 10:
            break
        # Try different greedy strategies
        for heuristic in ['min_degree', 'max_degree', 'random']:
            if time.time() > deadline - 10:
                break
            path = [start]
            seen = {start}
            while True:
                last = path[-1]
                candidates = adj[last] - seen
                if not candidates:
                    break
                if heuristic == 'min_degree':
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                elif heuristic == 'max_degree':
                    v = max(candidates, key=lambda w: (len(adj[w] - seen), w))
                else:
                    import random
                    rng = random.Random(hash((start, len(path))) % 10000)
                    v = rng.choice(sorted(candidates))
                path.append(v)
                seen.add(v)
            path = try_extend_both_ends(path)
            if len(path) > len(best):
                best = path
    
    # Phase 4: Branch and bound / DFS for longer paths
    if time.time() < deadline - 15:
        bb_result = branch_and_bound(adj, deadline - 5)
        if len(bb_result) > len(best):
            best = bb_result
    
    # Phase 5: Final local search polish
    best = local_search_2opt(best, adj, max_iters=30)
    best = try_extend_both_ends(best)
    
    # Validate
    for i in range(len(best) - 1):
        if best[i+1] not in adj[best[i]]:
            # Should not happen, but truncate to valid prefix
            return best[:i+1]
    
    return best
