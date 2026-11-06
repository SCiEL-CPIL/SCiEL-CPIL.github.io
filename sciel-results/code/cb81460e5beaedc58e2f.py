"""Combined: efficient Warnsdorff DFS + matching-based path cover + 2-opt local search
+ beam search + path stitching, with tighter time management."""

import time
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

adj = [set() for _ in range(N)]
for u, v in EDGES:
    adj[u].add(v)
    adj[v].add(u)

DEG = [len(adj[i]) for i in range(N)]

# Precompute adjacency as sorted lists for speed
ADJ_LISTS = [sorted(adj[i]) for i in range(N)]

# Warnsdorff ordering: prefer low-degree neighbors
NEIGHBORS_SORTED = [sorted(adj[i], key=lambda w: (DEG[w], w)) for i in range(N)]


def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in adj[path[i]]:
            return False
    return True


def greedy_grow(start):
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len(adj[w] - seen), DEG[w], w))
        path.append(v)
        seen.add(v)
    return path


def try_extend_both_ends(path):
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        cur_l = path[0]
        cands_l = adj[cur_l] - seen
        cur_r = path[-1]
        cands_r = adj[cur_r] - seen
        if cands_l and cands_r:
            if len(cands_l) <= len(cands_r):
                v = min(cands_l, key=lambda w: (len(adj[w] - seen), DEG[w], w))
                path.insert(0, v)
            else:
                v = min(cands_r, key=lambda w: (len(adj[w] - seen), DEG[w], w))
                path.append(v)
            seen.add(v)
            changed = True
        elif cands_l:
            v = min(cands_l, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        elif cands_r:
            v = min(cands_r, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            path.append(v)
            seen.add(v)
            changed = True
    return path


def local_search_improve(path, deadline):
    """2-opt local search: reverse segments to create new extension opportunities."""
    best = list(path)
    best_len = len(path)
    n = len(best)
    
    # Multiple passes
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = False
        n = len(best)
        for i in range(n - 1):
            if time.time() > deadline:
                return best
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return best
                left_ok = (i == 0) or (best[j] in adj[best[i - 1]])
                right_ok = (j == n - 1) or (best[i] in adj[best[j + 1]])
                if left_ok and right_ok:
                    new_path = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    ext = try_extend_both_ends(new_path)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best = ext
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best


def dfs_longest_path(start, deadline, best_len):
    """Fast DFS with Warnsdorff ordering and simple pruning."""
    best_path = [start]
    best_len_local = max(1, best_len)
    
    path = [start]
    seen = [False] * N
    seen[start] = True
    
    # Use a counter for unvisited
    unvisited = N - 1
    
    def _dfs(cur, depth):
        nonlocal best_path, best_len_local, unvisited
        if time.time() > deadline:
            return
        
        if depth > best_len_local:
            best_len_local = depth
            best_path = list(path)
        
        # Prune: can't beat best
        if depth + unvisited <= best_len_local:
            return
        
        for nxt in NEIGHBORS_SORTED[cur]:
            if time.time() > deadline:
                return
            if not seen[nxt]:
                seen[nxt] = True
                path.append(nxt)
                unvisited -= 1
                _dfs(nxt, depth + 1)
                path.pop()
                seen[nxt] = False
                unvisited += 1
    
    _dfs(start, 1)
    return best_path


def beam_search_path(start, beam_width, max_depth, deadline):
    beam = [([start], {start})]
    best_path = [start]
    
    for depth in range(max_depth):
        if time.time() > deadline:
            break
        if not beam:
            break
        new_beam = []
        for p, seen in beam:
            cur = p[-1]
            candidates = sorted(adj[cur] - seen, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            for v in candidates:
                new_path = p + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
        if not new_beam:
            break
        new_beam.sort(key=lambda x: len(adj[x[0][-1]] - x[1]))
        beam = new_beam[:beam_width]
    
    return best_path


def find_max_path_cover():
    rows = []
    cols = []
    for u, v in EDGES:
        rows.append(u)
        cols.append(v)
        rows.append(v)
        cols.append(u)
    
    data = np.ones(len(rows))
    bipartite_graph = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    try:
        match_left, match_right = maximum_bipartite_matching(bipartite_graph)
    except Exception:
        return []
    
    succ = [int(match_left[i]) for i in range(N)]
    pred = [int(match_right[i]) for i in range(N)]
    
    starts = [v for v in range(N) if pred[v] == -1]
    paths = []
    in_path = set()
    
    for s in starts:
        path = [s]
        curr = s
        while True:
            nxt = succ[curr]
            if nxt == -1 or nxt == curr:
                break
            path.append(nxt)
            curr = nxt
        if len(path) > 1:
            paths.append(path)
            in_path.update(path)
    
    # Handle cycles
    visited_cycle = set()
    for v in range(N):
        if v not in in_path and v not in visited_cycle:
            cycle = []
            curr = v
            while curr not in visited_cycle and curr not in in_path:
                cycle.append(curr)
                visited_cycle.add(curr)
                nxt = succ[curr]
                if nxt == -1 or nxt == curr:
                    break
                curr = nxt
            if len(cycle) > 1:
                paths.append(cycle)
                in_path.update(cycle)
    
    return paths


def stitch_paths(paths):
    if not paths:
        return []
    
    active_paths = [list(p) for p in paths]
    
    def can_connect(p1, p2):
        if p1[-1] in adj[p2[0]]:
            return p1 + p2
        if p2[-1] in adj[p1[0]]:
            return p2 + p1
        if p1[-1] in adj[p2[-1]]:
            return p1 + p2[::-1]
        if p1[0] in adj[p2[-1]]:
            return p1[::-1] + p2
        return None
    
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(active_paths):
            j = i + 1
            while j < len(active_paths):
                combined = can_connect(active_paths[i], active_paths[j])
                if combined is not None:
                    active_paths[i] = combined
                    active_paths.pop(j)
                    changed = True
                    j = i + 1
                    continue
                j += 1
            i += 1
    
    if not active_paths:
        return []
    return max(active_paths, key=len)


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_path = []
    best_len = 0
    all_paths = []
    
    # Phase 0: Quick initialization via matching + greedy
    if time.time() < deadline - 55:
        cover_paths = find_max_path_cover()
        if cover_paths:
            stitched = stitch_paths(cover_paths)
            if stitched:
                extended = try_extend_both_ends(stitched)
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_path = extended
                all_paths.append(extended)
            
            for cp in cover_paths:
                if len(cp) > 1:
                    ext_cp = try_extend_both_ends(cp)
                    if len(ext_cp) > best_len:
                        best_len = len(ext_cp)
                        best_path = list(ext_cp)
                    if len(ext_cp) >= max(1, best_len - 5):
                        all_paths.append(list(ext_cp))
    
    # Phase 1: Greedy from all starts
    for start in range(N):
        if time.time() > deadline - 50:
            break
        path = greedy_grow(start)
        path = try_extend_both_ends(path)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        if len(path) >= max(1, best_len - 5):
            all_paths.append(list(path))
    
    # Phase 2: Beam search from promising starts
    if time.time() < deadline - 30:
        promising_starts = set()
        if best_path:
            promising_starts.add(best_path[0])
            promising_starts.add(best_path[-1])
        # Add low-degree vertices (likely path endpoints)
        for v in range(N):
            if DEG[v] <= 2:
                promising_starts.add(v)
        # Add high-degree vertices
        high_deg = sorted(range(N), key=lambda x: -DEG[x])[:5]
        promising_starts.update(high_deg)
        
        for start in list(promising_starts)[:15]:
            if time.time() > deadline - 30:
                break
            path = beam_search_path(start, beam_width=50, max_depth=N, deadline=deadline - 30)
            path = try_extend_both_ends(path)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            if len(path) >= max(1, best_len - 5):
                all_paths.append(list(path))
    
    # Phase 3: DFS from multiple starts - this is where the real gains come from
    if best_len > 0 and time.time() < deadline - 10:
        dfs_starts = set()
        if best_path:
            dfs_starts.add(best_path[0])
            dfs_starts.add(best_path[-1])
        # Low-degree vertices first (natural endpoints)
        low_deg = sorted(range(N), key=lambda x: DEG[x])[:10]
        dfs_starts.update(low_deg)
        # High-degree vertices
        high_deg = sorted(range(N), key=lambda x: -DEG[x])[:5]
        dfs_starts.update(high_deg)
        
        dfs_starts = list(dfs_starts)
        dfs_time_budget = deadline - 10 - time.time()
        per_start_time = max(1.0, dfs_time_budget / max(1, len(dfs_starts)))
        
        for dfs_start in dfs_starts:
            if time.time() > deadline - 10:
                break
            t_limit = min(time.time() + per_start_time, deadline - 10)
            dfs_path = dfs_longest_path(dfs_start, t_limit, best_len)
            if len(dfs_path) > best_len:
                best_len = len(dfs_path)
                best_path = list(dfs_path)
    
    # Phase 4: Local search (2-opt) on best path
    if best_len > 0 and time.time() < deadline - 5:
        improved = local_search_improve(best_path, deadline - 5)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Phase 5: Try local search on other good paths
    if best_len > 0 and len(all_paths) > 1 and time.time() < deadline - 3:
        for ap in all_paths:
            if time.time() > deadline - 3:
                break
            if ap != best_path and len(ap) >= best_len - 3:
                improved2 = local_search_improve(ap, deadline - 3)
                if len(improved2) > best_len:
                    best_len = len(improved2)
                    best_path = improved2
                    break
    
    # Phase 6: Final extension
    if best_len > 0 and time.time() < deadline:
        improved = try_extend_both_ends(best_path)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Phase 7: One more local search pass if time allows
    if best_len > 0 and time.time() < deadline - 1:
        improved = local_search_improve(best_path, deadline - 1)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
        improved = try_extend_both_ends(best_path)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Ensure valid nonempty path
    if not best_path:
        best_path = [0]
    
    if not path_is_valid(best_path):
        for u, v in EDGES:
            best_path = [u, v]
            break
        else:
            best_path = [0]
    
    return best_path
