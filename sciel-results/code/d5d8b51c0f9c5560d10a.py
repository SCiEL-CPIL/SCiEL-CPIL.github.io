"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

import time
import random
import sys
from collections import deque

def _build_adj():
    a = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        a[u].add(v)
        a[v].add(u)
    return a

def _find_largest_component(adj):
    visited = set()
    components = []
    for start in range(P['n']):
        if start not in visited:
            comp = []
            stack = [start]
            visited.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in adj[v]:
                    if u not in visited:
                        visited.add(u)
                        stack.append(u)
            components.append(comp)
    return max(components, key=len) if components else []

def _bipartite_matching(adj, comp_set):
    """
    Compute a maximum bipartite matching on the component's edge set.
    We create a bipartite graph where left nodes are (vertex, 'L') and
    right nodes are (vertex, 'R'), with an edge from u_L to v_R if (u,v) in adj.
    The matching size tells us about path structure.
    Returns: (match_L, match_R, match_size) where match_L[u] = v or -1, match_R[v] = u or -1.
    """
    vertices = sorted(comp_set)
    # Hopcroft-Karp for bipartite matching
    # Left side: vertices, Right side: vertices
    # Edge u_L -> v_R exists if u!=v and v in adj[u]
    
    match_L = {v: -1 for v in vertices}
    match_R = {v: -1 for v in vertices}
    dist = {}
    
    from collections import defaultdict
    # Build adjacency for BFS/DFS in matching
    # For each left node u, its neighbors on the right are adj[u] intersect comp_set
    left_adj = {u: [v for v in adj[u] if v in comp_set and v != u] for u in vertices}
    
    def bfs():
        """Find augmenting paths via BFS."""
        queue = deque()
        for u in vertices:
            if match_L[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in left_adj[u]:
                mu = match_R[v]
                if mu == -1:
                    found = True
                elif dist[mu] == -1:
                    dist[mu] = dist[u] + 1
                    queue.append(mu)
        return found
    
    def dfs(u):
        """Try to find augmenting path from u."""
        for v in left_adj[u]:
            mu = match_R[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                match_L[u] = v
                match_R[v] = u
                return True
        dist[u] = -1
        return False
    
    matching_size = 0
    while bfs():
        for u in vertices:
            if match_L[u] == -1:
                if dfs(u):
                    matching_size += 1
    
    return match_L, match_R, matching_size

def _matching_slack(adj, comp_set, match_L, match_R):
    """
    Compute matching slack for each vertex.
    Slack = degree_in_component - 1 if matched, else degree_in_component.
    Vertices with low slack are bottlenecks - they constrain the path.
    Returns dict: vertex -> slack value.
    """
    slack = {}
    for v in comp_set:
        deg = len([u for u in adj[v] if u in comp_set])
        if match_L[v] != -1:
            slack[v] = deg - 1
        else:
            slack[v] = deg
    return slack

def _greedy_path(adj, start, strategy, rng=None, deadline=None):
    """Greedy path extension with different neighbor-selection strategies."""
    path = [start]
    seen = {start}
    while True:
        if deadline and time.time() > deadline:
            break
        cands = adj[path[-1]] - seen
        if not cands:
            break
        if strategy == 'min_deg':
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_deg':
            v = max(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'random':
            v = rng.choice(sorted(cands))
        elif strategy == 'slack_aware':
            def slack_score(w):
                slack = len(adj[w] - seen)
                return (slack, len(adj[w]), w)
            v = min(cands, key=slack_score)
        elif strategy == 'endpoint_aware':
            v = min(cands, key=lambda w: (len(adj[w]), len(adj[w] - seen), w))
        else:
            v = min(cands)
        path.append(v)
        seen.add(v)
    return path

def _extend_from_both_ends(adj, path, deadline):
    """Try to extend a path from both ends using min-degree heuristic."""
    if not path:
        return path
    result = list(path)
    seen = set(result)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cands = adj[result[-1]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            result.append(v)
            seen.add(v)
            improved = True
        cands = adj[result[0]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            result.insert(0, v)
            seen.add(v)
            improved = True
    return result

def _two_end_grow(adj, start1, start2, rng, deadline):
    """Grow path from two ends inward."""
    if start1 == start2:
        return [start1]
    path1 = [start1]
    path2 = [start2]
    seen = {start1, start2}
    while True:
        if time.time() > deadline:
            break
        cands1 = adj[path1[-1]] - seen
        if cands1:
            v = min(cands1, key=lambda w: len(adj[w] - seen))
            path1.append(v)
            seen.add(v)
        else:
            cands2 = adj[path2[-1]] - seen
            if cands2:
                v = min(cands2, key=lambda w: len(adj[w] - seen))
                path2.append(v)
                seen.add(v)
            else:
                break
    result = path1 + path2[::-1]
    seen_r = set()
    unique = []
    for x in result:
        if x not in seen_r:
            unique.append(x)
            seen_r.add(x)
    return unique

def _backtrack_search(adj, start, max_depth, rng, deadline, endpoint_hints=None, slack_map=None):
    """DFS with backtracking, using matching-based endpoint hints and slack to guide choices."""
    best_path = [start]
    n = P['n']
    endpoint_set = set(endpoint_hints) if endpoint_hints else set()
    sys.setrecursionlimit(10000)
    
    def dfs(path, seen, depth):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = path[:]
        if depth >= max_depth:
            return
        cands = adj[path[-1]] - seen
        if not cands:
            return
        def score(w):
            slack = len(adj[w] - seen)
            # Use global matching slack as secondary signal
            g_slack = slack_map.get(w, 0) if slack_map else 0
            is_endpoint = 1 if w in endpoint_set else 0
            progress = depth / max(max_depth, 1)
            endpoint_penalty = is_endpoint * (1.0 - progress) * 2
            # Prioritize vertices with low global slack (bottlenecks) early
            return (slack + 0.5 * g_slack * (1 - progress) + endpoint_penalty, w)
        cands_list = sorted(cands, key=score)
        for v in cands_list:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen, depth + 1)
            path.pop()
            seen.discard(v)
    
    dfs([start], {start}, 0)
    return best_path

def _bfs_path(adj, start, deadline):
    """BFS to find a simple path (not necessarily longest, but fast)."""
    visited = {start}
    parent = {start: None}
    queue = deque([start])
    farthest = start
    while queue:
        if time.time() > deadline:
            break
        v = queue.popleft()
        farthest = v
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                parent[u] = v
                queue.append(u)
    # Reconstruct path from start to farthest
    path = [farthest]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def _random_walk_path(adj, start, steps, deadline):
    """Random walk that backtracks when stuck."""
    path = [start]
    seen = {start}
    for _ in range(steps):
        if time.time() > deadline:
            break
        cands = adj[path[-1]] - seen
        if not cands:
            break
        v = random.choice(list(cands))
        path.append(v)
        seen.add(v)
    return path

def _local_search_improve(adj, path, deadline, rng=None):
    """Try to improve a path by local swaps and insertions."""
    if len(path) < 2:
        return path
    best = list(path)
    best_len = len(best)
    seen = set(best)
    
    # Try inserting vertices between consecutive path vertices
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best) - 1):
            if time.time() > deadline:
                break
            a, b = best[i], best[i+1]
            # Find common neighbors of a and b not in seen
            common = (adj[a] & adj[b]) - seen
            if common:
                # Try inserting each common neighbor
                for w in sorted(common, key=lambda x: len(adj[x])):
                    new_path = best[:i+1] + [w] + best[i+1:]
                    new_seen = set(new_path)
                    # Verify it's a valid path
                    valid = True
                    for j in range(len(new_path) - 1):
                        if new_path[j+1] not in adj[new_path[j]]:
                            valid = False
                            break
                    if valid and len(new_path) > best_len:
                        best = new_path
                        best_len = len(best)
                        seen = new_seen
                        improved = True
                        break
        # Also try extending ends
        cands = adj[best[-1]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            best.append(v)
            seen.add(v)
            best_len = len(best)
            improved = True
        cands = adj[best[0]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            best.insert(0, v)
            seen.add(v)
            best_len = len(best)
            improved = True
    return best

def _deep_backtrack(adj, start, deadline, endpoint_hints=None, slack_map=None):
    """
    Iterative deepening DFS with strong pruning.
    Uses matching slack to guide candidate ordering.
    """
    best_path = [start]
    endpoint_set = set(endpoint_hints) if endpoint_hints else set()
    n = P['n']
    
    # Use iterative deepening with increasing depth limits
    current_depth = 10
    max_possible = n
    
    while current_depth <= max_possible and time.time() < deadline:
        found_deeper = False
        
        def dfs(path, seen, depth, limit):
            nonlocal best_path, found_deeper
            if time.time() > deadline:
                return False
            if len(path) > len(best_path):
                best_path = path[:]
            if depth >= limit:
                return len(path) > len(best_path) - 1
            cands = adj[path[-1]] - seen
            if not cands:
                return False
            # Sort candidates using matching slack
            def score(w):
                slack = len(adj[w] - seen)
                g_slack = slack_map.get(w, 0) if slack_map else 0
                is_endpoint = 1 if w in endpoint_set else 0
                progress = depth / max(limit, 1)
                endpoint_penalty = is_endpoint * (1.0 - progress) * 1.5
                # Low global slack vertices are bottlenecks - visit them earlier
                return (slack + 0.5 * g_slack * (1 - progress) + endpoint_penalty, w)
            cands_list = sorted(cands, key=score)
            for v in cands_list:
                if time.time() > deadline:
                    return False
                path.append(v)
                seen.add(v)
                if dfs(path, seen, depth + 1, limit):
                    found_deeper = True
                    path.pop()
                    seen.discard(v)
                    return True
                path.pop()
                seen.discard(v)
            return False
        
        if dfs([start], {start}, 0, current_depth):
            found_deeper = True
            current_depth += 2
        else:
            current_depth += 1
            if current_depth > len(best_path) + 5:
                break
    
    return best_path

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    adj = _build_adj()
    best = []
    
    # Find the largest component
    largest_comp = _find_largest_component(adj)
    if not largest_comp:
        return [0]
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    # Compute bipartite matching and slack for guidance
    match_L, match_R, match_size = _bipartite_matching(adj, comp_set)
    slack_map = _matching_slack(adj, comp_set, match_L, match_R)
    
    # Identify bottleneck vertices (low slack) as likely path endpoints
    sorted_by_slack = sorted(comp_set, key=lambda v: slack_map.get(v, 0))
    low_slack_vertices = sorted_by_slack[:20]  # Top 20 lowest slack
    
    # Step 1: Quick greedy from every start
    for start in largest_comp:
        path = _greedy_path(adj, start, 'min_deg')
        if len(path) > len(best):
            best = path
    
    # Step 2: BFS paths from every start (fast, gives a baseline)
    for start in largest_comp:
        if time.time() > deadline:
            break
        path = _bfs_path(adj, start, deadline)
        if len(path) > len(best):
            best = path
    
    # Step 3: Randomized greedy with multiple strategies
    rng = random.Random(42)
    strategies = ['min_deg', 'max_deg', 'random', 'slack_aware', 'endpoint_aware']
    for strat in strategies:
        for start in largest_comp:
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
    
    # Step 4: Two-end growth from pairs of low-slack vertices
    degs = [(len(adj[i]), i) for i in largest_comp]
    degs.sort(reverse=True)
    low_deg = [i for _, i in degs[-20:]]
    for i in range(len(low_slack_vertices)):
        for j in range(i + 1, min(i + 15, len(low_slack_vertices))):
            if time.time() > deadline:
                break
            path = _two_end_grow(adj, low_slack_vertices[i], low_slack_vertices[j], rng, deadline)
            if len(path) > len(best):
                best = path
    
    # Step 5: Deep backtracking from low-slack vertices (likely endpoints)
    for start in low_slack_vertices[:15]:
        if time.time() > deadline:
            break
        path = _deep_backtrack(adj, start, deadline, endpoint_hints=low_slack_vertices[:10], slack_map=slack_map)
        if len(path) > len(best):
            best = path
    
    # Step 6: Local search improvement on best path
    if best and time.time() < deadline:
        improved = _local_search_improve(adj, best[:], deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    # Step 7: More randomized searches with slack-guided backtracking
    for _ in range(30):
        if time.time() > deadline:
            break
        start = rng.choice(largest_comp)
        strat = rng.choice(strategies)
        path = _greedy_path(adj, start, strat, rng)
        if len(path) > len(best):
            best = path
        if len(path) >= 3:
            extended = _extend_from_both_ends(adj, path[:], deadline)
            if len(extended) > len(best):
                best = extended
    
    # Step 8: Backtracking from high-degree vertices (internal path candidates)
    for deg, start in degs[:15]:
        if time.time() > deadline:
            break
        path = _deep_backtrack(adj, start, deadline, endpoint_hints=low_slack_vertices[:10], slack_map=slack_map)
        if len(path) > len(best):
            best = path
    
    # Step 9: Final local search polish
    if best and time.time() < deadline:
        improved = _local_search_improve(adj, best[:], deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    # Step 10: Final extension
    if best and time.time() < deadline:
        extended = _extend_from_both_ends(adj, best[:], deadline)
        if len(extended) > len(best):
            best = extended
    
    # Step 11: More deep backtracks with different seeds, guided by slack
    for seed in [123, 456, 789]:
        if time.time() > deadline:
            break
        rng2 = random.Random(seed)
        for _ in range(5):
            if time.time() > deadline:
                break
            start = rng2.choice(largest_comp)
            path = _deep_backtrack(adj, start, deadline, endpoint_hints=low_slack_vertices[:10], slack_map=slack_map)
            if len(path) > len(best):
                best = path
    
    # Final local search
    if best and time.time() < deadline:
        improved = _local_search_improve(adj, best[:], deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    if not best:
        best = [largest_comp[0]]
    
    return best
# EVOLVE-BLOCK-END
