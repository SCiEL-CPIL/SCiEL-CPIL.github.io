import time
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_farthest(start, adj):
    """Return (farthest_node, parent_array) using BFS."""
    n = len(adj)
    dist = [-1] * n
    parent = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    far = start
    for i in range(n):
        if dist[i] > dist[far]:
            far = i
    return far, parent, dist

def _reconstruct(far, parent):
    path = [far]
    while parent[far] != -1:
        far = parent[far]
        path.append(far)
    path.reverse()
    return path

def _tree_diameter_path(adj, nodes):
    """Longest path in a tree (or forest restricted to nodes). Two BFS."""
    if len(nodes) == 1:
        return [nodes[0]]
    start = next(iter(nodes))
    # BFS from start within the component
    far1, parent1, _ = _bfs_farthest(start, adj)
    # BFS from far1
    far2, parent2, _ = _bfs_farthest(far1, adj)
    return _reconstruct(far2, parent2)

def _find_cycle(adj, comp_nodes):
    """Find a cycle in a unicyclic component. Returns list of cycle nodes in order, or None."""
    # Use DFS to find back edge
    comp_set = set(comp_nodes)
    state = {}  # 0=unvisited, 1=in-stack, 2=done
    parent = {}
    cycle = []
    
    def dfs(u):
        state[u] = 1
        parent[u] = -1
        for v in sorted(adj[u]):
            if v not in comp_set:
                continue
            if state.get(v, 0) == 0:
                parent[v] = u
                res = dfs(v)
                if res:
                    return res
            elif state.get(v, 0) == 1:
                # Found cycle: reconstruct from v to u
                cyc = [u]
                cur = u
                while cur != v:
                    cur = parent[cur]
                    cyc.append(cur)
                return cyc
        state[u] = 2
        return None
    
    for node in comp_nodes:
        if state.get(node, 0) == 0:
            res = dfs(node)
            if res:
                return res
    return None

def _longest_path_unicyclic(adj, comp_nodes):
    """Longest simple path in a unicyclic component."""
    cycle = _find_cycle(adj, comp_nodes)
    if not cycle:
        # Fallback: it's a tree
        return _tree_diameter_path(adj, comp_nodes)
    
    cycle_set = set(cycle)
    
    # For each cycle node, compute the longest tree arm extending from it
    # (not using the cycle edges)
    arm_lengths = {}
    for c in cycle:
        # BFS from c, but not going through cycle edges
        dist = {c: 0}
        q = deque([c])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in comp_set and v not in dist:
                    # Only traverse non-cycle edges
                    if (u in cycle_set and v in cycle_set):
                        continue
                    dist[v] = dist[u] + 1
                    q.append(v)
        arm_lengths[c] = max(dist.values()) if dist else 0
    
    # Option 1: path entirely within the cycle (max is cycle length, but we can't use all if we need to extend)
    # Actually a path can traverse at most the whole cycle (cycle length nodes)
    # Option 2: path goes through part of cycle + two arms
    # The path can enter cycle at node i, traverse some arc, exit at node j, with arms at i and j
    # Max path = max over pairs (i,j) of arm[i] + arm[j] + (distance along cycle between i and j)
    # But we can also just use one arm + whole cycle? No, path is simple.
    # A simple path in unicyclic graph: either entirely in a tree arm + part of cycle, 
    # or uses two arms connected by a cycle arc.
    
    m = len(cycle)
    best = 0
    best_path = None
    
    # For each pair of cycle nodes (i, j), the path uses arm[i] + arc(i->j) + arm[j]
    # arc length in nodes between i and j going one way: min(dist, m-dist) but path uses one specific arc
    # Actually the path enters at i, goes along cycle to j (one direction), so arc nodes = the nodes on that arc
    # Total nodes = arm[i] (nodes in arm i excluding i? no including) 
    # Let me define: arm_lengths[c] = number of nodes in the tree arm rooted at c INCLUDING c
    # So the path through i and j: (arm[i] nodes) + (arc nodes from i to j, excluding i and j to avoid double count) + (arm[j] nodes)
    # = arm[i] + arm[j] + (arc_nodes_between - 2) where arc_nodes_between includes i and j
    
    # Simpler: total = arm[i] + arm[j] + (number of cycle nodes on the chosen arc between i and j, excluding endpoints i,j)
    # = arm[i] + arm[j] + (arc_length_edges) where arc_length_edges is number of edges on that arc
    # Because arm[i] includes i, arm[j] includes j, and the arc adds the intermediate cycle nodes.
    # Total nodes = arm[i] + arm[j] + (edges on arc)
    
    for i_idx, i in enumerate(cycle):
        for j_idx, j in enumerate(cycle):
            if i_idx == j_idx:
                continue
            # two directions around cycle
            for direction in [1, -1]:
                # number of edges from i to j going in 'direction'
                steps = (j_idx - i_idx) % m
                if direction == -1:
                    steps = (i_idx - j_idx) % m
                # steps is number of edges on that arc
                total = arm_lengths[i] + arm_lengths[j] + steps
                if total > best:
                    best = total
                    # We'll reconstruct later if needed
                    best_pair = (i, j, direction)
    
    # Also consider path that is just within one arm + part of cycle (one arm only)
    # That's covered when i==j? No we skip i==j. 
    # A path using only one arm and the cycle: enter at i, go around, come back to i? No, simple path can't.
    # Actually a path can start in arm[i], go to i, traverse the entire cycle and end... no that would revisit.
    # The max with one arm: arm[i] + (cycle length - 1) if we go from i around and stop at a neighbor? 
    # This is covered by choosing j as a neighbor of i with steps=1.
    
    # Now reconstruct the best path
    if best_pair is None:
        # Just return the cycle
        return cycle
    
    i, j, direction = best_pair
    i_idx = cycle.index(i)
    j_idx = cycle.index(j)
    
    # Get arm path for i (from i outward to deepest)
    def get_arm_path(root, comp_set, cycle_set):
        # BFS to find deepest node in arm
        dist = {root: 0}
        parent = {root: None}
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in comp_set and v not in dist:
                    if (u in cycle_set and v in cycle_set):
                        continue
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        deepest = max(dist, key=lambda x: dist[x])
        path = [deepest]
        cur = deepest
        while parent[cur] is not None:
            cur = parent[cur]
            path.append(cur)
        path.reverse()  # now starts at root
        return path
    
    arm_i = get_arm_path(i, comp_set, cycle_set)
    arm_j = get_arm_path(j, comp_set, cycle_set)
    # arm_i starts at i, arm_j starts at j
    # Reverse arm_i so it ends at i
    arm_i = list(reversed(arm_i))  # deepest ... i
    # arm_j: we want to go from j outward, so keep as is but we'll append reversed
    # Path: arm_i (ends at i) + arc from i to j + arm_j (starts at j, goes outward)
    # arm_j currently is [j, ..., deepest_j]. We want to append from j outward.
    # So path = arm_i + arc_nodes + arm_j[1:]
    
    # Build arc from i to j in given direction
    arc = [i]
    cur_idx = i_idx
    for _ in range(steps):
        cur_idx = (cur_idx + direction) % m
        arc.append(cycle[cur_idx])
    # arc ends at j
    
    full_path = arm_i + arc[1:] + arm_j[1:]
    # Verify no duplicates
    assert len(full_path) == len(set(full_path)), f"Duplicate in path: {full_path}"
    return full_path

def _longest_path_component(adj, comp_nodes):
    """Exact longest path in a connected component."""
    nodes = list(comp_nodes)
    if len(nodes) == 1:
        return [nodes[0]]
    
    # Count edges within component
    edge_count = 0
    for u in nodes:
        for v in adj[u]:
            if v in comp_nodes and u < v:
                edge_count += 1
    
    if edge_count == len(nodes) - 1:
        # Tree
        return _tree_diameter_path(adj, comp_nodes)
    elif edge_count == len(nodes):
        # Unicyclic
        return _longest_path_unicyclic(adj, comp_nodes)
    else:
        # More complex (shouldn't happen for this graph, but fallback to greedy)
        # Use greedy with multiple restarts
        best = []
        for start in nodes:
            path = [start]
            seen = {start}
            while True:
                candidates = [v for v in adj[path[-1]] if v not in seen]
                if not candidates:
                    break
                # Choose the one with fewest unvisited neighbors (Warnsdorff-like)
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
            if len(path) > len(best):
                best = path
        return best if best else [nodes[0]]

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    adj = _build_adj()
    n = P['n']
    
    # Find connected components
    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            # BFS
            comp = []
            q = deque([i])
            visited.add(i)
            while q:
                u = q.popleft()
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            components.append(comp)
    
    best_path = []
    for comp in components:
        if time.time() > deadline:
            break
        path = _longest_path_component(adj, comp)
        if len(path) > len(best_path):
            best_path = path
    
    # Fallback: if no path found (shouldn't happen), return a single vertex
    if not best_path:
        best_path = [0]
    
    return best_path

# EVOLVE-BLOCK-END
