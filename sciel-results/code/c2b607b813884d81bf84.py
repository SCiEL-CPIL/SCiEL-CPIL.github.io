# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave 35s margin for safety
    
    q = k
    
    def f_add(a, b): return (a + b) % q
    def f_sub(a, b): return (a - b) % q
    def f_mul(a, b): return (a * b) % q

    def build_luw(d, rules):
        start_key = (0, (0,) * (d + 1))
        label = {start_key: 0}
        queue = [start_key]
        edge_list = []

        for vkey in queue:
            side, coords = vkey
            vidx = label[vkey]
            for first_val in range(q):
                ncoords = [0] * (d + 1)
                ncoords[1] = first_val
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                    else:
                        ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
                nkey = (1 - side, tuple(ncoords))
                if nkey not in label:
                    label[nkey] = len(queue)
                    queue.append(nkey)
                if side == 0:
                    edge_list.append((vidx, label[nkey]))

        return len(queue), edge_list

    def check_girth(n, edges, min_girth):
        from collections import deque
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Non-backtracking BFS from each vertex to verify girth
        # A cycle of length < min_girth exists iff there's a non-backtracking walk
        # that returns to a vertex within min_girth-1 steps
        for start in range(n):
            # State: (current_vertex, previous_vertex, distance)
            visited = set()
            queue = deque()
            # Start from each neighbor of start
            for neighbor in adj[start]:
                state = (neighbor, start, 1)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)
            
            while queue:
                curr, prev, dist = queue.popleft()
                if dist >= min_girth - 1:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # non-backtracking
                    if nxt == start:
                        # Found a cycle of length dist + 1
                        if dist + 1 < min_girth:
                            return False
                        continue
                    new_state = (nxt, curr, dist + 1)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)
        return True

    def is_simple_graph(n, edges, expected_degree):
        """Check that graph is simple and k-regular."""
        from collections import defaultdict
        adj = defaultdict(set)
        for u, v in edges:
            if u == v:
                return False
            if v in adj[u]:
                return False
            adj[u].add(v)
            adj[v].add(u)
        for v in range(n):
            if len(adj[v]) != expected_degree:
                return False
        return True

    # Try d=4 with various rules
    base_rules = {2: (1, 1), 3: (2, 1)}
    
    candidates = []
    for r4 in [(1, 2), (2, 1), (1, 1), (2, 2)]:
        rules = base_rules.copy()
        rules[4] = r4
        candidates.append(rules)

    best_result = None
    
    for rules in candidates:
        if time.time() > deadline:
            break
        n, edges = build_luw(4, rules)
        if not is_simple_graph(n, edges, k):
            continue
        if check_girth(n, edges, g):
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
    
    if best_result is not None:
        return best_result

    # Fallback to d=5
    d = 5
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates_r = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates_r[r]
        
    n, edges = build_luw(d, rules)
    return n, edges
# EVOLVE-BLOCK-END
