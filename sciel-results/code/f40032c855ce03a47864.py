"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # Filter out numbers where 2*a is a square (diagonal constraint: a+a must not be square)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build conflict sets: a and b conflict if a+b is a perfect square (a != b)
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                conflict[i].add(val_to_idx[b])

    # Make symmetric (should already be, but ensure)
    for i in range(num_valid):
        for j in conflict[i]:
            conflict[j].add(i)

    # Partition by parity: even indices in group 0, odd in group 1
    # Cross-parity conflicts (even+odd = odd sum) can be odd squares
    # Same-parity conflicts (even+even or odd+odd = even sum) can be even squares
    # The bipartite (cross-parity) part is solvable exactly via max-flow.
    # Same-parity conflicts will be handled by local search repair.

    group0 = [i for i in range(num_valid) if valid_numbers[i] % 2 == 0]
    group1 = [i for i in range(num_valid) if valid_numbers[i] % 2 == 1]

    # Build bipartite conflict edges (cross-parity only)
    # For max-flow: source -> group0 (capacity 1), group0 -> group1 edges (capacity INF) for conflicts, group1 -> sink (capacity 1)
    # Max independent set in bipartite graph = |V| - min vertex cover = |V| - max matching
    # But we want the actual set, so use the flow-based construction.

    # Actually, let's use the standard reduction:
    # Max independent set in bipartite graph G = (U, V, E):
    # Build flow network: s->u (cap 1), u->v (cap INF) for each edge, v->t (cap 1)
    # Min vertex cover = max flow value
    # The independent set = (U \ (reachable from s in residual)) ∪ (V \ (reachable to t in residual))
    # More precisely: Let R be set of nodes reachable from s in residual graph.
    # Min vertex cover = (U ∩ R) ∪ (V \ R)
    # Max independent set = (U \ R) ∪ (V ∩ R)

    INF = float('inf')

    # Build flow network as adjacency list
    # Nodes: 0 = source, 1..len(group0) = group0 nodes, len(group0)+1..len(group0)+len(group1) = group1 nodes, len(group0)+len(group1)+1 = sink
    num_g0 = len(group0)
    num_g1 = len(group1)
    num_nodes = 1 + num_g0 + num_g1 + 1
    source = 0
    sink = num_nodes - 1

    # Adjacency list for residual graph: adj[u] = list of (v, capacity)
    # We'll use a dictionary-based approach for clarity, but for speed use arrays
    # Edge list: forward edges and reverse edges
    # For each edge (u, v, cap), we store:
    #   edge[u].append((v, cap, reverse_index))
    #   edge[v].append((u, 0, forward_index))

    edges = [[] for _ in range(num_nodes)]

    def add_edge(u, v, cap):
        edges[u].append([v, cap, len(edges[v])])
        edges[v].append([u, 0, len(edges[u]) - 1])

    # Source to group0
    for i, gi in enumerate(group0):
        add_edge(source, 1 + i, 1)

    # Group0 to group1 for conflicts
    for i, gi in enumerate(group0):
        for j in conflict[gi]:
            if valid_numbers[j] % 2 == 1:  # cross-parity
                # Find j's index in group1
                # We need a mapping from vertex index to group1 index
                pass

    # Let me build proper mappings
    g0_idx = {gi: i for i, gi in enumerate(group0)}
    g1_idx = {gi: i for i, gi in enumerate(group1)}

    # Rebuild flow network properly
    edges = [[] for _ in range(num_nodes)]

    def add_edge(u, v, cap):
        edges[u].append([v, cap, len(edges[v])])
        edges[v].append([u, 0, len(edges[u]) - 1])

    # Source to group0
    for i, gi in enumerate(group0):
        add_edge(source, 1 + i, 1)

    # Group0 to group1 for cross-parity conflicts
    for i, gi in enumerate(group0):
        for j in conflict[gi]:
            if j in g1_idx:
                add_edge(1 + i, 1 + num_g0 + g1_idx[j], INF)

    # Group1 to sink
    for i, gi in enumerate(group1):
        add_edge(1 + num_g0 + i, sink, 1)

    # Dinic's algorithm for max flow
    def dinic_max_flow():
        n_nodes = num_nodes
        flow = 0
        # BFS to build level graph
        from collections import deque

        while True:
            level = [-1] * n_nodes
            level[source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for e in edges[u]:
                    v, cap, _ = e
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[sink] < 0:
                break

            # DFS with current arc optimization
            iter_ptr = [0] * n_nodes

            def dfs(u, f):
                if u == sink:
                    return f
                while iter_ptr[u] < len(edges[u]):
                    e = edges[u][iter_ptr[u]]
                    v, cap, rev_idx = e
                    if cap > 0 and level[v] == level[u] + 1:
                        d = dfs(v, min(f, cap))
                        if d > 0:
                            e[1] -= d
                            edges[v][rev_idx][1] += d
                            return d
                    iter_ptr[u] += 1
                return 0

            while True:
                f = dfs(source, float('inf'))
                if f == 0:
                    break
                flow += f

        # Find reachable nodes from source in residual graph
        visited = [False] * n_nodes
        visited[source] = True
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for e in edges[u]:
                v, cap, _ = e
                if cap > 0 and not visited[v]:
                    visited[v] = True
                    queue.append(v)

        return visited

    visited = dinic_max_flow()

    # Max independent set in bipartite graph:
    # U = group0, V = group1
    # Independent set = (U \ R) ∪ (V ∩ R) where R is reachable from source in residual
    bipartite_is = set()
    for i, gi in enumerate(group0):
        if not visited[1 + i]:  # not reachable -> in independent set
            bipartite_is.add(gi)
    for i, gi in enumerate(group1):
        if visited[1 + num_g0 + i]:  # reachable -> in independent set
            bipartite_is.add(gi)

    # Now we have a maximum independent set for the bipartite (cross-parity) subgraph.
    # But there may be same-parity conflicts within this set.
    # Check and repair with local search.

    # Convert to value set
    current_set = set(valid_numbers[i] for i in bipartite_is)

    # Verify and repair same-parity conflicts
    def is_valid(s):
        for a in s:
            if 2 * a in squares:
                return False
            for b in s:
                if b >= a and a + b in squares:
                    return False
        return True

    def count_conflicts(s):
        """Count number of conflicting pairs (including diagonal)."""
        cnt = 0
        vals = list(s)
        for i, a in enumerate(vals):
            if 2 * a in squares:
                cnt += 1
            for b in vals[i+1:]:
                if a + b in squares:
                    cnt += 1
        return cnt

    if not is_valid(current_set):
        # Local search: try to remove conflicting elements or swap
        best_set = set(current_set)
        best_conflicts = count_conflicts(best_set)

        # Try removing elements that are in conflicts
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            # Try removing each element and see if it reduces conflicts
            for a in list(best_set):
                new_set = best_set - {a}
                new_conf = count_conflicts(new_set)
                if new_conf < best_conflicts:
                    best_set = new_set
                    best_conflicts = new_conf
                    improved = True
                    break
            if not improved:
                break

        # Try adding elements that don't create conflicts
        for a in valid_numbers:
            if a in best_set:
                continue
            if 2 * a in squares:
                continue
            # Check if adding a creates any conflict
            can_add = True
            for b in best_set:
                if a + b in squares:
                    can_add = False
                    break
            if can_add:
                best_set.add(a)

        current_set = best_set

    # Final verification
    if not is_valid(current_set) or len(current_set) == 0:
        # Fallback: greedy
        current_set = set()
        for a in valid_numbers:
            can_add = True
            for b in current_set:
                if a + b in squares:
                    can_add = False
                    break
            if can_add:
                current_set.add(a)

    return sorted(current_set)
# EVOLVE-BLOCK-END
