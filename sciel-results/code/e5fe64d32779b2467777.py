import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50

    n = 12

    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_cycle_of_length(adj, L):
        """Check if there's a simple cycle of exactly length L.
        For L=4: check pairs with >=2 common neighbors.
        For L=6: check via path enumeration with pruning.
        For L>=8: DFS with depth limit.
        """
        if L == 4:
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        common = adj[u] & adj[v]
                        if len(common) >= 2:
                            return True
            return False

        if L == 6:
            # A 6-cycle: u-v-w-x-y-z-u
            # Enumerate paths of length 5 (6 vertices) and check closure.
            # To reduce work, use the fact that in a 6-cycle, opposite vertices
            # are at distance 3. 
            # Alternative: for each pair (u, w) at distance 3, check if they
            # share a common neighbor path of length 2 on both sides.
            # Simpler: DFS from each start, depth 5, check if last connects to start.
            # With n=12 and pruning, this is fast.
            for start in range(n):
                # DFS: find path start -> ... -> start of length 6
                # Vertices in path: start, v1, v2, v3, v4, v5, start
                # v1, v2, v3, v4, v5 all distinct, none equal to start except at ends
                stack = [(start, [start], {start})]
                while stack:
                    curr, path, visited = stack.pop()
                    if len(path) == L:  # path has L vertices, need to close to start
                        if start in adj[curr]:
                            return True
                        continue
                    for nb in adj[curr]:
                        if nb == start and len(path) < L - 1:
                            continue  # can't close too early
                        if nb == start and len(path) == L - 1:
                            return True
                        if nb not in visited and nb > start:
                            stack.append((nb, path + [nb], visited | {nb}))
            return False

        # General case for L >= 8
        for start in range(n):
            stack = [(start, [start], {start})]
            while stack:
                curr, path, visited = stack.pop()
                if len(path) == L:
                    if start in adj[curr]:
                        return True
                    continue
                for nb in adj[curr]:
                    if nb == start and len(path) < L - 1:
                        continue
                    if nb == start and len(path) == L - 1:
                        return True
                    if nb not in visited and nb > start:
                        stack.append((nb, path + [nb], visited | {nb}))
        return False

    def get_cycle_lengths(adj):
        """Return the set of cycle lengths present in the graph."""
        lengths = set()
        for L in range(4, n + 1):
            if has_cycle_of_length(adj, L):
                lengths.add(L)
        return lengths

    def is_valid(edges):
        """Check all constraints."""
        adj = build_adj(edges)
        for i in range(n):
            if len(adj[i]) < 3:
                return False
        # Triangle-free: check no two neighbors of any vertex are adjacent
        for i in range(n):
            nbrs = list(adj[i])
            for j in range(len(nbrs)):
                for k in range(j + 1, len(nbrs)):
                    if nbrs[k] in adj[nbrs[j]]:
                        return False
        return True

    # Baseline: two disjoint K_{3,3}
    base_edges = []
    for i in range(3):
        for j in range(3):
            base_edges.append([i, j + 3])
            base_edges.append([i + 6, j + 9])

    adj_base = build_adj(base_edges)
    base_lengths = get_cycle_lengths(adj_base)
    best_score = len(base_lengths)
    best_edges = [e[:] for e in base_edges]

    # Local search: start from baseline, try edge swaps to eliminate 6-cycles
    # while keeping degree >= 3 and triangle-free.
    # Strategy: try to add edges that create 4-cycles and remove edges that only
    # contribute to 6-cycles.

    # Actually, a better approach: random search with the fast evaluator.
    # Generate random 3-regular bipartite graphs (bipartite => triangle-free).
    # Check cycle lengths. If we find one with only 4-cycles, great.

    A = list(range(6))
    B = list(range(6, 12))

    def random_3reg_bipartite():
        """Generate a random 3-regular bipartite graph on A x B."""
        # Use configuration model with rejection
        for _ in range(100):
            adj_a = {v: [] for v in A}
            adj_b = {v: [] for v in B}
            valid = True
            for a in A:
                neighbors = random.sample(B, 3)
                adj_a[a] = neighbors
                for b in neighbors:
                    adj_b[b].append(a)
            for b in B:
                if len(adj_b[b]) != 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check for multi-edges (shouldn't happen with sample)
            edges = []
            for a in A:
                for b in adj_a[a]:
                    edges.append([a, b])
            return edges
        return None

    def local_search_from(edges, max_iters=2000):
        """Try to improve cycle length set via edge swaps."""
        adj = build_adj(edges)
        current_lengths = get_cycle_lengths(adj)
        current_score = len(current_lengths)
        
        if current_score <= 1:
            return edges, current_score
        
        best_local = edges[:]
        best_local_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Try a random edge swap: remove one edge, add another
            # that maintains degree >= 3 and triangle-free.
            u, v = random.choice(edges)
            new_u = random.randint(0, n - 1)
            new_v = random.randint(0, n - 1)
            
            if new_u == new_v:
                continue
            if new_u > new_v:
                new_u, new_v = new_v, new_u
            
            # Check if edge already exists
            if new_v in adj[new_u]:
                continue
            
            # Remove (u, v), add (new_u, new_v)
            new_edges = []
            for e in edges:
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    continue
                new_edges.append(e[:])
            new_edges.append([new_u, new_v])
            
            # Quick degree check
            deg = [0] * n
            for a, b in new_edges:
                deg[a] += 1
                deg[b] += 1
            if any(d < 3 for d in deg):
                continue
            
            # Triangle-free check
            new_adj = build_adj(new_edges)
            tri_free = True
            for i in range(n):
                nbrs = list(new_adj[i])
                for j in range(len(nbrs)):
                    for k in range(j + 1, len(nbrs)):
                        if nbrs[k] in new_adj[nbrs[j]]:
                            tri_free = False
                            break
                    if not tri_free:
                        break
                if not tri_free:
                    break
            
            if not tri_free:
                continue
            
            new_lengths = get_cycle_lengths(new_adj)
            new_score = len(new_lengths)
            
            if new_score < best_local_score:
                best_local_score = new_score
                best_local = [e[:] for e in new_edges]
                if best_local_score == 1:
                    return best_local, best_local_score
        
        return best_local, best_local_score

    # Random search
    random.seed(42)
    trials = 0
    while time.time() < deadline - 15:
        trials += 1
        
        # Random 3-regular bipartite
        edges = random_3reg_bipartite()
        if edges is None:
            continue
        
        adj = build_adj(edges)
        
        # Quick check: does it have a 4-cycle?
        has_4 = False
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if len(adj[u] & adj[v]) >= 2:
                        has_4 = True
                        break
            if has_4:
                break
        
        if not has_4:
            continue
        
        lengths = get_cycle_lengths(adj)
        score = len(lengths)
        
        if score < best_score:
            best_score = score
            best_edges = [e[:] for e in edges]
            if best_score == 1:
                break
        
        # Local search from this candidate
        if score <= 2:
            ls_edges, ls_score = local_search_from(edges, max_iters=500)
            if ls_score < best_score:
                best_score = ls_score
                best_edges = ls_edges
                if best_score == 1:
                    break
    
    # Also try local search from the baseline
    if best_score > 1:
        ls_edges, ls_score = local_search_from(base_edges, max_iters=3000)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return sorted(result)
