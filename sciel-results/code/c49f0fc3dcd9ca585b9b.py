"""Path cover via odd-vertex pairing + Eulerian trail extraction with iterative refinement."""
import time
import heapq
from collections import deque
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin under 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def odd_vertices(adj, n):
        deg = [len(a) for a in adj]
        return [v for v in range(n) if deg[v] % 2 == 1]

    def all_pairs_dists(adj, n, sources):
        # BFS from each source; return dict source -> (dist, prev)
        res = {}
        for s in sources:
            dist = [-1] * n
            prev = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
            res[s] = (dist, prev)
        return res

    def min_cost_matching(costs, k):
        # Hungarian algorithm for min cost perfect matching, k even, k small
        # costs: k x k symmetric with inf on diagonal
        # Returns list of k//2 pairs (i, j) with i < j
        if k == 0:
            return []
        INF = 10**9
        n = k
        u = [0] * (n + 1)
        v = [0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [INF] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(0, n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        # p[j] = row assigned to column j
        pairs = []
        for j in range(1, n + 1):
            i = p[j]
            if i > 0 and i < j:
                pairs.append((i - 1, j - 1))
        return pairs

    def extract_paths_from_multigraph(n, edge_list):
        """
        Given a list of undirected edges (u,v), find Eulerian trails/circuits.
        Returns a list of paths (each path is a list of vertices).
        The multigraph may have multiple edges between same vertices.
        """
        if not edge_list:
            return []
        
        # Build adjacency with edge indices
        adj = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edge_list):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        
        used = [False] * len(edge_list)
        paths = []
        
        # Find all vertices with odd degree
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        
        odd_verts = [v for v in range(n) if deg[v] % 2 == 1]
        
        if not odd_verts:
            # All even: single Eulerian circuit
            # Find any vertex with edges
            start = None
            for v in range(n):
                if deg[v] > 0:
                    start = v
                    break
            if start is None:
                return []
            # Hierholzer
            stack = [start]
            path = []
            while stack:
                u = stack[-1]
                found = None
                for (v, idx) in adj[u]:
                    if not used[idx]:
                        found = (v, idx)
                        break
                if found is None:
                    path.append(u)
                    stack.pop()
                else:
                    v, idx = found
                    used[idx] = True
                    stack.append(v)
            path.reverse()
            paths.append(path)
        else:
            # Odd degree vertices: need Eulerian trails
            # Pair them up (we'll do this via the matching above)
            # For now, just do Hierholzer starting from each odd vertex
            # Actually, for a connected multigraph with exactly 2 odd vertices,
            # there's a single Eulerian trail
            # For more odd vertices, we need to decompose
            
            # Since we're adding edges to make all degrees even (via matching),
            # this shouldn't happen. But let's handle it.
            # Find connected components and do Eulerian trails in each
            
            # For simplicity, assume the graph is connected (it should be)
            # and do a single Eulerian trail from one odd vertex to another
            if len(odd_verts) == 2:
                start = odd_verts[0]
                stack = [start]
                path = []
                while stack:
                    u = stack[-1]
                    found = None
                    for (v, idx) in adj[u]:
                        if not used[idx]:
                            found = (v, idx)
                            break
                    if found is None:
                        path.append(u)
                        stack.pop()
                    else:
                        v, idx = found
                        used[idx] = True
                        stack.append(v)
                path.reverse()
                paths.append(path)
            else:
                # More than 2 odd vertices: this means our matching didn't work
                # Fall back to individual edges
                for u, v in edge_list:
                    paths.append([u, v])
        
        return paths

    def solve_one(n, p, seed):
        edges = instance(n, p, seed)
        adj = build_adj(edges, n)
        odds = odd_vertices(adj, n)
        k = len(odds)
        
        if k == 0:
            # All even: single Euler circuit
            return extract_paths_from_multigraph(n, edges)
        
        # Compute all-pairs distances from each odd vertex
        dists = all_pairs_dists(adj, n, odds)
        
        # Build cost matrix
        INF = 10**9
        costs = [[INF] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                d = dists[odds[i]][0][odds[j]]
                if d >= 0:
                    costs[i][j] = d
                else:
                    costs[i][j] = INF
        
        # Min cost perfect matching
        pairs = min_cost_matching(costs, k)
        
        # Build extra edges: for each pair, add shortest path
        extra_edges = []
        for (i, j) in pairs:
            u, v = odds[i], odds[j]
            # Reconstruct shortest path from u to v
            dist, prev = dists[u]
            path = []
            cur = v
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            # path is u -> ... -> v
            for a, b in zip(path, path[1:]):
                extra_edges.append((a, b))
        
        # Now we have original edges + extra edges. 
        # The original edges form the graph we need to cover.
        # The extra edges are "virtual" - they allow us to connect odd vertices.
        # We need to extract paths that use only original edges.
        
        # Strategy: Do Hierholzer on the multigraph (original + extra),
        # then split at extra edges to get paths using only original edges.
        
        all_edges = list(edges) + extra_edges
        paths = extract_paths_from_multigraph(n, all_edges)
        
        # Now we need to split each path at extra edges
        # An extra edge is one that's in extra_edges but not in edges
        edge_set = set(tuple(sorted(e)) for e in edges)
        extra_set = set(tuple(sorted(e)) for e in extra_edges)
        
        result_paths = []
        for path in paths:
            # Split path into segments where each segment uses only original edges
            # A segment breaks when we traverse an extra edge
            cur_seg = [path[0]]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = tuple(sorted((u, v)))
                if e in extra_set and e not in edge_set:
                    # This is an extra edge, so split here
                    if len(cur_seg) >= 2:
                        result_paths.append(cur_seg)
                    cur_seg = [v]
                else:
                    cur_seg.append(v)
            if len(cur_seg) >= 2:
                result_paths.append(cur_seg)
        
        return result_paths

    def verify_and_fix(n, edges, paths):
        """Verify paths are valid and fix if needed."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        valid_paths = []
        
        for path in paths:
            # Check if path is simple
            if len(set(path)) != len(path):
                # Not simple, skip or fix
                continue
            # Check if all edges are in graph and not used
            valid = True
            path_edges = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                path_edges.append(e)
            if valid:
                valid_paths.append(path)
                used_edges.update(path_edges)
        
        # Check coverage
        missing = edge_set - used_edges
        if missing:
            # Add missing edges as individual paths
            for e in missing:
                valid_paths.append(list(e))
        
        return valid_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE - 5:
            # Fallback: trivial edge-as-path cover
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_one(n, p, seed)
        
        # Verify and fix
        paths = verify_and_fix(n, edges, paths)
        results.append(paths)
    
    return results


if __name__ == "__main__":
    r = construct()
    for i, lst in enumerate(r):
        print(i, len(lst))
    print("Total:", sum(len(lst) for lst in r))
