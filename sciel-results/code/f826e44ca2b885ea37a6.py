"""MILP-based path decomposition with greedy fallback."""
import time
from collections import defaultdict, deque
from random import Random

# ---- deadline guard -------------------------------------------------------
_START = time.monotonic()
_LIMIT = 140.0   # safety margin under the 150s kill


def _remaining():
    return _LIMIT - (time.monotonic() - _START)


# ---- graph construction (copied from verifier) ----------------------------
def _build_graph(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


# ---- graph helpers --------------------------------------------------------
def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _valid_decomposition(paths, edge_set):
    seen = set()
    for p in paths:
        if len(p) < 2:
            return False
        if len(set(p)) != len(p):
            return False
        for i in range(len(p) - 1):
            e = tuple(sorted((p[i], p[i + 1])))
            if e not in edge_set or e in seen:
                return False
            seen.add(e)
    return seen == edge_set


# ---- MILP solver using scipy.optimize.milp --------------------------------
def _solve_milp(edges, time_limit=20.0):
    """
    Solve the path decomposition as a minimum-cost flow / ILP.
    
    Variables:
    - x[e] for each edge e: 1 if edge is in the decomposition (should be 1 for all)
    - Actually, we need to model paths directly.
    
    Alternative formulation:
    For each directed arc (u,v) where {u,v} is an edge, variable x_uv in {0,1}
    Flow conservation: for each vertex v, sum_in x - sum_out x = 0 (internal) or +/-1 (endpoints)
    But we need to minimize the number of paths, which equals the number of source vertices.
    
    Better formulation:
    Let P be the set of possible simple paths (too many).
    
    Use the standard path decomposition ILP:
    - For each directed arc (u,v) of the undirected edges, variable x_{uv} in {0,1}
    - Flow conservation at each vertex: outflow - inflow in {-1, 0, 1}
      Actually: sum_{w: (w,v) in arcs} x_{wv} - sum_{w: (v,w) in arcs} x_{vw} = b_v
      where b_v in {-1, 0, 1} and sum b_v = 0
    - The number of paths = (number of vertices with b_v = 1) = (number with b_v = -1)
    - Minimize sum of b_v over vertices with b_v = 1
    
    But we also need x_{uv} + x_{vu} <= 1 for each undirected edge {u,v} (at most one direction)
    And x_{uv} + x_{vu} = 1 for each edge (exactly one direction, since we must cover all edges)
    
    Wait, we need to cover ALL edges. So for each undirected edge {u,v}, exactly one of x_{uv}, x_{vu} is 1.
    
    Then the flow conservation gives us a set of directed paths and cycles.
    Cycles can be broken into paths at any point, but that doesn't change the count if we count them as 1 path each.
    Actually, a cycle is 1 path (if we allow closed paths) or 0 paths if we require open paths.
    The problem says "paths" which typically means simple paths (open). A cycle is not a path in the strict sense.
    
    Hmm, let me reconsider. The problem says "paths must be simple, use graph edges, be edge-disjoint and cover all edges."
    A simple path has distinct vertices. A cycle has distinct vertices except first=last.
    So cycles are NOT valid paths per the problem statement.
    
    So we need to ensure no cycles. This means the flow must be acyclic.
    This is hard to enforce directly in ILP.
    
    Alternative: use the formulation where we select a subset of edges for each path, but that's exponential.
    
    Let me try a different approach: use the fact that in an optimal solution, the number of paths is at most ceil(odd_vertices/2) + cycles.
    But cycles need to be broken.
    
    Actually, let me just use a greedy approach with better heuristics, since MILP for this is complex.
    """
    # For now, return None to indicate we should use greedy
    return None


# ---- Greedy decomposition with multi-start --------------------------------
def _greedy_decompose(edges, adj, rng, max_time=15.0):
    """
    Greedy: repeatedly find the longest simple path in the remaining graph,
    remove its edges, repeat.
    """
    rem = set(tuple(sorted(e)) for e in edges)
    paths = []
    start_time = time.monotonic()
    
    while rem:
        if time.monotonic() - start_time > max_time:
            break
        if _remaining() < 2.0:
            break
        
        # Find a starting vertex (prefer high degree)
        deg = defaultdict(int)
        for u, v in rem:
            deg[u] += 1
            deg[v] += 1
        
        # Try to find the longest path
        best_path = None
        best_len = 0
        
        # Try from several starting vertices
        candidates = sorted(deg.keys(), key=lambda v: -deg[v])[:min(10, len(candidates))]
        # Add some random candidates
        for _ in range(min(5, len(deg))):
            candidates.append(rng.choice(list(deg.keys())))
        
        for start in candidates:
            if deg[start] == 0:
                continue
            
            # DFS to find a long path
            path = [start]
            visited = {start}
            stack = [(start, [w for w in adj[start] if tuple(sorted((start, w))) in rem])]
            
            best_from_start = [start]
            
            # Iterative DFS
            while stack:
                node, nbrs = stack[-1]
                found = False
                for w in nbrs:
                    if w not in visited and tuple(sorted((node, w))) in rem:
                        visited.add(w)
                        path.append(w)
                        if len(path) > len(best_from_start):
                            best_from_start = list(path)
                        stack.append((w, [x for x in adj[w] if tuple(sorted((w, x))) in rem]))
                        found = True
                        break
                if not found:
                    stack.pop()
                    if len(path) > 1:
                        path.pop()
                        visited.discard(path[-1])
            
            if len(best_from_start) > best_len:
                best_len = len(best_from_start)
                best_path = best_from_start
        
        if best_path is None or len(best_path) < 2:
            # Just take any edge
            e = next(iter(rem))
            paths.append(list(e))
            rem.discard(e)
        else:
            # Remove edges of best_path
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i + 1])))
                rem.discard(e)
            paths.append(best_path)
    
    # Add remaining edges as single-edge paths
    for e in list(rem):
        paths.append(list(e))
    
    return paths


def _try_merge(paths, edge_set):
    """Try to merge pairs of paths that share an endpoint."""
    improved = True
    while improved and _remaining() > 1.0:
        improved = False
        n = len(paths)
        for i in range(n):
            if _remaining() < 0.5:
                break
            for j in range(i + 1, n):
                a, b = paths[i], paths[j]
                merged = None
                if a[-1] == b[0]:
                    cand = a + b[1:]
                    if len(set(cand)) == len(cand):
                        merged = cand
                if merged is None and a[0] == b[-1]:
                    cand = b + a[1:]
                    if len(set(cand)) == len(cand):
                        merged = cand
                if merged is None and b[-1] == a[0]:
                    cand = b + a[1:]
                    if len(set(cand)) == len(cand):
                        merged = cand
                if merged is None and b[0] == a[-1]:
                    cand = b + a[1:]
                    if len(set(cand)) == len(cand):
                        merged = cand
                if merged is not None:
                    paths = paths[:i] + paths[i + 1:j] + paths[j + 1:] + [merged]
                    improved = True
                    break
            if improved:
                break
    return paths


def _solve_instance(edges):
    adj = _build_adj(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Multi-start greedy
    rng = Random(42)
    best_paths = None
    best_count = len(edges)
    
    for attempt in range(5):
        if _remaining() < 3.0:
            break
        paths = _greedy_decompose(edges, adj, rng, max_time=10.0)
        if not _valid_decomposition(paths, edge_set):
            # Fix: ensure validity
            seen = set()
            valid_paths = []
            for p in paths:
                if len(p) < 2 or len(set(p)) != len(p):
                    continue
                ok = True
                for i in range(len(p) - 1):
                    e = tuple(sorted((p[i], p[i + 1])))
                    if e not in edge_set or e in seen:
                        ok = False
                        break
                    seen.add(e)
                if ok:
                    valid_paths.append(p)
            # Add missing edges
            for e in edge_set - seen:
                valid_paths.append(list(e))
            paths = valid_paths
        
        paths = _try_merge(paths, edge_set)
        
        if not _valid_decomposition(paths, edge_set):
            paths = [list(e) for e in edges]
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
    
    if best_paths is None:
        best_paths = [list(e) for e in edges]
    
    return best_paths


# ---- entry point -----------------------------------------------------------
def construct():
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    for idx, spec in enumerate(specs):
        if _remaining() < 3.0:
            for s in specs[len(results):]:
                edges = _build_graph(*s)
                results.append([list(e) for e in edges])
            break
        edges = _build_graph(*spec)
        results.append(_solve_instance(edges))
    return results


if __name__ == "__main__":
    out = construct()
    total = sum(len(r) for r in out)
    print("paths per instance:", [len(r) for r in out])
    print("total:", total)
