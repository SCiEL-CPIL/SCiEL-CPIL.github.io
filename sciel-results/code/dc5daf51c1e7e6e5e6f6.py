import random
import time

_N = 15
_ALL = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START = time.time()
_DEADLINE = _START + 80.0


def _precompute_bad_pairs():
    """For each edge (u,v), return the set of (s,t) pairs that would violate
    if both s~u and t~v (or s~v and t~u) hold, where s!=t, s!=v, t!=u, s!=t.
    Actually we need: for path s-u-v-t, s in N(u)\{v}, t in N(v)\{u}, s!=t.
    The constraint is deg[s] != deg[t].
    So for each edge (u,v), the 'bad' pairs are (s,t) where s in N(u)\{v}, t in N(v)\{u}, s!=t.
    But degrees change as we add/remove edges, so we can't precompute which are bad.
    Instead, precompute the candidate endpoint pairs for each edge."""
    # For each edge (u,v), the set of possible (s,t) endpoint pairs
    # s must be neighbor of u (not v), t must be neighbor of v (not u), s != t
    # This depends on the graph, so we compute on the fly.
    # But we can precompute for each edge (u,v) the list of (s,t) pairs that
    # COULD be endpoints given the full adjacency.
    pass


def _validity_fast(g_set):
    """Check validity. g_set is a set of (u,v) tuples with u<v."""
    deg = [0] * _N
    adj = [set() for _ in range(_N)]
    for (u, v) in g_set:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    
    for (u, v) in g_set:
        # s in adj[u] \ {v}, t in adj[v] \ {u}, s != t
        su = adj[u] - {v}
        tv = adj[v] - {u}
        for s in su:
            ds = deg[s]
            for t in tv:
                if t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True


def _count_violations_fast(g_set):
    """Count total violations."""
    deg = [0] * _N
    adj = [set() for _ in range(_N)]
    for (u, v) in g_set:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    
    total = 0
    for (u, v) in g_set:
        su = adj[u] - {v}
        tv = adj[v] - {u}
        for s in su:
            ds = deg[s]
            for t in tv:
                if t == s:
                    continue
                if ds == deg[t]:
                    total += 1
    return total


def _norm(e):
    a, b = e
    return (a, b) if a < b else (b, a)


def _solve_once(rng, start_dense=True):
    """Solve from either dense or random start."""
    if start_dense:
        g = set(_ALL)
    else:
        # Random subset
        g = set()
        for e in _ALL:
            if rng.random() < 0.5:
                g.add(e)
    
    # Greedy edge removal to reach a valid graph.
    # Use a priority-based approach: repeatedly find the edge whose removal
    # reduces violations the most.
    while True:
        if time.time() > _DEADLINE:
            return [(0, v) for v in range(1, _N)]
        
        # Check validity
        if _validity_fast(g):
            break
        
        # Find edge whose removal helps most
        # First, compute violation count for current graph
        cur_cnt = _count_violations_fast(g)
        
        # Try removing each edge in the graph, pick the one that reduces violations most
        best_e = None
        best_cnt = cur_cnt + 1  # must be less than current to accept
        best_delta = 0
        
        # For efficiency, only consider edges involved in violations
        # But for simplicity, try all edges (105 max, each check is O(n^2))
        # Actually let's be smarter: compute violations per edge
        deg = [0] * _N
        adj = [set() for _ in range(_N)]
        for (u, v) in g:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge, count how many violations it participates in
        edge_viol = {}
        for (u, v) in g:
            su = adj[u] - {v}
            tv = adj[v] - {u}
            cnt = 0
            for s in su:
                ds = deg[s]
                for t in tv:
                    if t == s:
                        continue
                    if ds == deg[t]:
                        cnt += 1
            edge_viol[(u, v)] = cnt
        
        # Remove the edge with highest violation count
        if not edge_viol:
            break
        max_viol = max(edge_viol.values())
        candidates = [e for e, c in edge_viol.items() if c == max_viol]
        rng.shuffle(candidates)
        g.discard(candidates[0])
    
    # Re-add every edge that preserves validity.
    for e in list(_ALL):
        if time.time() > _DEADLINE:
            break
        if e not in g:
            g2 = g | {e}
            if _validity_fast(g2):
                g = g2
    
    return g


def _solve_local_search(rng):
    """Start from a valid sparse graph and try to add edges via local search."""
    # Start with a star
    g = set()
    for v in range(1, _N):
        g.add((0, v))
    
    # Try to add edges
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for e in _ALL:
            if e in g:
                continue
            g2 = g | {e}
            if _validity_fast(g2):
                g = g2
                improved = True
    
    return g


def _solve_hybrid(rng):
    """Hybrid: start dense, remove, then do local search to add back."""
    g = set(_ALL)
    
    # Greedy removal
    while True:
        if time.time() > _DEADLINE:
            break
        if _validity_fast(g):
            break
        
        # Compute per-edge violation counts
        deg = [0] * _N
        adj = [set() for _ in range(_N)]
        for (u, v) in g:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        edge_viol = {}
        for (u, v) in g:
            su = adj[u] - {v}
            tv = adj[v] - {u}
            cnt = 0
            for s in su:
                ds = deg[s]
                for t in tv:
                    if t == s:
                        continue
                    if ds == deg[t]:
                        cnt += 1
            edge_viol[(u, v)] = cnt
        
        if not edge_viol:
            break
        max_viol = max(edge_viol.values())
        candidates = [e for e, c in edge_viol.items() if c == max_viol]
        rng.shuffle(candidates)
        g.discard(candidates[0])
    
    # Now try to add back edges
    for e in list(_ALL):
        if time.time() > _DEADLINE:
            break
        if e not in g:
            g2 = g | {e}
            if _validity_fast(g2):
                g = g2
    
    return g


def construct():
    best = None
    best_sz = -1
    base_seed = 42
    restart = 0
    
    while time.time() < _DEADLINE:
        rng = random.Random(base_seed + 7 * restart)
        
        # Alternate between strategies
        if restart % 3 == 0:
            g = _solve_once(rng, start_dense=True)
        elif restart % 3 == 1:
            g = _solve_hybrid(rng)
        else:
            g = _solve_local_search(rng)
        
        if _validity_fast(set(g)):
            if len(g) > best_sz:
                best_sz = len(g)
                best = list(g)
        
        restart += 1
    
    if best is None:
        best = [(0, v) for v in range(1, _N)]
    
    out = [[int(u), int(v)] for (u, v) in best]
    return out
