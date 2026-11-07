"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random
from collections import deque

# ---- Tree generation (fixed, must match task) ----
def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

_TREES = instance_trees(20, 611)
_N = 20

def _adjacency(trees):
    """Return list of adjacency lists (set of neighbors) for each tree."""
    out = []
    for edges in trees:
        if not edges:
            out.append([set()])
            continue
        k = len(edges) + 1
        adj = [set() for _ in range(k)]
        for a, b in edges:
            adj[a].add(b); adj[b].add(a)
        out.append(adj)
    return out

_ADJ = _adjacency(_TREES)
_DEG = [len(adj[v]) for i, adj in enumerate(_ADJ) for v in range(len(adj))]

# Tree indices 0..19 correspond to T1..T20
# Order: embed larger/more constrained trees first
_ORDER = sorted(range(_N), key=lambda t: (-(len(_TREES[t]) + 1), -max(_DEG[t * (t + 1) // 2 + _N * (t - 1) // 2 : (t + 1) * (t + 2) // 2] if t > 0 else 0)))

def _tree_degrees(t_idx):
    adj = _ADJ[t_idx]
    return [len(s) for s in adj]

def _try_embed(rng, deadline):
    """
    Backtracking: embed trees in _ORDER using leaf-peeling BFS with forward checking.
    """
    N = _N
    used = [False] * (N * N)
    
    def edge_idx(u, v):
        if u > v: u, v = v, u
        return u * N + v
    
    def edge_used(u, v):
        return used[edge_idx(u, v)]
    
    def mark_edge(u, v):
        used[edge_idx(u, v)] = True
    
    def unmark_edge(u, v):
        used[edge_idx(u, v)] = False
    
    # Free degree per host vertex
    host_free_deg = [N - 1] * N
    
    def embed_tree(t_idx, deadline):
        """
        Embed tree t_idx into K_N avoiding used edges.
        Returns dict tree_vertex -> host_vertex or None.
        Uses BFS leaf-peeling with backtracking.
        """
        adj = _ADJ[t_idx]
        k = len(adj)
        if k == 1:
            return {0: rng.randrange(N)}
        
        deg = [len(s) for s in adj]
        
        for attempt in range(5000):
            if time.time() > deadline:
                return None
            
            # Pick root: prefer high-degree vertex
            if rng.random() < 0.7:
                root = max(range(k), key=lambda v: deg[v])
            else:
                root = rng.randrange(k)
            
            root_host = rng.randrange(N)
            assigned = {root: root_host}
            
            # BFS order from root
            parent = {root: -1}
            order = [root]
            visited = {root}
            q = deque([root])
            while q:
                v = q.popleft()
                for nb in adj[v]:
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = v
                        order.append(nb)
                        q.append(nb)
            
            # Recursive backtracking
            def rec(pos):
                if time.time() > deadline:
                    return None
                if pos == len(order):
                    return dict(assigned)
                
                tv = order[pos]
                par = parent[tv]
                hp = assigned[par]
                
                # Candidate hosts: free neighbors of hp
                cands = []
                for h in range(N):
                    if h == hp:
                        continue
                    if edge_used(hp, h):
                        continue
                    if h in assigned.values():
                        continue
                    # Check that h has enough free degree for tv's unassigned neighbors
                    # tv's degree in tree: deg[tv]
                    # One edge to parent (already accounted)
                    # Remaining: deg[tv] - 1 edges to children
                    need = deg[tv] - 1
                    if host_free_deg[h] < need:
                        continue
                    cands.append(h)
                
                # Shuffle for diversity
                rng.shuffle(cands)
                
                for h in cands:
                    assigned[tv] = h
                    mark_edge(hp, h)
                    host_free_deg[hp] -= 1
                    host_free_deg[h] -= 1
                    
                    res = rec(pos + 1)
                    if res is not None:
                        return res
                    
                    # Undo
                    unmark_edge(hp, h)
                    host_free_deg[hp] += 1
                    host_free_deg[h] += 1
                    del assigned[tv]
                
                return None
            
            r = rec(1)
            if r is not None:
                return r
        
        return None
    
    # Main loop: embed trees in order
    result = [None] * _N
    for t_idx in _ORDER:
        if time.time() > deadline:
            return None
        r = embed_tree(t_idx, deadline)
        if r is None:
            return None
        result[t_idx] = r
    
    # Build final mapping
    out = [None] * _N
    for t_idx in range(_N):
        k = t_idx + 1
        if k == 1:
            out[0] = None
        else:
            r = result[t_idx]
            out[t_idx] = [r[i] for i in range(k)]
    return out

def construct():
    deadline = time.time() + 140  # safety margin for 150s limit
    best = None
    
    # Try randomized restarts with different seeds
    for seed in range(50000):
        if time.time() > deadline - 3:
            break
        rng = Random(seed)
        r = _try_embed(rng, deadline)
        if r is not None:
            # Validate
            ok = True
            used = set()
            for k in range(1, _N + 1):
                entry = r[k - 1]
                if k == 1:
                    if entry is not None:
                        ok = False
                        break
                    continue
                if not isinstance(entry, (list, tuple)):
                    ok = False
                    break
                if len(entry) != k:
                    ok = False
                    break
                if len(set(entry)) != k:
                    ok = False
                    break
                for v in entry:
                    if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= _N:
                        ok = False
                        break
                if not ok:
                    break
                for a, b in _TREES[k - 1]:
                    u, w = entry[a], entry[b]
                    if u == w:
                        ok = False
                        break
                    key = (min(u, w), max(u, w))
                    if key in used:
                        ok = False
                        break
                    used.add(key)
                if not ok:
                    break
            if ok:
                return [list(e) if e is not None else None for e in r]
    
    # Fallback: try to embed just T20
    if best is None:
        best = [None] * 19 + [list(range(20))]
    return best
# EVOLVE-BLOCK-END
