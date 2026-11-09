"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque
from scipy.optimize import milp, LinearConstraint, Bounds

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _generate_cycles(adj, n, rng, max_len=12):
    """Generate a set of candidate cycles: all triangles and random longer cycles."""
    cycles = set()
    
    # 1. Find all triangles
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj[u] & adj[v]
                for w in common:
                    tri = tuple(sorted([u, v, w]))
                    cycles.add(tri)
    
    # 2. Find random longer cycles
    num_attempts = 200
    for _ in range(num_attempts):
        start = rng.choice(list(range(n)))
        if len(adj[start]) < 2:
            continue
        
        # Randomized DFS
        visited = {start}
        path = [start]
        stack = [(start, list(adj[start]))]
        found = False
        
        while stack and not found:
            node, neighbors = stack[-1]
            rng.shuffle(neighbors)
            advanced = False
            for nxt in neighbors:
                if nxt == start and len(path) >= 4: # Skip triangles here as they are already found
                    found = True
                    break
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    stack.append((nxt, list(adj[nxt])))
                    advanced = True
                    break
                    if len(path) > max_len:
                        break
            if not advanced and not found:
                stack.pop()
                path.pop()
                if len(path) < 4:
                    break
            
            if len(path) > max_len:
                break
        
        if found:
            # Canonicalize cycle: rotate so min vertex is first, then sort direction
            cyc = path[:]
            min_idx = cyc.index(min(cyc))
            cyc = cyc[min_idx:] + cyc[:min_idx]
            # Check reverse
            rev = [cyc[0]] + cyc[1:][::-1]
            if tuple(cyc) < tuple(rev):
                cycles.add(tuple(cyc))
            else:
                cycles.add(tuple(rev))
    
    return [list(c) for c in cycles]

def _decompose_greedy(edges, n, rng=None):
    """Original greedy method as fallback."""
    if rng is None:
        rng = Random(42)
    
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    max_iterations = 500
    iters = 0
    while remaining and iters < max_iterations:
        best_cycle = None
        best_score = 0
        
        num_attempts = min(50, max(10, len(remaining) // 5))
        
        for attempt in range(num_attempts):
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            
            weights = [len(adj[v]) for v in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            start = candidates[0]
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    start = candidates[i]
                    break
            
            # Simple DFS
            visited = {start}
            path = [start]
            stack = [(start, list(adj[start]))]
            found = False
            
            while stack and not found:
                node, neighbors = stack[-1]
                rng.shuffle(neighbors)
                advanced = False
                for nxt in neighbors:
                    if nxt == start and len(path) >= 3:
                        found = True
                        break
                    if nxt not in visited:
                        visited.add(nxt)
                        path.append(nxt)
                        stack.append((nxt, list(adj[nxt])))
                        advanced = True
                        break
                if not advanced and not found:
                    stack.pop()
                    path.pop()
            
            if found:
                score = len(path)
                if score > best_score:
                    best_score = score
                    best_cycle = list(path)
        
        if best_cycle is None or len(best_cycle) < 3:
            break
        
        valid = True
        cycle_edges = []
        for i in range(len(best_cycle)):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % len(best_cycle)]
            edge = tuple(sorted((u, v)))
            if edge not in remaining:
                valid = False
                break
            cycle_edges.append(edge)
        
        if not valid:
            break
        
        for edge in cycle_edges:
            remaining.discard(edge)
            u, v = edge
            adj[u].discard(v)
            adj[v].discard(u)
        
        pieces.append(best_cycle[:])
        iters += 1
        
    for u, v in remaining:
        pieces.append([u, v])
        
    return pieces

def _decompose_ilp(edges, n, rng=None, deadline=None):
    if rng is None:
        rng = Random(42)
    
    if not edges:
        return []
        
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Generate candidates
    candidates = _generate_cycles(adj, n, rng)
    
    if not candidates:
        return [[u, v] for u, v in edges]
        
    # Map edges to indices
    edge_list = sorted(edges)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    k = len(candidates)
    
    # Objective: Maximize sum(L_c - 1) x_c
    # Minimize sum(-(L_c - 1)) x_c
    c_obj = []
    for cyc in candidates:
        L = len(cyc)
        c_obj.append(-(L - 1))
        
    # Precompute edge sets for candidates for speed
    cand_edge_sets = []
    for cyc in candidates:
        es = set()
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i+1) % len(cyc)]
            es.add(tuple(sorted((u, v))))
        cand_edge_sets.append(es)
        
    # Build matrix A (m x k)
    A = [[0.0] * k for _ in range(m)]
    for i, e in enumerate(edge_list):
        for j in range(k):
            if e in cand_edge_sets[j]:
                A[i][j] = 1.0
                
    # Constraints: A @ x <= 1
    constraints = LinearConstraint(A, 0, 1)
    integrality = [1] * k
    bounds = Bounds(0, 1)
    
    # Solve
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0 and res.x is not None:
            selected = []
            for j in range(k):
                if res.x[j] > 0.5:
                    selected.append(candidates[j])
            # Extract edges used
            used_edges = set()
            for cyc in selected:
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i+1) % len(cyc)]
                    used_edges.add(tuple(sorted((u, v))))
            
            # Remaining edges
            remaining = [e for e in edge_list if e not in used_edges]
            
            # Construct pieces
            pieces = [list(c) for c in selected]
            for u, v in remaining:
                pieces.append([u, v])
            return pieces
        else:
            # Fallback to greedy if ILP fails
            pass
    except Exception:
        pass
        
    # Fallback: Greedy
    return _decompose_greedy(edges, n, rng)

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                s = specs[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        rng = Random(42 + spec[2])
        
        # Use ILP based decomposition
        pieces = _decompose_ilp(edges, n, rng)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
