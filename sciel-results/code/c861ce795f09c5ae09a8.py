"""Finding max edge-disjoint Hamiltonian cycles using bipartite matching on line graph."""
import time
import sys
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_2_factor_matching(vertices, adj, used_edges):
    """Find a 2-factor (collection of cycles covering all vertices) using bipartite matching.
    
    For each vertex, we need to select exactly 2 incident edges such that the result
    forms a 2-regular graph (union of cycles). We model this as a bipartite matching
    problem where left side is vertices (outgoing) and right side is vertices (incoming).
    """
    vset = set(vertices)
    k = len(vertices)
    
    # Build cost matrix for Hungarian algorithm
    # Cost[i][j] = 0 if edge (i,j) is available, large cost otherwise
    INF = 10**9
    cost = [[INF] * k for _ in range(k)]
    
    for u in vertices:
        for v in adj[u]:
            if v in vset and u != v:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    # Map vertices to indices
                    ui = vertices.index(u)
                    vi = vertices.index(v)
                    cost[ui][vi] = 0
    
    # Find minimum cost assignment (each vertex matched to exactly one successor)
    # This gives a permutation where each vertex has out-degree 1
    # We need to do this twice to get in-degree 1 as well, but Hungarian gives us
    # a single cycle cover (permutation)
    
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except:
        return None
    
    # Check if all assignments are valid (cost 0)
    for r, c in zip(row_ind, col_ind):
        if cost[r][c] >= INF:
            return None
    
    # This gives us a permutation: each vertex u is mapped to v
    # The cycles are the cycles in this permutation
    perm = {}
    for r, c in zip(row_ind, col_ind):
        u = vertices[r]
        v = vertices[c]
        perm[u] = v
    
    # Extract cycles from permutation
    visited = set()
    cycles = []
    for start in vertices:
        if start not in visited:
            cycle = [start]
            visited.add(start)
            cur = start
            while True:
                cur = perm[cur]
                if cur == start:
                    break
                cycle.append(cur)
                visited.add(cur)
            cycles.append(cycle)
    
    return cycles

def _extract_hamiltonian_from_2factor(cycles, vertices):
    """If the 2-factor is a single cycle, it's Hamiltonian. Otherwise, try to merge cycles."""
    if len(cycles) == 1:
        return cycles[0]
    
    # For multiple cycles, we can try to merge them by swapping edges
    # This is complex, so for now just return None if not a single cycle
    # But we can try: if we have 2 cycles, we can swap 2 edges to merge them
    # For simplicity, return None for now
    return None

def _find_ham_cycle_dfs(vertices, adj, used_edges, timeout_per_try=None, start=None):
    """Fallback DFS with Warnsdorff's heuristic."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    if start is None:
        start = min(vset)
    
    path = [start]
    visited = set([start])
    deadline_local = time.time() + timeout_per_try if timeout_per_try else None
    call_count = [0]
    
    sys.setrecursionlimit(10000)
    
    avail = set()
    for u in vset:
        for v in adj[u]:
            if v in vset:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    avail.add(e)
    
    def dfs(cur):
        call_count[0] += 1
        if deadline_local and call_count[0] % 100 == 0 and time.time() > deadline_local:
            return None
        if len(path) == k:
            last = path[-1]
            e_close = (min(last, start), max(last, start))
            if e_close in avail:
                return list(path)
            return None
        
        remaining = k - len(path)
        cands = []
        for nxt in adj[cur]:
            if nxt not in visited and nxt in vset:
                e = (min(cur, nxt), max(cur, nxt))
                if e not in avail:
                    continue
                if remaining == 1:
                    e_close = (min(nxt, start), max(nxt, start))
                    if e_close not in avail:
                        continue
                    onward = 0
                else:
                    onward = 0
                    for nn in adj[nxt]:
                        if nn not in visited and nn in vset and nn != cur:
                            if (min(nxt, nn), max(nxt, nn)) in avail:
                                onward += 1
                    if onward == 0:
                        continue
                cands.append((onward, nxt))
        
        if not cands:
            return None
        cands.sort()
        for _, nxt in cands:
            visited.add(nxt)
            path.append(nxt)
            res = dfs(nxt)
            if res is not None:
                return res
            path.pop()
            visited.discard(nxt)
        return None
    
    return dfs(start)

def _find_max_disjoint_cycles(vertices, adj, max_cycles=3, time_budget=5.0):
    """Greedily find max edge-disjoint Hamiltonian cycles using matching + DFS fallback."""
    vlist = list(vertices)
    vset = set(vlist)
    used_edges = set()
    cycles = []
    t0 = time.time()
    
    for _ in range(max_cycles):
        if time.time() - t0 > time_budget:
            break
        
        # Try matching-based approach first
        factor_cycles = _find_2_factor_matching(vlist, adj, used_edges)
        found = None
        if factor_cycles:
            ham = _extract_hamiltonian_from_2factor(factor_cycles, vlist)
            if ham is not None:
                found = ham
        
        # If matching didn't give Hamiltonian, try DFS with multiple starts
        if found is None:
            rem_degs = []
            for v in vlist:
                d = 0
                for nb in adj[v]:
                    if nb in vset and (min(v, nb), max(v, nb)) not in used_edges:
                        d += 1
                rem_degs.append((d, v))
            rem_degs.sort()
            
            for _, st in rem_degs[:5]:
                if time.time() - t0 > time_budget:
                    break
                remaining_time = time_budget - (time.time() - t0)
                if remaining_time < 0.3:
                    break
                per_try = min(1.0, remaining_time / 3.0)
                found = _find_ham_cycle_dfs(vlist, adj, used_edges, timeout_per_try=per_try, start=st)
                if found is not None:
                    break
        
        if found is None:
            break
        
        # Verify and mark edges
        ok = True
        for i in range(len(found)):
            u, v = found[i], found[(i + 1) % len(found)]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                ok = False
                break
            used_edges.add(e)
        if not ok:
            break
        
        cycles.append(found)
    
    return cycles

def construct():
    t0 = time.time()
    DEADLINE = 85
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    best_cycles = []
    best_count = 0
    
    def try_set(vlist, budget):
        nonlocal best_cycles, best_count
        if len(vlist) < 3:
            return
        sset = set(vlist)
        md = min(sum(1 for nb in adj[v] if nb in sset) for v in vlist)
        if md < 2:
            return
        ec = sum(1 for u, v in edges if u in sset and v in sset)
        max_possible = min(md // 2, ec // len(vlist))
        if max_possible < best_count:
            return
        if time.time() - t0 > DEADLINE:
            return
        b = min(budget, max(0.5, DEADLINE - (time.time() - t0)))
        if b < 0.3:
            return
        cycles = _find_max_disjoint_cycles(vlist, adj, max_cycles=min(10, max_possible + 1), time_budget=b)
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
    
    # Try full vertex set
    try_set(list(range(n)), 20.0)
    
    # If we already have 3 cycles (theoretical max), return
    if best_count >= 3:
        return _finalize(best_cycles, best_count, n)
    
    # Try subsets if time remains
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 2:
        import random
        random.seed(42)
        tried = set()
        full = frozenset(range(n))
        tried.add(full)
        
        for size in range(14, 5, -1):
            if time.time() - t0 > DEADLINE - 1:
                break
            for _ in range(20):
                subset = set(random.sample(range(n), size))
                fs = frozenset(subset)
                if fs in tried:
                    continue
                tried.add(fs)
                sset = subset
                md = min(sum(1 for nb in adj[v] if nb in sset) for v in subset)
                if md < 2:
                    continue
                ec = sum(1 for u, v in edges if u in sset and v in sset)
                if ec < size:
                    continue
                try_set(list(subset), min(2.0, max(0.5, DEADLINE - (time.time() - t0))))
    
    return _finalize(best_cycles, best_count, n)

def _finalize(cycles, count, n):
    if not cycles:
        adj = _build_adj(n, P['edges'])
        used = set()
        c = _find_ham_cycle_dfs(list(range(n)), adj, used, timeout_per_try=5.0)
        if c:
            cycles = [c]
        else:
            cycles = [list(range(n))]
    
    out = []
    seen_edges = set()
    vset = None
    for cyc in cycles[:10]:
        cyc = list(cyc)
        if len(cyc) < 3:
            continue
        if len(set(cyc)) != len(cyc):
            continue
        if vset is None:
            vset = set(cyc)
        elif set(cyc) != vset:
            continue
        ok = True
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (min(u, v), max(u, v))
            if e in seen_edges:
                ok = False
                break
            seen_edges.add(e)
        if not ok:
            continue
        out.append(tuple(cyc))
    
    if not out:
        out = [tuple(range(n))]
    
    return out
