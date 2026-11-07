"""Rewritten program: explicit search over vertex sets for max edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _can_close(path, start, used_edges, adj):
    cur = path[-1]
    if len(path) != len(set(path)):
        return False
    e = (min(cur, start), max(cur, start))
    if start not in adj[cur]:
        return False
    if e in used_edges:
        return False
    return True

def _find_ham_cycle(vertices, adj, used_edges, timeout_per_try=None, start=None):
    """Find a Hamiltonian cycle on vertex set using only unused edges. Fast DFS with heuristic."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    if start is None:
        start = min(vset)
    if start not in vset:
        return None
    # quick degree feasibility
    for v in vertices:
        deg = 0
        for nb in adj[v]:
            if nb in vset and (min(v, nb), max(v, nb)) not in used_edges:
                deg += 1
        if deg < 2:
            return None
    
    path = [start]
    visited = {start}
    deadline_local = time.time() + timeout_per_try if timeout_per_try else None
    
    sys.setrecursionlimit(10000)
    
    def _connected_check(unvisited, last_vertex):
        """Check if unvisited vertices form a connected subgraph with last_vertex as entry."""
        if not unvisited:
            return True
        # BFS from last_vertex through unused edges, see if we reach all unvisited
        to_visit = set()
        for nb in adj[last_vertex]:
            if nb in unvisited and (min(last_vertex, nb), max(last_vertex, nb)) not in used_edges:
                to_visit.add(nb)
        if not to_visit:
            return False
        seen = set(to_visit)
        queue = list(to_visit)
        while queue:
            cur = queue.pop()
            for nb in adj[cur]:
                if nb in unvisited and nb not in seen and (min(cur, nb), max(cur, nb)) not in used_edges:
                    seen.add(nb)
                    queue.append(nb)
        return seen == unvisited
    
    def dfs(cur):
        if deadline_local and time.time() > deadline_local:
            return None
        if len(path) == k:
            if _can_close(path, start, used_edges, adj):
                return list(path)
            return None
        
        # Connectivity pruning: remaining unvisited must be reachable
        unvisited = vset - visited
        if unvisited:
            if not _connected_check(unvisited, cur):
                return None
        
        # candidates: unvisited vertices in vset adjacent to cur via unused edge
        cands = []
        for nxt in adj[cur]:
            if nxt not in visited and nxt in vset:
                e = (min(cur, nxt), max(cur, nxt))
                if e not in used_edges:
                    # Warnsdorff: prefer nxt with fewer onward options
                    onward = 0
                    if len(path) == k - 1:
                        # must be able to close
                        if (min(nxt, start), max(nxt, start)) in used_edges or start not in adj[nxt]:
                            continue
                        onward = 0
                    else:
                        for nn in adj[nxt]:
                            if nn not in visited and nn in vset and nn != cur:
                                if (min(nxt, nn), max(nxt, nn)) not in used_edges:
                                    onward += 1
                    # Tie-break: prefer vertices with lower degree in remaining graph
                    rem_deg = 0
                    for nn in adj[nxt]:
                        if nn in unvisited and (min(nxt, nn), max(nxt, nn)) not in used_edges:
                            rem_deg += 1
                    cands.append((onward, rem_deg, nxt))
        cands.sort()
        for _, _, nxt in cands:
            e = (min(cur, nxt), max(cur, nxt))
            if e in used_edges:
                continue
            visited.add(nxt)
            path.append(nxt)
            res = dfs(nxt)
            if res is not None:
                return res
            path.pop()
            visited.remove(nxt)
        return None
    
    return dfs(start)

def _find_max_disjoint_cycles(vertices, adj, all_edges_set, max_cycles=10, time_budget=5.0):
    """Greedily find max edge-disjoint Hamiltonian cycles on vertices."""
    vlist = list(vertices)
    used_edges = set()
    cycles = []
    t0 = time.time()
    for _ in range(max_cycles):
        if time.time() - t0 > time_budget:
            break
        # try different starts to help find a cycle
        found = None
        # heuristic: start at vertex with lowest remaining degree
        vset_local = set(vlist)
        def rem_deg(v):
            return sum(1 for nb in adj[v] if nb in vset_local and (min(v, nb), max(v, nb)) not in used_edges)
        order = sorted(vlist, key=rem_deg)
        for st in order:
            if time.time() - t0 > time_budget:
                break
            found = _find_ham_cycle(vlist, adj, used_edges, timeout_per_try=2.0, start=st)
            if found is not None:
                break
        if found is None:
            break
        # verify all edges unused
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

def _induced_min_deg(subset_set, adj):
    m = 10**9
    for v in subset_set:
        d = sum(1 for nb in adj[v] if nb in subset_set)
        if d < m:
            m = d
    return m

def _induced_edge_count(subset_set, edges):
    c = 0
    for u, v in edges:
        if u in subset_set and v in subset_set:
            c += 1
    return c

def construct():
    t0 = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add((min(u, v), max(u, v)))
    
    best_cycles = []
    best_count = 0
    best_vertices = None
    
    def try_set(vlist, budget):
        nonlocal best_cycles, best_count, best_vertices
        if len(vlist) < 3:
            return
        sset = set(vlist)
        # feasibility: min degree must be at least 2 for 1 cycle
        md = _induced_min_deg(sset, adj)
        if md < 2:
            return
        ec = _induced_edge_count(sset, edges)
        max_possible = min(md // 2, ec // len(vlist))
        if max_possible < best_count:
            # still try if equal to maybe tie-break, but skip if strictly less
            if max_possible < best_count:
                return
        if time.time() - t0 > DEADLINE:
            return
        b = min(budget, max(0.5, DEADLINE - (time.time() - t0)))
        if b < 0.3:
            return
        cycles = _find_max_disjoint_cycles(vlist, adj, all_edges_set, max_cycles=min(10, max_possible + 1), time_budget=b)
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
            best_vertices = list(vlist)
    
    # Strategy A: full vertex set - this is the primary target since we know max is 3
    try_set(list(range(n)), 15.0)
    
    # If we already found 3 cycles, we're at the theoretical max. Return early.
    if best_count >= 3:
        return _finalize(best_cycles, best_count, n)
    
    # Strategy B: try subsets for potentially better solutions (unlikely but robust)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 2:
        # Rank vertices by degree
        deg = {v: len(adj[v]) for v in range(n)}
        verts_by_deg = sorted(range(n), key=lambda v: -deg[v])
        
        random.seed(123)
        tried = set()
        full = frozenset(range(n))
        tried.add(full)
        
        # 1) Try high-degree core subsets
        for size in range(12, 5, -1):
            if time.time() - t0 > DEADLINE - 1:
                break
            core = verts_by_deg[:size]
            fs = frozenset(core)
            if fs not in tried:
                tried.add(fs)
                try_set(list(core), min(3.0, max(0.5, DEADLINE - (time.time() - t0))))
        
        # 2) Consecutive windows
        for start in range(n):
            if time.time() - t0 > DEADLINE - 1:
                break
            for size in range(6, min(13, n - start + 1)):
                window = list(range(start, start + size))
                fs = frozenset(window)
                if fs not in tried:
                    tried.add(fs)
                    try_set(window, min(2.0, max(0.5, DEADLINE - (time.time() - t0))))
        
        # 3) Random subsets biased to high degree and to edges
        if time.time() - t0 > DEADLINE - 3:
            return _finalize(best_cycles, best_count, n)
        
        budget_each = max(0.4, (DEADLINE - 2 - (time.time() - t0)) / 40.0)
        trials = 0
        while time.time() - t0 < DEADLINE - 2 and trials < 200:
            trials += 1
            size = random.randint(6, 12)
            # sample: with probability pick high-degree vertex
            subset = []
            for _ in range(size):
                if random.random() < 0.6:
                    v = random.choice(verts_by_deg[:size + 3])
                else:
                    v = random.randint(0, n - 1)
                if v not in subset:
                    subset.append(v)
            if len(subset) < size:
                # fill
                for v in range(n):
                    if v not in subset:
                        subset.append(v)
                    if len(subset) == size:
                        break
            fs = frozenset(subset)
            if fs in tried:
                continue
            tried.add(fs)
            # quick feasibility
            sset = set(subset)
            md = _induced_min_deg(sset, adj)
            if md < 2:
                continue
            ec = _induced_edge_count(sset, edges)
            if ec < 2 * size:
                continue
            try_set(subset, budget_each)
    
    return _finalize(best_cycles, best_count, n)

def _finalize(cycles, count, n):
    if not cycles:
        # fallback: single cycle on full set
        edges = P['edges']
        adj = _build_adj(n, edges)
        used = set()
        c = _find_ham_cycle(list(range(n)), adj, used, timeout_per_try=5.0)
        if c:
            cycles = [c]
        else:
            # last resort static
            cycles = [list(range(n))]
    # validate and convert types
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
        # check edges
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

# EVOLVE-BLOCK-END
