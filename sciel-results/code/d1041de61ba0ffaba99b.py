"""Biclique partition via simulated annealing with edge-reassignment moves."""
import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _validate(bicliques, adj, all_edges_set):
    n = P['n']
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    return covered == all_edges_set

def _greedy_seed(adj, all_edges_set, n):
    """Fast greedy to get initial partition."""
    remaining = set(all_edges_set)
    bicliques = []
    while remaining:
        e = next(iter(remaining))
        u, v = e
        left = {u}
        right = {v}
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in left or w in right:
                    continue
                if right.issubset(adj[w]):
                    has_rem = any((min(w, x), max(w, x)) in remaining for x in right)
                    if has_rem:
                        left.add(w)
                        changed = True
            for w in range(n):
                if w in left or w in right:
                    continue
                if left.issubset(adj[w]):
                    has_rem = any((min(w, x), max(w, x)) in remaining for x in left)
                    if has_rem:
                        right.add(w)
                        changed = True
        right -= left
        left -= right
        if not left:
            left = {u}
        if not right:
            right = {v}
        for a in left:
            for b in right:
                remaining.discard((min(a, b), max(a, b)))
        bicliques.append((set(left), set(right)))
    return bicliques

def _can_add_vertex_to_side(vertex, side, other_side, adj, remaining_edges):
    """Check if vertex can be added to side while maintaining biclique validity."""
    for v in other_side:
        if v not in adj[vertex]:
            return False
    return True

def _try_add_vertex(vertex, side, other_side, adj, remaining_edges):
    """Try adding vertex to side, return True if successful."""
    if not _can_add_vertex_to_side(vertex, side, other_side, adj, remaining_edges):
        return False
    # Check that all edges from vertex to other_side are in remaining
    for v in other_side:
        e = (min(vertex, v), max(vertex, v))
        if e not in remaining_edges:
            return False
    return True

def _simulated_annealing(bicliques, adj, all_edges_set, n, deadline, seed):
    """SA on biclique partition with edge-reassignment and vertex moves."""
    random.seed(seed)
    current = [list(bc) for bc in bicliques]
    current_cost = len(current)
    best = [list(bc) for bc in current]
    best_cost = current_cost
    
    start_time = time.time()
    T = 1.0
    cooling = 0.995
    
    # Precompute edge-to-biclique mapping
    def build_edge_map(bcl):
        emap = {}
        for i, (l, r) in enumerate(bcl):
            for u in l:
                for v in r:
                    e = (min(u, v), max(u, v))
                    emap[e] = i
        return emap
    
    def get_biclique_edges(bcl, i):
        l, r = bcl[i]
        edges = set()
        for u in l:
            for v in r:
                edges.add((min(u, v), max(u, v)))
        return edges
    
    def is_valid_partition(bcl):
        covered = set()
        for l, r in bcl:
            if not l or not r:
                return False
            if set(l) & set(r):
                return False
            for u in l:
                for v in r:
                    if v not in adj[u]:
                        return False
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == all_edges_set
    
    def try_move_vertex(bcl, biclique_idx, vertex, from_side, to_side):
        """Move vertex from one side to another within a biclique, or between bicliques."""
        l, r = bcl[biclique_idx]
        l_set, r_set = set(l), set(r)
        
        if from_side == 'L':
            if vertex not in l_set:
                return False
            new_l = l_set - {vertex}
            new_r = r_set
            # Check vertex can go to R
            if not _can_add_vertex_to_side(vertex, new_l, new_r, adj, None):
                return False
            # Check all edges from vertex to new_l are covered
            for u in new_l:
                e = (min(vertex, u), max(vertex, u))
                if e not in all_edges_set:
                    return False
            # Check no overlap
            if vertex in new_r:
                return False
            bcl[biclique_idx] = (new_l, new_r | {vertex})
            return True
        else:
            if vertex not in r_set:
                return False
            new_l = l_set
            new_r = r_set - {vertex}
            if not _can_add_vertex_to_side(vertex, new_l, new_r, adj, None):
                return False
            for u in new_l:
                e = (min(vertex, u), max(vertex, u))
                if e not in all_edges_set:
                    return False
            if vertex in new_l:
                return False
            bcl[biclique_idx] = (new_l | {vertex}, new_r)
            return True
    
    def try_move_vertex_between(bcl, src_idx, dst_idx, vertex, src_side, dst_side):
        """Move vertex from src biclique to dst biclique."""
        sl, sr = bcl[src_idx]
        dl, dr = bcl[dst_idx]
        sl_set, sr_set = set(sl), set(sr)
        dl_set, dr_set = set(dl), set(dr)
        
        if src_side == 'L':
            if vertex not in sl_set:
                return False
            new_sl = sl_set - {vertex}
            new_sr = sr_set
            if not new_sl or not new_sr:
                return False
            if set(new_sl) & new_sr:
                return False
        else:
            if vertex not in sr_set:
                return False
            new_sl = sl_set
            new_sr = sr_set - {vertex}
            if not new_sl or not new_sr:
                return False
            if set(new_sl) & new_sr:
                return False
        
        if dst_side == 'L':
            new_dl = dl_set | {vertex}
            new_dr = dr_set
            if not new_dl or not new_dr:
                return False
            if set(new_dl) & new_dr:
                return False
            # Check all cross edges exist
            for u in new_dl:
                for v in new_dr:
                    if v not in adj[u]:
                        return False
        else:
            new_dl = dl_set
            new_dr = dr_set | {vertex}
            if not new_dl or not new_dr:
                return False
            if set(new_dl) & new_dr:
                return False
            for u in new_dl:
                for v in new_dr:
                    if v not in adj[u]:
                        return False
        
        bcl[src_idx] = (new_sl, new_sr)
        bcl[dst_idx] = (new_dl, new_dr)
        return True
    
    def try_merge(bcl, i, j):
        """Try to merge bicliques i and j."""
        li, ri = bcl[i]
        lj, rj = bcl[j]
        li_s, ri_s, lj_s, rj_s = set(li), set(ri), set(lj), set(rj)
        
        candidates = []
        if ri_s & rj_s:
            nl, nr = li_s | lj_s, ri_s & rj_s
            if nl and nr and not (nl & nr):
                candidates.append((nl, nr))
        if li_s & lj_s:
            nl, nr = li_s & lj_s, ri_s | rj_s
            if nl and nr and not (nl & nr):
                candidates.append((nl, nr))
        
        for nl, nr in candidates:
            valid = True
            for u in nl:
                for v in nr:
                    if v not in adj[u]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check edge coverage matches
            merge_edges = set()
            for u in nl:
                for v in nr:
                    merge_edges.add((min(u, v), max(u, v)))
            old_edges = set()
            for u in li_s:
                for v in ri_s:
                    old_edges.add((min(u, v), max(u, v)))
            for u in lj_s:
                for v in rj_s:
                    old_edges.add((min(u, v), max(u, v)))
            if merge_edges != old_edges:
                continue
            
            # Check no overlap with others
            ok = True
            for k in range(len(bcl)):
                if k == i or k == j:
                    continue
                lk, rk = bcl[k]
                for e in merge_edges:
                    found = False
                    for u2 in lk:
                        for v2 in rk:
                            if (min(u2, v2), max(u2, v2)) == e:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            new_bcl = []
            for k in range(len(bcl)):
                if k != i and k != j:
                    new_bcl.append(bcl[k])
            new_bcl.append((nl, nr))
            return new_bcl
        return None
    
    steps = 0
    while time.time() < deadline and T > 0.01:
        steps += 1
        if steps % 5000 == 0 and not is_valid_partition(current):
            # Recover
            current = [list(bc) for bc in best]
            current_cost = best_cost
            continue
        
        move_type = random.random()
        num_bcl = len(current)
        
        if move_type < 0.3 and num_bcl >= 2:
            # Try merge
            i, j = random.sample(range(num_bcl), 2)
            new_bcl = try_merge(current, i, j)
            if new_bcl is not None and is_valid_partition(new_bcl):
                delta_cost = len(new_bcl) - current_cost
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / max(T, 0.001)):
                    current = new_bcl
                    current_cost = len(current)
                    if current_cost < best_cost:
                        best = [list(bc) for bc in current]
                        best_cost = current_cost
        elif move_type < 0.6:
            # Try moving vertex within biclique
            if num_bcl > 0:
                bi = random.randint(0, num_bcl - 1)
                l, r = current[bi]
                if l and r:
                    if random.random() < 0.5:
                        vertex = random.choice(list(l))
                        from_side, to_side = 'L', 'R'
                    else:
                        vertex = random.choice(list(r))
                        from_side, to_side = 'R', 'L'
                    saved = list(current[bi])
                    if try_move_vertex(current, bi, vertex, from_side, to_side):
                        if is_valid_partition(current):
                            if current_cost <= best_cost:
                                pass  # same cost, keep for diversity
                        else:
                            current[bi] = saved
        elif move_type < 0.8 and num_bcl >= 2:
            # Try moving vertex between bicliques
            si, di = random.sample(range(num_bcl), 2)
            sl, sr = current[si]
            if sl and sr:
                vertex = random.choice(list(sl) + list(sr))
                src_side = 'L' if vertex in set(sl) else 'R'
                dst_side = random.choice(['L', 'R'])
                saved_si = list(current[si])
                saved_di = list(current[di])
                if try_move_vertex_between(current, si, di, vertex, src_side, dst_side):
                    if is_valid_partition(current):
                        pass
                    else:
                        current[si] = saved_si
                        current[di] = saved_di
        elif move_type < 0.9 and num_bcl > 0:
            # Try adding a vertex to a biclique side
            bi = random.randint(0, num_bcl - 1)
            l, r = current[bi]
            l_s, r_s = set(l), set(r)
            candidates = [w for w in range(n) if w not in l_s and w not in r_s]
            if candidates:
                vertex = random.choice(candidates)
                to_side = random.choice(['L', 'R'])
                saved = list(current[bi])
                if to_side == 'L':
                    new_l = l_s | {vertex}
                    new_r = r_s
                    if new_l and new_r and not (new_l & new_r):
                        valid = True
                        for u in new_l:
                            for v in new_r:
                                if v not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            current[bi] = (new_l, new_r)
                            if not is_valid_partition(current):
                                current[bi] = saved
                else:
                    new_l = l_s
                    new_r = r_s | {vertex}
                    if new_l and new_r and not (new_l & new_r):
                        valid = True
                        for u in new_l:
                            for v in new_r:
                                if v not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            current[bi] = (new_l, new_r)
                            if not is_valid_partition(current):
                                current[bi] = saved
        
        T *= cooling
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    # Get initial greedy partition
    bicliques = _greedy_seed(adj, all_edges_set, n)
    if not _validate(bicliques, adj, all_edges_set):
        # Fallback: star partition
        bicliques = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                bicliques.append(({u}, set(neighbors)))
    
    best_result = [list(bc) for bc in bicliques]
    best_count = len(best_result)
    
    # Run SA with multiple restarts
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        
        # Create varied starting points
        if trial == 0:
            start = [list(bc) for bc in bicliques]
        else:
            start = [list(bc) for bc in bicliques]
            # Randomly perturb
            random.seed(trial * 1000)
            for _ in range(random.randint(1, 3)):
                if len(start) >= 2:
                    i, j = random.sample(range(len(start)), 2)
                    # Try to merge
                    li, ri = start[i]
                    lj, rj = start[j]
                    li_s, ri_s, lj_s, rj_s = set(li), set(ri), set(lj), set(rj)
                    if ri_s & rj_s:
                        nl, nr = li_s | lj_s, ri_s & rj_s
                        if nl and nr and not (nl & nr):
                            valid = True
                            for u in nl:
                                for v in nr:
                                    if v not in adj[u]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                new_start = []
                                for k in range(len(start)):
                                    if k != i and k != j:
                                        new_start.append(start[k])
                                new_start.append((nl, nr))
                                start = new_start
                                break
        
        sa_result, sa_count = _simulated_annealing(start, adj, all_edges_set, n, min(deadline, start_time + 80 - trial), trial)
        
        if sa_result and _validate(sa_result, adj, all_edges_set):
            if sa_count < best_count:
                best_count = sa_count
                best_result = [list(bc) for bc in sa_result]
    
    # Final local search polish on best
    if best_result and time.time() < deadline - 2:
        current = [list(bc) for bc in best_result]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if time.time() > deadline - 2:
                        break
                    li, ri = current[i]
                    lj, rj = current[j]
                    li_s, ri_s, lj_s, rj_s = set(li), set(ri), set(lj), set(rj)
                    
                    candidates = []
                    if ri_s & rj_s:
                        nl, nr = li_s | lj_s, ri_s & rj_s
                        if nl and nr and not (nl & nr):
                            candidates.append((nl, nr))
                    if li_s & lj_s:
                        nl, nr = li_s & lj_s, ri_s | rj_s
                        if nl and nr and not (nl & nr):
                            candidates.append((nl, nr))
                    
                    for nl, nr in candidates:
                        valid = True
                        for u in nl:
                            for v in nr:
                                if v not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        merge_edges = set()
                        for u in nl:
                            for v in nr:
                                merge_edges.add((min(u, v), max(u, v)))
                        old_edges = set()
                        for u in li_s:
                            for v in ri_s:
                                old_edges.add((min(u, v), max(u, v)))
                        for u in lj_s:
                            for v in rj_s:
                                old_edges.add((min(u, v), max(u, v)))
                        if merge_edges != old_edges:
                            continue
                        
                        ok = True
                        for k in range(len(current)):
                            if k == i or k == j:
                                continue
                            lk, rk = current[k]
                            for e in merge_edges:
                                found = False
                                for u2 in lk:
                                    for v2 in rk:
                                        if (min(u2, v2), max(u2, v2)) == e:
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        
                        new_current = []
                        for k in range(len(current)):
                            if k != i and k != j:
                                new_current.append(current[k])
                        new_current.append((nl, nr))
                        
                        if _validate(new_current, adj, all_edges_set) and len(new_current) < len(current):
                            current = new_current
                            improved = True
                            break
                    if improved:
                        break
        
        if _validate(current, adj, all_edges_set) and len(current) < best_count:
            best_count = len(current)
            best_result = [list(bc) for bc in current]
    
    # Convert to proper types
    result = []
    for left, right in best_result:
        result.append([list(int(x) for x in left), list(int(x) for x in right)])
    
    return result
