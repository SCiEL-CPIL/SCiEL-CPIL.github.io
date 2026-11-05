import time
import random
import sys

_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _check_c4(adj):
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _alpha_greedy_lb(adj):
    n = len(adj)
    remaining = set(range(n))
    alpha = 0
    while remaining:
        best_v = None
        best_deg = 10**9
        for v in remaining:
            d = len(adj[v] & remaining)
            if d < best_deg:
                best_deg = d
                best_v = v
        alpha += 1
        remaining.discard(best_v)
        remaining -= adj[best_v]
    return alpha

def _max_independent_set(adj, limit=11):
    n = len(adj)
    masks = []
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks.append(m)

    sys.setrecursionlimit(10000)

    def branch(remaining_mask, current_size):
        if current_size >= limit:
            return True
        if remaining_mask == 0:
            return False
        if current_size + bin(remaining_mask).count("1") < limit:
            return False
        rem = remaining_mask
        best_v = -1
        best_deg = -1
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            deg = bin(remaining_mask & masks[v]).count("1")
            if deg > best_deg:
                best_deg = deg
                best_v = v
        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        return False

    full = (1 << n) - 1
    return branch(full, 0)

def _has_indep_11(adj):
    n = len(adj)
    if n < 11:
        return False
    return _max_independent_set(adj, 11)

def _build_polarity_graph_q(q):
    """Polarity graph of PG(2,q)."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv = pow(x, q-2, q)
                    cx, cy, cz = 1, (y*inv) % q, (z*inv) % q
                elif y != 0:
                    inv = pow(y, q-2, q)
                    cx, cy, cz = 0, 1, (z*inv) % q
                else:
                    cx, cy, cz = 0, 0, 1
                key = (cx, cy, cz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    npts = len(points)
    adj = [set() for _ in range(npts)]
    for i in range(npts):
        for j in range(i+1, npts):
            a = points[i]
            b = points[j]
            s = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) % q
            if s == 0:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _local_search_reduce_alpha(adj, target_n, max_iters=2000):
    n = len(adj)
    adj = [set(s) for s in adj]
    best_adj = [set(s) for s in adj]
    best_alpha = _alpha_greedy_lb(adj)
    
    for it in range(max_iters):
        if _deadline_hit():
            break
        remaining = set(range(n))
        indep = []
        while remaining:
            v = min(remaining, key=lambda x: len(adj[x] & remaining))
            indep.append(v)
            remaining.difference_update(adj[v])
            remaining.discard(v)
        
        if len(indep) <= 10:
            break
        
        added = False
        random.shuffle(indep)
        for i in range(len(indep)):
            for j in range(i+1, len(indep)):
                u, v = indep[i], indep[j]
                if v in adj[u]:
                    continue
                # Check C4
                common = adj[u] & adj[v]
                c4 = False
                for a in common:
                    for b in common:
                        if a != b and b in adj[a]:
                            c4 = True
                            break
                    if c4:
                        break
                if c4:
                    continue
                adj[u].add(v); adj[v].add(u)
                added = True
                break
            if added:
                break
        
        if not added:
            for _ in range(10):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or v in adj[u]:
                    continue
                common = adj[u] & adj[v]
                c4 = False
                for a in common:
                    for b in common:
                        if a != b and b in adj[a]:
                            c4 = True
                            break
                    if c4:
                        break
                if c4:
                    continue
                adj[u].add(v); adj[v].add(u)
                added = True
                break
        if not added:
            break
        
        cur_alpha = _alpha_greedy_lb(adj)
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_adj = [set(s) for s in adj]
            
    return best_adj, best_alpha

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 118.0
    
    best_result = None
    best_n = 0
    
    # Try q=7 first (57 vertices), then q=9 (91 vertices)
    for q in [7, 9]:
        if _deadline_hit():
            break
        try:
            full_adj = _build_polarity_graph_q(q)
            n_full = len(full_adj)
            
            # Try to find a subset of size n_target with alpha < 11
            for n_target in range(min(43, n_full), 30, -1):
                if _deadline_hit():
                    break
                
                # First try a fixed contiguous subset
                if n_target <= n_full:
                    indices = list(range(n_target))
                    sub_adj = [set() for _ in range(n_target)]
                    for i in range(n_target):
                        for j in range(i+1, n_target):
                            if j in full_adj[i]:
                                sub_adj[i].add(j)
                                sub_adj[j].add(i)
                    
                    if _alpha_greedy_lb(sub_adj) < 11:
                        if not _has_indep_11(sub_adj):
                            if n_target > best_n:
                                best_n = n_target
                                edges = []
                                for i in range(n_target):
                                    for j in sub_adj[i]:
                                        if i < j:
                                            edges.append([i, j])
                                best_result = {'n': n_target, 'edges': edges}
                                break
                
                # Then try random subsets
                for attempt in range(20):
                    if _deadline_hit():
                        break
                    indices = random.sample(range(n_full), n_target)
                    sub_adj = [set() for _ in range(n_target)]
                    idx_map = {old: new for new, old in enumerate(indices)}
                    
                    for new_i in range(n_target):
                        old_i = indices[new_i]
                        for new_j in range(new_i+1, n_target):
                            old_j = indices[new_j]
                            if old_j in full_adj[old_i]:
                                sub_adj[new_i].add(new_j)
                                sub_adj[new_j].add(new_i)
                    
                    if _alpha_greedy_lb(sub_adj) < 11:
                        if not _has_indep_11(sub_adj):
                            if n_target > best_n:
                                best_n = n_target
                                edges = []
                                for i in range(n_target):
                                    for j in sub_adj[i]:
                                        if i < j:
                                            edges.append([i, j])
                                best_result = {'n': n_target, 'edges': edges}
                                break
                if best_result and best_result['n'] >= n_target:
                    break
        except Exception:
            pass
    
    # Fallback to q=5 full graph
    if best_result is None or best_n < 31:
        adj1 = _build_polarity_graph_q(5)
        if _alpha_greedy_lb(adj1) < 11 and not _has_indep_11(adj1):
            n = len(adj1)
            if n > best_n:
                best_n = n
                edges = []
                for i in range(n):
                    for j in adj1[i]:
                        if i < j:
                            edges.append([i, j])
                best_result = {'n': n, 'edges': edges}
    
    if best_result is None:
        # Fallback to 30 vertices
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    return best_result
