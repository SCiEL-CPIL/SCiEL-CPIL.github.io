import time
import sys

def _polarity_graph(q):
    if q < 2:
        return 0, [], []
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    n = len(points)
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        x1, y1, z1 = points[i]
        for j in range(i+1, n):
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    return n, edges, adj

def _verify_c4_free(n, adj):
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if len(au & adj[v]) >= 2:
                return False
    return True

def _max_independent_set_size(adj, n, time_limit=1.0):
    start_time = time.time()
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    remap = [0]*n
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    new_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            new_adj[remap[i]].add(remap[j])
    adj_bits = [0]*n
    for i in range(n):
        bits = 0
        for j in new_adj[i]:
            bits |= (1 << j)
        adj_bits[i] = bits
    comp_bits = [0]*n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_bits[i] = (all_mask ^ adj_bits[i]) & ~(1 << i)
    max_size = 0
    def backtrack(start, current_size, candidates):
        nonlocal max_size
        if time.time() - start_time > time_limit:
            return
        if current_size > max_size:
            max_size = current_size
        if candidates == 0:
            return
        if current_size + candidates.bit_count() <= max_size:
            return
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        backtrack(v+1, current_size + 1, (candidates ^ lsb) & comp_bits[v])
        backtrack(v+1, current_size, candidates ^ lsb)
    backtrack(0, 0, all_mask)
    return max_size

def construct():
    deadline = time.time() + 110
    best_n = 0
    best_edges = []
    
    n, edges, adj = _polarity_graph(5)
    if _verify_c4_free(n, adj):
        alpha = _max_independent_set_size(adj, n, time_limit=5.0)
        if alpha <= 10:
            best_n = n
            best_edges = edges[:]
            
    # Extension
    if best_n > 0 and time.time() < deadline - 10:
        cur_n, cur_edges, cur_adj = _polarity_graph(5)
        
        for new_v in range(cur_n, 43):
            if time.time() > deadline - 5:
                break
            
            # Try to add new_v
            # We try different neighbor sets S
            # S must be independent in the conflict graph H
            # H(u,v) exists if N(u) & N(v) is not empty
            
            # Generate all valid S? Too many.
            # Try S = empty, S = {u}, S = {u,v}
            
            found_valid = False
            # Try S = empty
            trial_adj = [set(s) for s in cur_adj]
            trial_adj.append(set())
            trial_n = cur_n + 1
            if _verify_c4_free(trial_n, trial_adj):
                alpha = _max_independent_set_size(trial_adj, trial_n, time_limit=2.0)
                if alpha <= 10:
                    # Accept
                    cur_n = trial_n
                    cur_adj = trial_adj
                    cur_edges.append([]) # No edges added
                    found_valid = True
                    best_n = cur_n
                    best_edges = cur_edges[:]
                    break # Move to next vertex
            
            # Try S = {u} for u in range(cur_n)
            if not found_valid:
                for u in range(cur_n):
                    if time.time() > deadline - 5:
                        break
                    trial_adj = [set(s) for s in cur_adj]
                    trial_adj.append({u})
                    trial_adj[u].add(trial_n)
                    # Check C4: u and new_v share no common neighbor?
                    # new_v is only connected to u.
                    # C4 would be new_v - u - w - x - new_v? No, new_v has degree 1.
                    # So no C4 involving new_v.
                    # Check C4 in rest? Already C4-free.
                    # So C4-free is guaranteed.
                    alpha = _max_independent_set_size(trial_adj, trial_n, time_limit=2.0)
                    if alpha <= 10:
                        cur_n = trial_n
                        cur_adj = trial_adj
                        cur_edges.append([u, trial_n])
                        found_valid = True
                        best_n = cur_n
                        best_edges = cur_edges[:]
                        break
                        
            if not found_valid:
                # Try S = {u, v} where N(u) & N(v) is empty
                # This is O(n^2) checks
                for u in range(cur_n):
                    if time.time() > deadline - 5:
                        break
                    for v in range(u+1, cur_n):
                        if time.time() > deadline - 5:
                            break
                        if len(cur_adj[u] & cur_adj[v]) == 0:
                            trial_adj = [set(s) for s in cur_adj]
                            trial_adj.append({u, v})
                            trial_adj[u].add(trial_n)
                            trial_adj[v].add(trial_n)
                            # C4 check:
                            # new_v connected to u, v.
                            # C4: new_v - u - w - v - new_v?
                            # Requires w common neighbor of u, v.
                            # We checked N(u) & N(v) is empty. So no such w.
                            # So C4-free.
                            alpha = _max_independent_set_size(trial_adj, trial_n, time_limit=2.0)
                            if alpha <= 10:
                                cur_n = trial_n
                                cur_adj = trial_adj
                                cur_edges.append([u, trial_n])
                                cur_edges.append([v, trial_n])
                                found_valid = True
                                best_n = cur_n
                                best_edges = cur_edges[:]
                                break
                    if found_valid:
                        break
                        
            if not found_valid:
                # Cannot add vertex
                break

    if best_n == 0:
        return {'n': 1, 'edges': []}
    return {'n': best_n, 'edges': best_edges}
