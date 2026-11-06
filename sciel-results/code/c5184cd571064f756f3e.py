"""Construct a C4-free graph with no independent 11-set, maximizing n."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin
    
    def build_polarity_graph(q):
        """Build the polarity graph of PG(2,q) for prime q."""
        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                for z in range(1, q):
                    inv_z = pow(z, q-2, q)
                    p = (x * inv_z % q, y * inv_z % q, 1)
                    if p not in seen:
                        seen.add(p)
                        points.append(p)
        for x in range(q):
            p = (x, 1, 0)
            if p not in seen:
                seen.add(p)
                points.append(p)
        p = (1, 0, 0)
        if p not in seen:
            seen.add(p)
            points.append(p)
        
        n_pts = len(points)
        adj = [set() for _ in range(n_pts)]
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                if (x1*x2 + y1*y2 + z1*z2) % q == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        for i in range(n_pts):
            adj[i].discard(i)
        
        return adj, n_pts
    
    def check_c4_free(adj, n):
        for u in range(n):
            for w in range(u+1, n):
                if len(adj[u] & adj[w]) >= 2:
                    return False
        return True
    
    def greedy_independent_set_size(adj, n, trials=200):
        best = 0
        for _ in range(trials):
            order = list(range(n))
            random.shuffle(order)
            S = set()
            for v in order:
                if not (adj[v] & S):
                    S.add(v)
            best = max(best, len(S))
        return best
    
    def try_extend(adj, n, target_n, time_budget):
        """Try to extend graph to target_n vertices."""
        deadline = time.time() + time_budget
        cur_adj = [set(s) for s in adj]
        cur_n = n
        
        for _ in range(500):
            if time.time() > deadline:
                break
            if cur_n >= target_n:
                break
            
            # Build conflict graph
            H = [set() for _ in range(cur_n)]
            for u in range(cur_n):
                for w in range(u+1, cur_n):
                    if len(cur_adj[u] & cur_adj[w]) >= 1:
                        H[u].add(w)
                        H[w].add(u)
            
            # Find independent set in H
            best_S = set()
            for trial in range(50):
                order = list(range(cur_n))
                random.shuffle(order)
                S = set()
                for v in order:
                    if not (H[v] & S):
                        S.add(v)
                if len(S) > len(best_S):
                    best_S = set(S)
            
            if len(best_S) == 0:
                break
            
            # Validate: no two in best_S share a common neighbor
            valid = True
            S_list = list(best_S)
            random.shuffle(S_list)
            new_S = set()
            for v in S_list:
                temp = new_S | {v}
                ok = True
                for a, b in combinations(temp, 2):
                    if len(cur_adj[a] & cur_adj[b]) >= 1:
                        ok = False
                        break
                if ok:
                    new_S = temp
            
            if len(new_S) == 0:
                break
            
            # Add vertex
            nv = cur_n
            cur_adj.append(set(new_S))
            for v in new_S:
                cur_adj[v].add(nv)
            cur_n += 1
        
        # Verify
        if not check_c4_free(cur_adj, cur_n):
            return None, 0
        
        alpha = greedy_independent_set_size(cur_adj, cur_n, 100)
        return cur_adj, alpha
    
    # Try different q values
    best_result = None
    best_n = 0
    
    for q in [3, 5, 7]:
        if time.time() > DEADLINE:
            break
        
        adj, n_core = build_polarity_graph(q)
        if not check_c4_free(adj, n_core):
            continue
        
        alpha_core = greedy_independent_set_size(adj, n_core, 50)
        
        # Try to extend
        for target in [39, 40, 41, 42, 43]:
            if time.time() > DEADLINE:
                break
            
            cur_adj, alpha = try_extend(adj, n_core, target, 20.0)
            if cur_adj is not None and alpha < 11:
                if len(cur_adj) > best_n:
                    best_n = len(cur_adj)
                    best_result = cur_adj
    
    # Fallback: use core if no extension worked
    if best_result is None:
        # Use q=3 core
        adj, n_core = build_polarity_graph(3)
        best_result = adj
        best_n = n_core
    
    # Final check
    if not check_c4_free(best_result, best_n):
        adj, n_core = build_polarity_graph(3)
        best_result = adj
        best_n = n_core
    
    edges = []
    for u in range(best_n):
        for v in best_result[u]:
            if u < v:
                edges.append([u, v])
    
    return {'n': best_n, 'edges': edges}
