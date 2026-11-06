import numpy as np
import random
import time

def construct():
    start_time = time.time()
    time_limit = 110.0

    def find_graph(n):
        # Adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        
        # Initialize random
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i,j] = True
                    adj[j,i] = True
        
        # Precompute indices for K4 and I6 checks?
        # We can just use vectorized operations for delta calculation.
        
        def calc_delta(u, v):
            # u, v are the edge being flipped
            # Check if currently edge exists
            has_edge = adj[u, v]
            
            # Delta K4
            # K4s containing u,v.
            # Other vertices w,x.
            # Condition for K4: adj[u,w], adj[u,x], adj[v,w], adj[v,x], adj[w,x] all True.
            # We can vectorize over w,x.
            # Let M be a matrix where M[w,x] = adj[u,w] & adj[u,x] & adj[v,w] & adj[v,x] & adj[w,x]
            # We need to count True in M for w != u,v and x != u,v and w < x.
            
            # Create mask for w, x
            # w, x in 0..n-1
            # Exclude u, v
            
            # This might be slow to construct masks every step.
            # Alternative: Iterate w, x but use numpy slicing?
            
            # Let's stick to Python loops for K4 (528 checks) and try to optimize I6.
            # For I6, we need to check 40920 sets.
            # We can precompute the list of 4-sets of other vertices.
            
            pass

        # Let's use the bitmask approach but with PyPy-like optimizations? 
        # Or just accept it might be slow and reduce steps.
        
        # Actually, let's try the bitmask approach with precomputed K4s.
        # For I6, we will just recalculate the total cost from scratch every 100 steps?
        # No, that's too slow.
        
        # Let's try a different strategy: 
        # Use a SAT solver via `subprocess` if `minisat` or `pysat` is available? 
        # The prompt says "standard library, NumPy and SciPy". No pysat.
        
        # Let's optimize the bitmask I6 check.
        # We can precompute for each pair (u,v) the list of 4-tuples (w,x,y,z).
        # 40920 tuples per edge. 595 edges. 24M tuples.
        # Memory: 24M * 4 ints * 4 bytes = 400MB. Might be too high.
        
        # Let's reduce n to 30 for testing? No, we want 35.
        
        # Let's try to just run the simple bitmask SA and hope it's fast enough.
        # 40k bitwise ops per step. 
        # If 1 step takes 1ms, 1000 steps take 1s.
        # 110s allows 110,000 steps.
        # This should be enough for convergence.
        
        vertices = list(range(n))
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    
        k4s_for_edge = {}
        for u in range(n):
            for v in range(u+1, n):
                k4s_for_edge[(u,v)] = []
                for w in range(n):
                    if w == u or w == v: continue
                    for x in range(w+1, n):
                        if x == u or x == v: continue
                        k4s_for_edge[(u,v)].append((w,x))
        
        def is_k4_mask(a,b,c,d, adj_m):
            return (adj_m[a] & (1<<b)) and (adj_m[a] & (1<<c)) and (adj_m[a] & (1<<d)) and \
                   (adj_m[b] & (1<<c)) and (adj_m[b] & (1<<d)) and \
                   (adj_m[c] & (1<<d))
                   
        def calc_cost_mask(adj_m):
            cost = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            if is_k4_mask(i,j,k,l, adj_m):
                                cost += 1
            # I6 cost
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            for m in range(l+1, n):
                                for o in range(m+1, n):
                                    if not (adj_m[i] & (1<<j)) and not (adj_m[i] & (1<<k)) and not (adj_m[i] & (1<<l)) and \
                                       not (adj_m[i] & (1<<m)) and not (adj_m[i] & (1<<o)) and \
                                       not (adj_m[j] & (1<<k)) and not (adj_m[j] & (1<<l)) and \
                                       not (adj_m[j] & (1<<m)) and not (adj_m[j] & (1<<o)) and \
                                       not (adj_m[k] & (1<<l)) and not (adj_m[k] & (1<<m)) and \
                                       not (adj_m[k] & (1<<o)) and \
                                       not (adj_m[l] & (1<<m)) and not (adj_m[l] & (1<<o)) and \
                                       not (adj_m[m] & (1<<o)):
                                        cost += 1
            return cost

        current_cost = calc_cost_mask(adj_mask)
        if current_cost == 0:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj_mask[i] & (1<<j):
                        edges.append([i,j])
            return {'n': n, 'edges': edges}
            
        best_cost = current_cost
        best_adj = adj_mask[:]
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            has_edge = bool(adj_mask[u] & (1<<v))
            
            delta_k4 = 0
            for (w,x) in k4s_for_edge[(u,v)]:
                if has_edge:
                    if is_k4_mask(u,v,w,x, adj_mask):
                        delta_k4 -= 1
                else:
                    if (adj_mask[u] & (1<<w)) and (adj_mask[u] & (1<<x)) and \
                       (adj_mask[v] & (1<<w)) and (adj_mask[v] & (1<<x)) and \
                       (adj_mask[w] & (1<<x)):
                        delta_k4 += 1
                        
            delta_i6 = 0
            # Generate 4-tuples on the fly
            for w in range(n):
                if w == u or w == v: continue
                for x in range(w+1, n):
                    if x == u or x == v: continue
                    for y in range(x+1, n):
                        if y == u or y == v: continue
                        for z in range(y+1, n):
                            if z == u or z == v: continue
                            # Check independence of {w,x,y,z}
                            if (adj_mask[w] & (1<<x)) or (adj_mask[w] & (1<<y)) or (adj_mask[w] & (1<<z)) or \
                               (adj_mask[x] & (1<<y)) or (adj_mask[x] & (1<<z)) or \
                               (adj_mask[y] & (1<<z)):
                                continue
                            
                            u_conn = (adj_mask[u] & (1<<w)) or (adj_mask[u] & (1<<x)) or (adj_mask[u] & (1<<y)) or (adj_mask[u] & (1<<z))
                            v_conn = (adj_mask[v] & (1<<w)) or (adj_mask[v] & (1<<x)) or (adj_mask[v] & (1<<y)) or (adj_mask[v] & (1<<z))
                            
                            if has_edge:
                                if not u_conn and not v_conn:
                                    delta_i6 += 1
                            else:
                                if not u_conn and not v_conn:
                                    delta_i6 -= 1
                                    
            total_delta = delta_k4 + delta_i6
            
            if total_delta <= 0 or random.random() < pow(2.71828, -total_delta / T):
                if has_edge:
                    adj_mask[u] &= ~(1<<v)
                    adj_mask[v] &= ~(1<<u)
                else:
                    adj_mask[u] |= (1<<v)
                    adj_mask[v] |= (1<<u)
                
                current_cost += total_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
                    if best_cost == 0:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if best_adj[i] & (1<<j):
                                    edges.append([i,j])
                        return {'n': n, 'edges': edges}
                
            T *= cooling
            if T < T_min:
                T = T_min
                
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i] & (1<<j):
                    edges.append([i,j])
        return {'n': n, 'edges': edges}

    result = find_graph(35)
    
    # Verify
    n = result['n']
    edges = result['edges']
    adj = [0]*n
    for u,v in edges:
        adj[u] |= (1<<v)
        adj[v] |= (1<<u)
        
    def is_k4(a,b,c,d):
        return (adj[a] & (1<<b)) and (adj[a] & (1<<c)) and (adj[a] & (1<<d)) and \
               (adj[b] & (1<<c)) and (adj[b] & (1<<d)) and \
               (adj[c] & (1<<d))
    def is_independent_6(a,b,c,d,e,f):
        return not (adj[a] & (1<<b)) and not (adj[a] & (1<<c)) and not (adj[a] & (1<<d)) and \
               not (adj[a] & (1<<e)) and not (adj[a] & (1<<f)) and \
               not (adj[b] & (1<<c)) and not (adj[b] & (1<<d)) and \
               not (adj[b] & (1<<e)) and not (adj[b] & (1<<f)) and \
               not (adj[c] & (1<<d)) and not (adj[c] & (1<<e)) and \
               not (adj[c] & (1<<f)) and \
               not (adj[d] & (1<<e)) and not (adj[d] & (1<<f)) and \
               not (adj[e] & (1<<f))
               
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if is_k4(i,j,k,l):
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            for o in range(m+1, n):
                                if is_independent_6(i,j,k,l,m,o):
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if not valid:
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
        
    return result
