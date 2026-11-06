import time
import random
import sys

def construct():
    deadline = time.time() + 290.0
    
    # We will try to find a valid graph for n = 42 down to 37.
    # We use Simulated Annealing to minimize the number of K5s and independent 5-sets.
    
    def count_violations(adj, n):
        # adj is a list of bitmasks
        k5_count = 0
        alpha5_count = 0
        # Iterate over all 5-subsets
        # This is O(n^5) which is too slow for n=42 in Python if done naively.
        # We need a faster way or rely on the incremental update.
        # For initialization, we can do a slower count or just start SA and update.
        # Let's do a somewhat optimized count for initialization.
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    # Edge exists
                    for k in range(j+1, n):
                        if (adj[i] >> k) & 1 and (adj[j] >> k) & 1:
                            for l in range(k+1, n):
                                if (adj[i] >> l) & 1 and (adj[j] >> l) & 1 and (adj[k] >> l) & 1:
                                    for m in range(l+1, n):
                                        if (adj[i] >> m) & 1 and (adj[j] >> m) & 1 and (adj[k] >> m) & 1 and (adj[l] >> m) & 1:
                                            k5_count += 1
                else:
                    # No edge
                    for k in range(j+1, n):
                        if not ((adj[i] >> k) & 1) and not ((adj[j] >> k) & 1):
                            for l in range(k+1, n):
                                if not ((adj[i] >> l) & 1) and not ((adj[j] >> l) & 1) and not ((adj[k] >> l) & 1):
                                    for m in range(l+1, n):
                                        if not ((adj[i] >> m) & 1) and not ((adj[j] >> m) & 1) and not ((adj[k] >> m) & 1) and not ((adj[l] >> m) & 1):
                                            alpha5_count += 1
        return k5_count + alpha5_count

    def get_delta(adj, n, u, v):
        # Returns change in violation count if we flip edge (u, v)
        # Current state: edge exists if (adj[u] >> v) & 1
        edge_exists = (adj[u] >> v) & 1
        delta = 0
        
        # Iterate over all pairs (a, b) from the remaining n-2 vertices
        # such that a > b? No, just all combinations of 3 vertices from the rest.
        # We need to check triples (a, b, c)
        # To avoid duplicates, we can iterate a < b < c
        for a in range(n):
            if a == u or a == v: continue
            for b in range(a+1, n):
                if b == u or b == v: continue
                for c in range(b+1, n):
                    if c == u or c == v: continue
                    
                    # Check the 9 edges:
                    # u-a, u-b, u-c, v-a, v-b, v-c, a-b, b-c, a-c
                    # If all are 1, then {u,v,a,b,c} is a K5 if u-v is 1.
                    # If all are 0, then {u,v,a,b,c} is an independent set if u-v is 0.
                    
                    e_ua = (adj[u] >> a) & 1
                    e_ub = (adj[u] >> b) & 1
                    e_uc = (adj[u] >> c) & 1
                    e_va = (adj[v] >> a) & 1
                    e_vb = (adj[v] >> b) & 1
                    e_vc = (adj[v] >> c) & 1
                    e_ab = (adj[a] >> b) & 1
                    e_bc = (adj[b] >> c) & 1
                    e_ac = (adj[a] >> c) & 1
                    
                    all_ones = (e_ua and e_ub and e_uc and e_va and e_vb and e_vc and e_ab and e_bc and e_ac)
                    all_zeros = (not e_ua and not e_ub and not e_uc and not e_va and not e_vb and not e_vc and not e_ab and not e_bc and not e_ac)
                    
                    if all_ones:
                        if edge_exists:
                            # It is currently a K5. If we remove edge, it is no longer a K5.
                            # So K5 count decreases by 1.
                            delta -= 1
                        else:
                            # It is not currently a K5 (missing u-v). If we add edge, it becomes a K5.
                            # So K5 count increases by 1.
                            delta += 1
                    elif all_zeros:
                        if edge_exists:
                            # It is currently NOT an independent set (has u-v edge).
                            # If we remove edge, it becomes an independent set.
                            # So Alpha5 count increases by 1.
                            delta += 1
                        else:
                            # It is currently an independent set.
                            # If we add edge, it is no longer an independent set.
                            # So Alpha5 count decreases by 1.
                            delta -= 1
        return delta

    def sa_search(n, time_limit):
        start_time = time.time()
        
        # Initialize adj
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_violations(adj, n)
        best_cost = current_cost
        best_adj = adj[:]
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if current_cost == 0:
                return adj
            
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            delta = get_delta(adj, n, u, v)
            new_cost = current_cost + delta
            
            if new_cost < current_cost or random.random() < pow(2.718, -delta / T):
                # Flip edge
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    
                T = max(T_min, T * cooling)
            else:
                T = max(T_min, T * cooling)
                
        return best_adj

    # Try to find a valid graph for n = 42 down to 37
    # We prioritize higher n
    for target_n in range(42, 36, -1):
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on n
        # Larger n is harder, give it more time?
        # But if we find a valid graph for 42, we stop.
        # So we should try 42 first with a decent time budget.
        time_budget = 30.0
        if target_n > 40:
            time_budget = 50.0
            
        if time.time() + time_budget > deadline:
            time_budget = deadline - time.time() - 5.0
            if time_budget < 5.0:
                break
                
        adj_found = sa_search(target_n, time_budget)
        
        # Verify
        if count_violations(adj_found, target_n) == 0:
            # Convert to edge list
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if (adj_found[i] >> j) & 1:
                        edges.append((i, j))
            return target_n, edges
            
    # If we failed to find a valid graph for any n >= 37,
    # we should return the Paley graph (37 vertices) as a fallback.
    # But we can also return the best_adj from the last SA run if it's valid?
    # No, we only return if cost is 0.
    
    # Fallback: Paley Graph
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
                
    return q, edges
