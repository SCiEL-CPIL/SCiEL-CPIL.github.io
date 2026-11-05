import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_result = {'n': 39, 'edges': []}
    
    for n in [47, 46, 45, 44]:
        if time.time() > deadline - 5:
            break
            
        min_deg = n - 39
        
        # State
        best_energy = float('inf')
        best_adj = None
        
        # Number of restarts
        # We will run until time runs out for this n, or find a solution
        
        while time.time() < deadline - 5:
            # Initialize
            adj = [0] * n
            p = min_deg / n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            current_adj = adj
            current_energy = _calc_energy_full(current_adj, n, min_deg)
            
            T = 10.0
            cooling = 0.9995
            min_T = 0.01
            
            # Run SA
            steps_in_burst = 20000 # Can be larger now
            for _ in range(steps_in_burst):
                if time.time() > deadline - 5:
                    break
                    
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                mask_u = 1 << v
                mask_v = 1 << u
                
                is_edge = (current_adj[u] & mask_u) != 0
                
                # Calculate delta energy
                delta_c4 = 0
                Su = current_adj[u]
                Sv = current_adj[v]
                
                # Count paths v-w-x-u
                # Iterate over neighbors of v
                # To iterate bits, we can use a while loop or precomputed list
                # Since n is small, we can just loop 0..n-1
                # But bit iteration is faster if we use bit ops
                
                # Let's just loop 0..n-1 for simplicity first, n=47 is small
                # But 20000 steps * 47 * 2 = ~2M ops. OK.
                
                # Actually, iterating set bits is better.
                # Helper to iterate set bits
                # But Python doesn't have a builtin.
                # We can use:
                # x = Sv
                # while x:
                #   lsb = x & -x
                #   w = lsb.bit_length() - 1
                #   ...
                #   x ^= lsb
                
                # Let's use the simple loop for now, n is small
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if (Sv >> w) & 1: # w is neighbor of v
                        # Check if w is connected to any neighbor of u (excluding v)
                        # adj[w] & Su
                        # But we must exclude u? No, w is not u.
                        # Su contains v. We don't want to count w-v-u-w? No.
                        # The path is v-w-x-u.
                        # x must be in N(u).
                        # The edge is (w,x).
                        # So we count bits in adj[w] & Su.
                        # Does this count x=v?
                        # v is in Su. Is (w,v) an edge? Yes, w in N(v).
                        # So if we count (w,v), that corresponds to path v-w-v-u? No, w-v is edge, v-u is edge.
                        # This would be a triangle v-w-u? No, w-u is not necessarily edge.
                        # Wait.
                        # Path v-w-x-u.
                        # If x=v, path is v-w-v-u. This requires edge (w,v) and (v,u).
                        # This is not a C4.
                        # So we must exclude x=v from Su?
                        # Su = N(u). v is in Su.
                        # So `adj[w] & Su` includes v if (w,v) is edge.
                        # We should mask out v from Su?
                        # Su_masked = Su & ~(1 << v)
                        # Similarly, w is in Sv.
                        
                        # Let's refine:
                        # We want x in N(u) \ {v}.
                        # And edge (w,x).
                        
                        # So:
                        # count += (adj[w] & (Su & ~(1 << v))).bit_count()
                        
                        # But we are looping w.
                        # This is O(deg(v) * 1).
                        
                        pass # Placeholder

                # This inner loop is getting complex.
                # Let's stick to full recalc if it's fast enough?
                # 1000 ops per step. 20000 steps = 20M ops.
                # Python 20M ops ~ 1-2s.
                # 115s budget. ~50-100 restarts.
                # This might be enough.
                
                # Let's try full recalc first for simplicity and correctness.
                
                if is_edge:
                    current_adj[u] &= ~mask_u
                    current_adj[v] &= ~mask_v
                else:
                    current_adj[u] |= mask_u
                    current_adj[v] |= mask_v
                
                new_energy = _calc_energy_full(current_adj, n, min_deg)
                delta = new_energy - current_energy
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_adj = current_adj[:]
                        if new_energy == 0:
                            return _adj_to_result(current_adj, n)
                else:
                    if is_edge:
                        current_adj[u] |= mask_u
                        current_adj[v] |= mask_v
                    else:
                        current_adj[u] &= ~mask_u
                        current_adj[v] &= ~mask_v
                
                T *= cooling
                if T < min_T:
                    T = min_T
            
            if best_energy == 0:
                return _adj_to_result(best_adj, n)
                
    return best_result

def _calc_energy_full(adj, n, min_deg):
    c4_count = 0
    for u in range(n):
        bu = adj[u]
        for v in range(u + 1, n):
            c4_count += (bu & adj[v]).bit_count()
            
    deg_penalty = 0
    for i in range(n):
        d = adj[i].bit_count()
        if d < min_deg:
            deg_penalty += min_deg - d
            
    return c4_count * 10 + deg_penalty

def _adj_to_result(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return {'n': n, 'edges': edges}
