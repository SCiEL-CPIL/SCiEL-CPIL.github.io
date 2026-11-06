import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0

    def time_left():
        return deadline - (time.time() - start_time)

    def orient_to_adj(n, orient):
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orient[idx] == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1
        return adj

    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    def get_out_masks(n, adj):
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
        return out_mask

    def verify_tournament(n, adj, out_mask):
        """
        Fast verification using bitmasks.
        A set of 4 vertices {a,b,c,d} is dominated if there exists v such that
        out_mask[v] contains all of a,b,c,d.
        """
        # Precompute masks for all 4-sets is too slow for n~60.
        # Instead, for each v, check which 4-sets it dominates? No.
        # Check each 4-set.
        # Optimization: If n is small enough, iterate.
        # For n=60, C(60,4) ~ 487k. This is fast in Python if optimized?
        # 487k * (4 bit ops) is very fast.
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            dominated = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    dominated = True
                    break
            if not dominated:
                return False
        return True

    def count_violations(n, adj, out_mask):
        violations = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            dominated = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    dominated = True
                    break
            if not dominated:
                violations += 1
        return violations

    def structured_construction(n, seed=42):
        """
        Strategy:
        Divide vertices into Hubs (H) and Clients (C).
        |H| = h, |C| = c.
        1. Hubs dominate all Clients? No, that's too strong and wastes edges.
        2. Random tournament on Hubs.
        3. Random tournament on Clients.
        4. Hub-Client edges: 
           Each Hub h has a "profile" P_h (subset of Clients it dominates).
           We want the union of profiles to cover all 4-sets of Clients? 
           No, a 4-set can include Hubs too.
           
           Let's refine:
           The condition is: For every 4-set S, exists v in V such that v->S.
           
           If we make Hubs dominate all Clients, then any 4-set consisting only of Clients is dominated by any Hub.
           But 4-sets with Hubs are harder.
           
           Let's try:
           - Hubs form a regular-ish tournament.
           - Each Hub dominates a random subset of Clients with prob p.
           - Clients form a random tournament.
           
           To ensure robustness, we can use a "nested" structure or just random with high p.
           Actually, if we just pick n small enough and p high, it might work.
           
           Alternative: 
           Use a known construction? 
           Schütte's problem k=4. 
           Known smallest is 67. 
           We can try to construct n=67, 68... using this heuristic.
        """
        random.seed(seed)
        adj = [[0]*n for _ in range(n)]
        
        # Choose number of hubs
        # If we have h hubs, and they dominate clients with prob p,
        # a specific client 4-set {c1,c2,c3,c4} is dominated by hub h with prob p^4.
        # Prob NOT dominated by any hub = (1-p^4)^h.
        # We want this small.
        # If p=0.8, p^4=0.4096.
        # If h=10, (1-0.4096)^10 = 0.59^10 ~ 0.005.
        # So most client 4-sets are dominated by a hub.
        # What about 4-sets with hubs?
        # A 4-set {h1, h2, h3, h4} must be dominated by some v.
        # If v is a hub, it must dominate the other 3 hubs.
        # If v is a client, it must dominate the 4 hubs.
        
        # Let's set h = 15, p = 0.7.
        h = 15
        if h > n: h = n // 2
        c = n - h
        
        H = list(range(h))
        C = list(range(h, n))
        
        # 1. Orient Hubs
        for i in range(len(H)):
            for j in range(i+1, len(H)):
                if random.random() < 0.5:
                    adj[H[i]][H[j]] = 1
                else:
                    adj[H[j]][H[i]] = 1
        
        # 2. Orient Clients
        for i in range(len(C)):
            for j in range(i+1, len(C)):
                if random.random() < 0.5:
                    adj[C[i]][C[j]] = 1
                else:
                    adj[C[j]][C[i]] = 1
        
        # 3. Orient Hub-Client
        # Each hub dominates each client with prob p
        p = 0.7
        for d in H:
            for c_idx in C:
                if random.random() < p:
                    adj[d][c_idx] = 1
                else:
                    adj[c_idx][d] = 1
        
        return adj

    def local_search_hybrid(n, adj, out_mask, max_iters=5000):
        """
        Flip edges to reduce violations.
        Focus on edges between Hubs and Clients, as they are more flexible.
        """
        # Identify hubs and clients based on initial structure?
        # Or just treat all edges equally but prioritize high-degree vertices?
        # Let's just do random edge flips that reduce violation count.
        
        current_violations = count_violations(n, adj, out_mask)
        if current_violations == 0:
            return adj, out_mask, True
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            # Pick a random edge to flip
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            
            # Flip edge i->j
            adj[i][j] = 1 - adj[i][j]
            adj[j][i] = 1 - adj[j][i]
            
            # Update masks
            if adj[i][j]:
                out_mask[i] |= (1 << j)
                out_mask[j] &= ~(1 << i)
            else:
                out_mask[i] &= ~(1 << j)
                out_mask[j] |= (1 << i)
            
            new_violations = count_violations(n, adj, out_mask)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if current_violations == 0:
                    return adj, out_mask, True
            else:
                # Revert
                adj[i][j] = 1 - adj[i][j]
                adj[j][i] = 1 - adj[j][i]
                if adj[i][j]:
                    out_mask[i] |= (1 << j)
                    out_mask[j] &= ~(1 << i)
                else:
                    out_mask[i] &= ~(1 << j)
                    out_mask[j] |= (1 << i)
                
                # Occasionally accept worse move to escape local minima (Simulated Annealing style)
                if random.random() < 0.05:
                    adj[i][j] = 1 - adj[i][j]
                    adj[j][i] = 1 - adj[j][i]
                    if adj[i][j]:
                        out_mask[i] |= (1 << j)
                        out_mask[j] &= ~(1 << i)
                    else:
                        out_mask[i] &= ~(1 << j)
                        out_mask[j] |= (1 << i)
                    current_violations = new_violations

        return adj, out_mask, current_violations == 0

    # Fallback: Paley(83)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj_paley = [[0]*p for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                adj_paley[i][j] = 1
            else:
                adj_paley[j][i] = 1
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Try structured construction for n from 60 upwards
    # We know f(4) >= 31, and smallest known is 67.
    # Let's try 60 to 70.
    for n in range(60, 75):
        if time_left() < 10:
            break
        
        found = False
        for seed in range(50):
            if time_left() < 5:
                break
            
            adj = structured_construction(n, seed=seed)
            out_mask = get_out_masks(n, adj)
            
            # Check if already valid
            if verify_tournament(n, adj, out_mask):
                best_n = n
                best_orient = adj_to_orient(n, adj)
                found = True
                break
            
            # If not, try local search
            # Only do local search if violation count is low enough?
            # For n=60, violations might be high.
            # Let's just try a few steps of local search.
            adj, out_mask, success = local_search_hybrid(n, adj, out_mask, max_iters=2000)
            
            if success:
                # Final verify
                if verify_tournament(n, adj, out_mask):
                    best_n = n
                    best_orient = adj_to_orient(n, adj)
                    found = True
                    break
        
        if found:
            break
    
    return {'n': best_n, 'orientation': best_orient}
