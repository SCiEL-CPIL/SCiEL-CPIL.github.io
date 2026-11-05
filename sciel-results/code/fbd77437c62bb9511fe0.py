import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_score = -1
    
    # We use a structural approach:
    # 1. Try different initial structures (random, structured)
    # 2. Use simulated annealing with edge flips
    # 3. Greedy fill at the end
    
    # Precompute some useful data
    # For each vertex, list of all other vertices
    others = {v: [u for u in range(n) if u != v] for v in range(n)}
    
    def compute_cn(adj):
        """Compute common neighbor matrix from adjacency matrix."""
        cn = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                cnt = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u][w] and adj[v][w]:
                        cnt += 1
                cn[u][v] = cnt
                cn[v][u] = cnt
        return cn
    
    def get_penalty(cn):
        """Sum of max(0, cn[u][v] - 2) for all u < v."""
        pen = 0
        for u in range(n):
            for v in range(u+1, n):
                if cn[u][v] > 2:
                    pen += cn[u][v] - 2
        return pen
    
    def try_greedy_fill(adj, cn, max_time):
        """Greedily add edges until no more can be added. Returns final adj, cn, m."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for u, v in all_pairs:
                if adj[u][v]:
                    continue
                # Check if we can add (u, v)
                valid = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[v][w] and cn[u][w] >= 2:
                        valid = False
                        break
                    if adj[u][w] and cn[v][w] >= 2:
                        valid = False
                        break
                if not valid:
                    continue
                # Add edge (u, v)
                adj[u][v] = 1
                adj[v][u] = 1
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[v][w]:
                        cn[u][w] += 1
                        cn[w][u] += 1
                    if adj[u][w]:
                        cn[v][w] += 1
                        cn[w][v] += 1
                improved = True
        m = sum(adj[u][v] for u in range(n) for v in range(u+1, n))
        return adj, cn, m
    
    def simulated_annealing(adj_init, cn_init, time_budget):
        """Run simulated annealing on the given initial graph. Returns best adj, cn, m found."""
        adj = [row[:] for row in adj_init]
        cn = [row[:] for row in cn_init]
        m = sum(adj[u][v] for u in range(n) for v in range(u+1, n))
        penalty = get_penalty(cn)
        
        best_m = m
        best_adj = [row[:] for row in adj]
        best_cn = [row[:] for row in cn]
        best_score = m - 5 * penalty
        
        T = 8.0
        cooling = 0.995
        min_T = 0.001
        
        def flip_edge(u, v, adding):
            nonlocal m, penalty
            adj[u][v] = 1 if adding else 0
            adj[v][u] = 1 if adding else 0
            
            delta = 1 if adding else -1
            
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[v][w]:
                    cn[u][w] += delta
                    cn[w][u] += delta
                if adj[u][w]:
                    cn[v][w] += delta
                    cn[w][v] += delta
        
        while T > min_T and time.time() < time_budget:
            # Do multiple steps at each temperature
            for _ in range(200):
                if time.time() > time_budget:
                    break
                
                u, v = random.choice(all_pairs)
                adding = adj[u][v] == 0
                
                # Calculate delta penalty
                delta_pen = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Pair (u, w)
                    if adj[v][w]:
                        old_cn = cn[u][w]
                        new_cn = old_cn + (1 if adding else -1)
                        delta_pen += max(0, new_cn - 2) - max(0, old_cn - 2)
                    # Pair (v, w)
                    if adj[u][w]:
                        old_cn = cn[v][w]
                        new_cn = old_cn + (1 if adding else -1)
                        delta_pen += max(0, new_cn - 2) - max(0, old_cn - 2)
                
                delta_m = 1 if adding else -1
                delta_score = delta_m - 5 * delta_pen
                
                if delta_score > 0:
                    accept = True
                else:
                    if T < 0.01:
                        accept = random.random() < math.exp(delta_score / T)
                    else:
                        accept = random.random() < math.exp(delta_score / T)
                
                if accept:
                    flip_edge(u, v, adding)
                    m += delta_m
                    penalty += delta_pen
                    
                    current_score = m - 5 * penalty
                    if current_score > best_score:
                        best_score = current_score
                        best_m = m
                        best_adj = [row[:] for row in adj]
                        best_cn = [row[:] for row in cn]
            
            T *= cooling
        
        # Greedy fill on the best found
        adj_final, cn_final, m_final = try_greedy_fill(best_adj, best_cn, time_budget)
        return adj_final, cn_final, m_final
    
    # Try multiple different strategies
    
    # Strategy 1: Random initialization
    for restart in range(8):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with varying density
        p = 0.3 + 0.1 * restart
        adj = [[0]*n for _ in range(n)]
        for u, v in all_pairs:
            if random.random() < p:
                adj[u][v] = 1
                adj[v][u] = 1
        
        cn = compute_cn(adj)
        
        time_budget = deadline - 5
        adj_final, cn_final, m_final = simulated_annealing(adj, cn, time_budget)
        
        # Verify validity
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if cn_final[u][v] > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and m_final > best_score:
            best_score = m_final
            best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj_final[u][v]]
    
    # Strategy 2: Start from a known good structure - complement of a matching
    # or a sparse random graph
    if time.time() < deadline - 10:
        # Start with a very sparse graph and let SA build up
        adj = [[0]*n for _ in range(n)]
        # Add a few random edges
        for _ in range(30):
            u, v = random.choice(all_pairs)
            adj[u][v] = 1
            adj[v][u] = 1
        
        cn = compute_cn(adj)
        
        time_budget = deadline - 5
        adj_final, cn_final, m_final = simulated_annealing(adj, cn, time_budget)
        
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if cn_final[u][v] > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and m_final > best_score:
            best_score = m_final
            best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj_final[u][v]]
    
    # Strategy 3: Use a structured initialization based on cyclic patterns
    if time.time() < deadline - 10:
        # Try a circulant-like graph
        adj = [[0]*n for _ in range(n)]
        # Add edges based on a pattern
        for i in range(n):
            for j in range(1, 7):  # Connect to next 6 vertices
                if i != (i+j) % n:
                    adj[i][(i+j) % n] = 1
                    adj[(i+j) % n][i] = 1
        
        cn = compute_cn(adj)
        
        time_budget = deadline - 5
        adj_final, cn_final, m_final = simulated_annealing(adj, cn, time_budget)
        
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if cn_final[u][v] > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and m_final > best_score:
            best_score = m_final
            best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj_final[u][v]]
    
    # If we didn't find anything valid, return an empty graph
    if not best_edges:
        return []
    
    return best_edges
