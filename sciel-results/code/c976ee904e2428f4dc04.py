import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def max_clique_tomita(n, adj, rng):
        """
        Find a maximum clique in the graph defined by adj (list of bitmasks).
        Uses Tomita's algorithm with randomized tie-breaking.
        Returns the list of vertices in the clique.
        """
        best_clique = []
        
        def expand(R, P):
            nonlocal best_clique
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return
            
            # Upper bound
            pc = popcount(P)
            if len(R) + pc <= len(best_clique):
                return
            
            # Pivot selection: choose u in P with max degree in P
            max_deg = -1
            candidates = []
            P_copy = P
            while P_copy:
                v_bit = P_copy & (-P_copy)
                v = v_bit.bit_length() - 1
                P_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [v]
                elif deg == max_deg:
                    candidates.append(v)
                    
            if not candidates:
                return
            
            u = rng.choice(candidates)
            
            # Q = P \ N(u)
            Q = P & ~adj[u]
            
            while Q:
                # Pick a vertex v from Q
                max_v_deg = -1
                v_candidates = []
                Q_copy = Q
                while Q_copy:
                    v_bit = Q_copy & (-Q_copy)
                    i = v_bit.bit_length() - 1
                    Q_copy ^= v_bit
                    deg = popcount(P & adj[i])
                    if deg > max_v_deg:
                        max_v_deg = deg
                        v_candidates = [i]
                    elif deg == max_v_deg:
                        v_candidates.append(i)
                        
                if not v_candidates:
                    break
                
                v = rng.choice(v_candidates)
                
                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)
                
                P &= ~(1 << v)
                Q = Q & P  # Update Q to reflect P changes
                
        expand([], (1 << n) - 1)
        return best_clique

    def greedy_clique_extraction(n, adj, rng):
        """
        Greedy clique extraction: repeatedly find a large clique,
        remove its edges, and continue.
        """
        rem_adj = [0] * n
        for i in range(n):
            rem_adj[i] = adj[i]
        
        cliques = []
        
        # Check if graph is empty
        has_edges = any(rem_adj[i] != 0 for i in range(n))
        
        while has_edges:
            # Find max clique
            clique = max_clique_tomita(n, rem_adj, rng)
            
            if len(clique) < 2:
                # Fallback: take an edge
                found = False
                for u in range(n):
                    if rem_adj[u]:
                        v_bit = rem_adj[u] & (-rem_adj[u])
                        v = v_bit.bit_length() - 1
                        if (rem_adj[u] >> v) & 1:
                            clique = [u, v]
                            found = True
                            break
                if not found:
                    break
            
            # Remove edges of clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
            
            # Update has_edges
            has_edges = any(rem_adj[i] != 0 for i in range(n))
        
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        # Baseline: each edge as its own clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        # Try multiple random restarts
        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            rng = random.Random(seed + attempt)
            
            result = greedy_clique_extraction(n, adj, rng)
            
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
            
            attempt += 1
            
        return best_cliques

    # Solve both instances
    time_budget_1 = 35.0
    time_budget_2 = 75.0
    
    current_time = time.time()
    remaining = DEADLINE - current_time
    if time_budget_1 + time_budget_2 > remaining:
        scale = remaining / (time_budget_1 + time_budget_2)
        time_budget_1 *= scale
        time_budget_2 *= scale

    result1 = solve_instance(32, 0.6, 601, time_budget_1)
    
    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, time_budget_2)
    
    return [result1, result2]
