import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s limit, leave 40s buffer)
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Uses bitmasks for speed.
        """
        # Precompute neighbor bitmasks
        neigh = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            neigh[u] = mask
        
        cliques = []
        all_mask = (1 << n) - 1
        
        def expand(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            
            # Pivot selection: choose u in P union X that maximizes |P & N(u)|
            union = P | X
            if union:
                # Find pivot with max |P & neigh[u]|
                best_u = -1
                best_count = -1
                # Iterate over bits in union
                u = union
                while u:
                    lsb = u & (-u)
                    v = lsb.bit_length() - 1
                    u ^= lsb
                    count = bin(P & neigh[v]).count('1')
                    if count > best_count:
                        best_count = count
                        best_u = v
                u = best_u
            else:
                u = None
                
            # Candidates: P \ N(u) if u is not None, else P
            if u is not None:
                candidates = P & ~neigh[u]
            else:
                candidates = P
                
            while candidates:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                new_R = R | lsb
                new_P = P & neigh[v]
                new_X = X & neigh[v]
                expand(new_R, new_P, new_X)
                
                P &= ~lsb
                X |= lsb
                
        expand(0, all_mask, 0)
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        # Filter cliques with size >= 2
        relevant_cliques = [c for c in all_cliques if bin(c).count('1') >= 2]
        
        if not relevant_cliques:
            return []
        
        # Convert cliques to sets of vertex indices for easier manipulation
        cliques_as_sets = []
        for c in relevant_cliques:
            s = set()
            temp = c
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                s.add(v)
                temp ^= lsb
            cliques_as_sets.append(s)
        
        # Formulate ILP
        # Minimize sum(x_i)
        # s.t. sum_{j in C} x_j >= 1 for all C in relevant_cliques
        # 0 <= x_i <= 1, x_i integer
        
        m = len(cliques_as_sets)
        A = np.zeros((m, n), dtype=np.int8)
        for i, clique in enumerate(cliques_as_sets):
            for v in clique:
                A[i, v] = 1
                
        c = np.ones(n, dtype=np.float64)
        
        # Bounds: (0, 1)
        bounds_obj = np.array([(0, 1)] * n)
        
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            
            constraints = LinearConstraint(A, 1, np.inf)
            bounds_obj = Bounds(0, 1)
            
            # Set time limit per instance to 30s
            # We have 3 instances, so 90s total. 
            # Check global deadline
            elapsed = time.time() - start_time
            time_left = deadline - elapsed
            if time_left <= 5:
                # Not enough time, return greedy fallback
                return greedy_hit_set(cliques_as_sets, n)
            
            instance_time_limit = max(5, (time_left - 5) / 3)
            
            res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                       bounds=bounds_obj, 
                       options={"time_limit": instance_time_limit, "presolve": True})
            
            if res.success or res.status == 1: # 1 might be optimal, 0 is success usually? 
                # res.x gives the solution. It might be fractional if not integer? 
                # We requested integer.
                # If res.x is None, fallback
                if res.x is not None:
                    sol = [i for i in range(n) if res.x[i] > 0.5]
                    # Verify validity just in case
                    if is_valid_sol(sol, cliques_as_sets):
                        return sol
            # If not successful or invalid, try incumbent
            if res.x is not None:
                sol = [i for i in range(n) if res.x[i] > 0.5]
                if is_valid_sol(sol, cliques_as_sets):
                    return sol
        except Exception:
            pass
            
        return greedy_hit_set(cliques_as_sets, n)

    def is_valid_sol(sol_set, cliques):
        sol_set = set(sol_set)
        for c in cliques:
            if not sol_set.intersection(c):
                return False
        return True

    def greedy_hit_set(cliques, n):
        # Greedy set cover approximation
        # At each step, pick vertex that covers the most uncovered cliques
        remaining_cliques = [set(c) for c in cliques]
        covered_cliques = [False] * len(remaining_cliques)
        selected = []
        
        while True:
            # Check if all covered
            if all(covered_cliques):
                break
            
            # Count how many uncovered cliques each vertex hits
            hits = [0] * n
            for i, c in enumerate(remaining_cliques):
                if not covered_cliques[i]:
                    for v in c:
                        hits[v] += 1
            
            max_hits = max(hits)
            if max_hits == 0:
                break # Should not happen if input is valid
                
            # Pick vertex with max hits (tie-break by index)
            v = hits.index(max_hits)
            selected.append(v)
            
            # Update coverage
            for i, c in enumerate(remaining_cliques):
                if not covered_cliques[i] and v in c:
                    covered_cliques[i] = True
                    
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))
        
        if time.time() > deadline:
             # Fill remaining with greedy quickly if we run out of time
             # But we should have enough time.
             break
             
    # If loop broke early, ensure we have 3 results
    # This shouldn't happen with the time allocation above, but safe.
    while len(results) < 3:
         results.append([])
         
    return results
