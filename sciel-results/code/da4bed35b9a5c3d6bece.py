import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(list(R))
            return
        
        pivot = None
        max_deg = -1
        for v in P.union(X):
            deg = len(adj[v].intersection(P))
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            return
            
        for v in list(P):
            if v not in adj[pivot]:
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)

    def solve_greedy(n, valid_cliques):
        M = len(valid_cliques)
        masks = [0] * n
        for i, clique in enumerate(valid_cliques):
            bit = 1 << i
            for v in clique:
                masks[v] |= bit
                
        all_cliques_mask = (1 << M) - 1
        best_sol = []
        best_size = float('inf')
        
        for _ in range(50):
            uncovered = all_cliques_mask
            selected = []
            
            while uncovered:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if not (uncovered & masks[v]):
                        continue
                    count = (uncovered & masks[v]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count and random.random() < 0.5:
                        best_v = v
                        
                if best_v == -1:
                    break
                    
                selected.append(best_v)
                uncovered &= ~masks[best_v]
                
            changed = True
            while changed:
                changed = False
                for i in range(len(selected) - 1, -1, -1):
                    v = selected[i]
                    other_mask = 0
                    for j in range(len(selected)):
                        if j != i:
                            other_mask |= masks[selected[j]]
                    if not (masks[v] & ~other_mask):
                        selected.pop(i)
                        changed = True
                        break
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_sol = selected[:]
                
        return sorted(best_sol)

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        valid_cliques = [c for c in cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        try:
            m = len(valid_cliques)
            c = np.ones(n)
            A = np.zeros((m, n))
            b = np.ones(m)
            
            for i, clique in enumerate(valid_cliques):
                for v in clique:
                    A[i, v] = 1
                    
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(n)
            
            res = milp(
                c=c,
                constraints=[LinearConstraint(A, lb=b)],
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0 and res.x is not None:
                sol = [int(v) for v in res.x if v > 0.5]
                # Verify
                is_valid = True
                sol_set = set(sol)
                for clique in valid_cliques:
                    if not any(v in sol_set for v in clique):
                        is_valid = False
                        break
                if is_valid:
                    return sorted(sol)
        except Exception:
            pass
            
        return solve_greedy(n, valid_cliques)
        
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
