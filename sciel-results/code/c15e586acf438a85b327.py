import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 120.0
    margin = 5.0 # Safety margin
    
    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    results = []
    
    for n, p, seed in instances:
        # 1. Generate Graph
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 2. Enumerate Maximal Cliques (Bron-Kerbosch with pivot)
        maximal_cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                best_u = -1
                max_count = -1
                for u in P:
                    count = len(P & adj[u])
                    if count > max_count:
                        max_count = count
                        best_u = u
                for u in X:
                    count = len(P & adj[u])
                    if count > max_count:
                        max_count = count
                        best_u = u
                
                if best_u != -1:
                    candidates = [v for v in P if v not in adj[best_u]]
                else:
                    candidates = list(P)
            else:
                candidates = list(P)
            
            for v in candidates:
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                bron_kerbosch(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
                
        P_initial = set(range(n))
        bron_kerbosch(set(), P_initial, set())
        
        # 3. Prepare Hitting Set Problem
        cliques = [list(c) for c in maximal_cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Precompute vertex to clique indices
        vertex_to_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        num_cliques = len(cliques)
        
        # Helper functions for ILS
        
        def get_cover_state(S):
            """Returns (cover_count, uncovered_set) for a given set S"""
            cover_count = [0] * num_cliques
            for v in S:
                for c_idx in vertex_to_cliques[v]:
                    cover_count[c_idx] += 1
            uncovered = set(i for i in range(num_cliques) if cover_count[i] == 0)
            return cover_count, uncovered

        def repair(S):
            """Greedy repair: add vertices to S until all cliques covered. Returns (S, cover_count, uncovered)"""
            S = set(S)
            cover_count, uncovered = get_cover_state(S)
            
            while uncovered:
                # Find vertex with max gain
                best_v = -1
                max_gain = -1
                # We can optimize this by only checking vertices not in S
                # But n is small, so checking all is fine
                for v in range(n):
                    if v in S:
                        continue
                    gain = 0
                    for c_idx in vertex_to_cliques[v]:
                        if c_idx in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_v = v
                
                if best_v == -1:
                    break # Should not happen if graph is connected enough, but safe
                    
                S.add(best_v)
                for c_idx in vertex_to_cliques[best_v]:
                    if c_idx in uncovered:
                        uncovered.remove(c_idx)
                    cover_count[c_idx] += 1
            
            return S, cover_count, uncovered

        def local_search(S, cover_count, uncovered):
            """Remove redundant vertices. Returns (S, cover_count, uncovered)"""
            S = set(S)
            # We need to check if a vertex is critical.
            # A vertex v is critical if there exists a clique C in vertex_to_cliques[v]
            # such that cover_count[C] == 1 (only v covers it).
            # If not critical, remove it.
            
            changed = True
            while changed:
                changed = False
                # Iterate over a copy of S to allow removal
                for v in list(S):
                    is_critical = False
                    for c_idx in vertex_to_cliques[v]:
                        if cover_count[c_idx] == 1:
                            is_critical = True
                            break
                    if not is_critical:
                        # Remove v
                        S.remove(v)
                        for c_idx in vertex_to_cliques[v]:
                            cover_count[c_idx] -= 1
                            if cover_count[c_idx] == 0:
                                uncovered.add(c_idx)
                        changed = True
                        # Restart loop to check new redundancies
                        break
            
            return S, cover_count, uncovered

        # 4. Greedy Initialization
        S_init = set()
        cover_count_init, uncovered_init = get_cover_state(S_init)
        S_init, cover_count_init, uncovered_init = repair(S_init)
        S_init, cover_count_init, uncovered_init = local_search(S_init, cover_count_init, uncovered_init)
        
        best_S = set(S_init)
        best_size = len(best_S)
        
        # 5. Iterated Local Search
        rng_ils = random.Random(seed + 500)
        
        # Time check helper
        def time_left():
            return time_limit - (time.time() - start_time) - margin - (len(instances) - instances.index((n,p,seed))) * 1.0
        
        # ILS Parameters
        # Perturbation strength: number of vertices to remove
        # Start with small, maybe increase? Or fixed.
        
        while time_left() > 0.1:
            # Perturb: Remove k random vertices from best_S
            # k can be adaptive. Let's try k=2 to 5.
            k = rng_ils.randint(2, 5)
            
            S_perturbed = set(best_S)
            # Remove k random vertices
            for _ in range(k):
                if S_perturbed:
                    # Random choice
                    v = rng_ils.choice(list(S_perturbed))
                    S_perturbed.remove(v)
            
            # Repair
            S_repaired, cc, unc = repair(S_perturbed)
            
            # Local Search
            S_final, cc, unc = local_search(S_repaired, cc, unc)
            
            if len(S_final) < best_size:
                best_size = len(S_final)
                best_S = S_final
            # Optional: Accept worse solutions with some probability? 
            # Usually ILS on best is good enough for these sizes.
            # If we get stuck, we might need to accept worse solutions.
            # For now, stick to best-improvement ILS.
            
        results.append(sorted(best_S))
        
    return results
