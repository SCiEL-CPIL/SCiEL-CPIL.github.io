# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex with max |P ∩ N(v)|
            if P | X:
                pivot = max(P | X, key=lambda v: len(P & adj[v]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        expand(frozenset(), set(range(n)), set())
        return maximal_cliques
    
    def solve_instance(cliques, n, time_limit):
        if not cliques:
            return set()
        
        num_cliques = len(cliques)
        clique_list = list(cliques)
        
        # For each vertex, which cliques does it cover?
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(clique_list):
            for v in clique:
                vertex_cliques[v].add(i)
        
        # Precompute: for each clique, how many vertices cover it?
        clique_cover_count = [0] * num_cliques
        for v in range(n):
            for i in vertex_cliques[v]:
                clique_cover_count[i] += 1
        
        # Initial Weights: Inverse of cover count
        clique_weights = [1.0 / max(1, c) for c in clique_cover_count]
        
        # Greedy Selection with Weighted Marginal Utility
        selected = set()
        covered = set()
        
        while len(covered) < num_cliques:
            if time.time() > time_limit:
                break
                
            best_v = None
            best_score = -1
            
            for v in range(n):
                if v in selected:
                    continue
                
                # Calculate marginal weighted utility
                score = 0.0
                for i in vertex_cliques[v]:
                    if i not in covered:
                        score += clique_weights[i]
                
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v is None or best_score <= 0:
                break
                
            selected.add(best_v)
            for i in vertex_cliques[best_v]:
                covered.add(i)
        
        # Local Search Polish
        # Build coverage counts for the current selection
        cov_count = [0] * num_cliques
        for v in selected:
            for i in vertex_cliques[v]:
                cov_count[i] += 1
        
        # Check if all are covered
        for i in range(num_cliques):
            if cov_count[i] == 0:
                for v in clique_list[i]:
                    if v not in selected:
                        selected.add(v)
                        for j in vertex_cliques[v]:
                            cov_count[j] += 1
                        break
        
        # Local Search Loop
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try removing a vertex
            for v in list(selected):
                is_critical = False
                for i in vertex_cliques[v]:
                    if cov_count[i] == 1:
                        is_critical = True
                        break
                
                if not is_critical:
                    selected.remove(v)
                    for i in vertex_cliques[v]:
                        cov_count[i] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 2-for-1 swaps: remove v1, v2, add v3
            unique_covers = {}
            for v in selected:
                uc = set()
                for i in vertex_cliques[v]:
                    if cov_count[i] == 1:
                        uc.add(i)
                unique_covers[v] = uc
            
            found_2_for_1 = False
            sel_list = list(selected)
            for idx1 in range(len(sel_list)):
                if found_2_for_1:
                    break
                v1 = sel_list[idx1]
                uc1 = unique_covers[v1]
                
                for idx2 in range(idx1 + 1, len(sel_list)):
                    v2 = sel_list[idx2]
                    uc2 = unique_covers[v2]
                    
                    combined_unique = uc1 | uc2
                    
                    if not combined_unique:
                        continue
                    
                    for v3 in range(n):
                        if v3 in selected:
                            continue
                        
                        covers_all = True
                        for c in combined_unique:
                            if c not in vertex_cliques[v3]:
                                covers_all = False
                                break
                        
                        if covers_all:
                            selected.remove(v1)
                            selected.remove(v2)
                            selected.add(v3)
                            
                            for i in vertex_cliques[v1]:
                                cov_count[i] -= 1
                            for i in vertex_cliques[v2]:
                                cov_count[i] -= 1
                            for i in vertex_cliques[v3]:
                                cov_count[i] += 1
                            
                            found_2_for_1 = True
                            break
                    if found_2_for_1:
                        break
                if found_2_for_1:
                    break
            
            if found_2_for_1:
                improved = True
                continue
            
            # 3. Try 1-for-1 swaps to enable future improvements
            # Remove v1, add v2 if it covers all uniquely covered cliques by v1
            # and v2 covers at least one additional clique that is critically covered
            found_swap = False
            for v1 in list(selected):
                if found_swap:
                    break
                uc1 = unique_covers.get(v1, set())
                if not uc1:
                    continue
                
                for v2 in range(n):
                    if v2 in selected:
                        continue
                    
                    # Check if v2 covers all uniquely covered cliques by v1
                    covers_all = True
                    for c in uc1:
                        if c not in vertex_cliques[v2]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Check if v2 adds coverage to some critically covered clique
                    # that is not in uc1 (i.e., covered by v1 and at least one other)
                    # This is a heuristic: we just accept the swap if it doesn't increase size
                    # and might enable future removals. But to be safe, let's only do
                    # swaps that strictly improve or maintain feasibility.
                    
                    # Actually, for set cover minimization, a 1-for-1 swap that keeps
                    # the same size is only useful if it enables a subsequent removal.
                    # Let's try the swap and see if it helps.
                    
                    # Tentative swap
                    old_v1_cliques = vertex_cliques[v1]
                    new_v2_cliques = vertex_cliques[v2]
                    
                    # Check if after removing v1 and adding v2, all cliques are still covered
                    valid = True
                    for i in old_v1_cliques:
                        if cov_count[i] == 1 and i not in new_v2_cliques:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        selected.remove(v1)
                        selected.add(v2)
                        
                        for i in old_v1_cliques:
                            cov_count[i] -= 1
                        for i in new_v2_cliques:
                            cov_count[i] += 1
                        
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if found_swap:
                improved = True
                continue
        
        return selected

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            results.append(list(range(n)))
            continue
        
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        remaining_instances = 3 - len(results)
        time_limit = min(40.0, (deadline - time.time()) / remaining_instances - 1)
        
        selected = solve_instance(cliques, n, time_limit)
        
        # Verify coverage
        covered_cliques = set()
        for clique in cliques:
            for v in selected:
                if v in clique:
                    covered_cliques.add(clique)
                    break
        
        if len(covered_cliques) != len(cliques):
            for clique in cliques:
                if clique not in covered_cliques:
                    for v in clique:
                        if v not in selected:
                            selected.add(v)
                            break
        
        results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
