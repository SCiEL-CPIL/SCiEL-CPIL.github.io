import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            P_X = P.union(X)
            if not P_X:
                return
            
            max_deg = -1
            pivot = None
            for u in P_X:
                deg = len(P.intersection(adj[u]))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P - adj[pivot]):
                bk(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
            
        m = len(cliques)
        
        # Precompute mappings
        vertex_to_cliques = [[] for _ in range(n)]
        clique_to_vertices = [set(c) for c in cliques]
        
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        best_solution = None
        best_size = float('inf')
        
        # Number of restarts
        restarts = 500
        
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            uncovered = set(range(m))
            solution = set()
            
            # Greedy Phase
            # We can speed this up slightly by precomputing clique counts for vertices
            # but the current O(n * m) per step is okay for n=40, m~100-200
            while uncovered:
                best_v = -1
                best_count = -1
                max_count = 0
                candidates = []
                for v in range(n):
                    if v in solution:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > max_count:
                        max_count = count
                        candidates = [v]
                    elif count == max_count and count > 0:
                        candidates.append(v)
                        
                if not candidates:
                    break # Should not happen
                    
                v = random.choice(candidates)
                solution.add(v)
                
                for ci in vertex_to_cliques[v]:
                    uncovered.discard(ci)
                    
            # Local Search Phase
            # Maintain clique_coverage_count for fast redundancy checks
            # coverage[ci] = number of vertices in solution that are in clique ci
            
            improved = True
            while improved:
                improved = False
                
                # 1. Redundancy Removal using coverage counts
                # Calculate coverage
                coverage = [0] * m
                for v in solution:
                    for ci in vertex_to_cliques[v]:
                        coverage[ci] += 1
                
                verts_to_check = list(solution)
                random.shuffle(verts_to_check)
                for v in verts_to_check:
                    if v not in solution:
                        continue
                    # Check if v is redundant
                    redundant = True
                    for ci in vertex_to_cliques[v]:
                        if coverage[ci] <= 1:
                            redundant = False
                            break
                    if redundant:
                        solution.remove(v)
                        # Update coverage
                        for ci in vertex_to_cliques[v]:
                            coverage[ci] -= 1
                        improved = True
                        break # Restart check to catch cascading redundancies
                
                if improved:
                    continue # Restart the while loop to check redundancies again
                
                # 2. 2-for-1 Swap
                # We want to find v1, v2 in solution and u not in solution
                # such that removing v1, v2 and adding u still covers all cliques.
                # This implies u must cover all cliques that were ONLY covered by v1 or v2.
                
                sol_list = list(solution)
                random.shuffle(sol_list)
                found_improvement = False
                for i in range(len(sol_list)):
                    if time.time() > deadline - 5.0:
                        break
                    v1 = sol_list[i]
                    if v1 not in solution:
                        continue
                    for j in range(i + 1, len(sol_list)):
                        v2 = sol_list[j]
                        if v2 not in solution:
                            continue
                        
                        # Find cliques that become uncovered if we remove v1 and v2
                        # A clique is uncovered if coverage[ci] == 1 and (v1 in clique or v2 in clique)
                        # Or coverage[ci] == 2 and (v1 in clique and v2 in clique) and no other vertex in clique is in solution?
                        # Actually, simpler:
                        # Temporarily "remove" v1 and v2 from coverage logic.
                        
                        critical_cliques = []
                        for ci in vertex_to_cliques[v1]:
                            if coverage[ci] == 1:
                                # Only v1 covers this
                                critical_cliques.append(ci)
                            elif coverage[ci] == 2:
                                # v1 and one other cover this. If that other is v2, it becomes uncovered.
                                # Check if v2 is in clique ci
                                if v2 in clique_to_vertices[ci]:
                                    critical_cliques.append(ci)
                        
                        # We must also check cliques where v2 is the sole coverer, 
                        # but those are handled if we iterate v1's cliques? 
                        # No. If coverage[ci]==1 and v2 in clique, but v1 not in clique, 
                        # then removing v2 uncovers it.
                        # My loop above only checks v1's cliques.
                        # I need to check v2's cliques too for cases where v1 is not in ci.
                        
                        # Let's refine:
                        # A clique ci is critical if:
                        # 1. v1 in ci and coverage[ci] == 1
                        # 2. v2 in ci and coverage[ci] == 1
                        # 3. v1 in ci and v2 in ci and coverage[ci] == 2
                        
                        # Let's re-calculate critical_cliques properly.
                        critical_cliques = set()
                        
                        # Check v1's cliques
                        for ci in vertex_to_cliques[v1]:
                            if coverage[ci] == 1:
                                critical_cliques.add(ci)
                            elif coverage[ci] == 2 and v2 in clique_to_vertices[ci]:
                                critical_cliques.add(ci)
                                
                        # Check v2's cliques (for cases where v1 not in ci)
                        for ci in vertex_to_cliques[v2]:
                            if coverage[ci] == 1:
                                critical_cliques.add(ci)
                            # The case coverage==2 and v1 in ci is already covered above
                            # because if v1 in ci and v2 in ci and coverage==2, it's added in v1's loop.
                            # If v1 not in ci, coverage==1 is the only critical case for v2.
                            # Wait, if coverage[ci]==2, v2 in ci, v1 NOT in ci.
                            # Then removing v2 leaves 1 cover. Not uncovered.
                            # So only coverage==1 matters for single vertex removal.
                            # And coverage==2 matters only if BOTH are removed.
                            
                        if not critical_cliques:
                            # Removing v1 and v2 uncovers nothing. They are jointly redundant.
                            solution.remove(v1)
                            solution.remove(v2)
                            improved = True
                            found_improvement = True
                            break
                        
                        # We need a vertex u not in solution that is in ALL critical_cliques
                        # And u must not be v1 or v2 (which are being removed, so effectively not in solution)
                        
                        # Candidate vertices must be in the intersection of all critical cliques
                        # Or, more efficiently, count how many critical cliques each candidate covers.
                        
                        # Initialize counts for candidates?
                        # Since n is small, we can just iterate u in range(n)
                        
                        best_u = None
                        for u in range(n):
                            if u in solution:
                                continue
                            # Check if u covers all critical cliques
                            covers_all = True
                            for ci in critical_cliques:
                                if u not in clique_to_vertices[ci]:
                                    covers_all = False
                                    break
                            if covers_all:
                                best_u = u
                                break
                                
                        if best_u is not None:
                            solution.remove(v1)
                            solution.remove(v2)
                            solution.add(best_u)
                            improved = True
                            found_improvement = True
                            break
                        
                    if found_improvement:
                        break
                        
                if found_improvement:
                    continue # Restart local search loop

            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = set(solution)
                
        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10.0:
            # Fallback: return all touched vertices
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u,v)}
            out.append(sorted(touched))
        else:
            out.append(solve_instance(n, p, seed))
            
    return out
