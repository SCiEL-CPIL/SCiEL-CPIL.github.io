import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin of 10s

    def instance(n, p, seed):
        rng = random.Random(seed)
        # The seed program uses: [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
        # This generates edges in a specific order. We must replicate this to get the same graph.
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, vertices, excluded, result):
        """
        Recursive Bron-Kerbosch with pivoting to find maximal cliques.
        adj: dict of vertex -> set of neighbors
        vertices: set of available vertices
        excluded: set of excluded vertices
        result: list to store maximal cliques
        """
        if not vertices and not excluded:
            # Found a maximal clique
            if len(result_clique) >= 2:
                result.append(tuple(sorted(result_clique)))
            return
        
        # Pivoting: choose a pivot u from vertices U excluded that maximizes |vertices ∩ adj[u]|
        # If vertices is empty, we just pick from excluded? No, standard BK picks from V ∪ X.
        # If V is empty, we can't pick a pivot from V. We can pick from X if X is not empty.
        # Actually, if V is empty, the clique is maximal. We handled that above.
        # So V is not empty.
        
        # Select pivot
        pivot_candidates = list(vertices | excluded)
        if not pivot_candidates:
            return
            
        # To speed up, we can just pick the first one or one with max degree in V
        # For small graphs, simple selection is fine.
        pivot = None
        max_overlap = -1
        # Optimization: Only consider pivots from vertices if possible? 
        # Standard BK: u in V ∪ X.
        # Let's just iterate.
        for u in pivot_candidates:
            overlap = len(vertices & adj.get(u, set()))
            if overlap > max_overlap:
                max_overlap = overlap
                pivot = u
        
        if pivot is None:
            return
            
        # Iterate over V \ N(u)
        for v in list(vertices - adj.get(pivot, set())):
            # Expand with v
            new_vertices = vertices & adj.get(v, set())
            new_excluded = excluded & adj.get(v, set())
            
            # Recursive call
            # We need to pass the current clique + v
            # To avoid copying the whole clique list every time, we can use a stack or pass a list
            # Since n is small (40), copying is cheap.
            
            # We modify a local copy of the current clique state
            # The function signature above doesn't pass the current clique. 
            # Let's restructure to pass current_clique as a list.
            
            # Actually, the standard recursive implementation usually passes the current set.
            # Let's rewrite BK to be cleaner.
            
            # This inner function approach is getting messy. Let's use a standard iterative or recursive BK that returns cliques.
            
            # I will re-implement BK outside this helper to be cleaner.
            pass

    def find_maximal_cliques(n, edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Ensure all vertices are in adj
        for i in range(n):
            if i not in adj:
                adj[i] = set()
                
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(tuple(sorted(R)))
                return
                
            # Pivoting
            # Choose u from P ∪ X that maximizes |P ∩ N(u)|
            candidates = P | X
            if not candidates:
                return
            
            # Pick pivot
            pivot = None
            max_count = -1
            for u in candidates:
                count = len(P & adj[u])
                if count > max_count:
                    max_count = count
                    pivot = u
            
            if pivot is None:
                return
            
            # Iterate over P \ N(pivot)
            # We need to be careful with set modifications
            for v in list(P - adj[pivot]):
                # R' = R ∪ {v}
                # P' = P ∩ N(v)
                # X' = X ∩ N(v)
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R_new, P_new, X_new)
                
                # Move v from P to X
                P.discard(v)
                X.add(v)
                
                # Time check
                if time.time() > deadline - 5.0:
                    # Abort if taking too long, return what we have
                    # This is a bit hacky, but prevents timeout
                    # We should probably structure this to raise an exception or use a flag
                    pass

        initial_P = set(range(n))
        bk(set(), initial_P, set())
        return cliques

    def solve_set_cover(n, cliques):
        if not cliques:
            return []
            
        # Map each vertex to the set of clique indices it covers
        vertex_cliques = defaultdict(list)
        for idx, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].append(idx)
        
        # Initialize coverage
        covered = [False] * len(cliques)
        solution = []
        
        # Greedy
        while True:
            uncovered = [i for i, c in enumerate(covered) if not c]
            if not uncovered:
                break
                
            best_vertex = None
            best_count = 0
            # For each vertex, count how many uncovered cliques it covers
            for v in range(n):
                if v in solution:
                    continue
                count = 0
                for idx in vertex_cliques[v]:
                    if not covered[idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_vertex = v
                    
            if best_vertex is None or best_count == 0:
                # Should not happen if cliques are non-empty and vertices exist
                break
                
            solution.append(best_vertex)
            for idx in vertex_cliques[best_vertex]:
                covered[idx] = True
                
            if time.time() > deadline - 10.0:
                break

        # Local Search: Removal and Swap
        # We try to remove a vertex, and if it breaks coverage, we try to fix it by adding the best vertex.
        # If the net change is negative (size decreases), we accept.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i, v in enumerate(list(solution)):
                # Try removing v
                covered_backup = covered[:]
                for idx in vertex_cliques[v]:
                    covered[idx] = False
                    
                # Check if still covered
                if all(covered):
                    # Safe removal
                    solution.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                else:
                    # We need to cover the gaps
                    uncovered = [idx for idx, c in enumerate(covered) if not c]
                    if not uncovered:
                        solution.pop(i)
                        improved = True
                        break
                        
                    # Find best replacement
                    best_rep = None
                    best_gain = 0
                    # We want to minimize the number of new vertices added.
                    # Ideally 1 vertex covers all uncovered.
                    # If not, we might need more.
                    # Simple heuristic: Find the vertex that covers the most uncovered cliques.
                    # Then add it. Then check if more are needed.
                    # This is essentially a local greedy repair.
                    
                    # Let's try to find a single vertex that covers ALL uncovered cliques.
                    found_single = False
                    for u in range(n):
                        if u == v or u in solution:
                            continue
                        covers_all = True
                        for idx in uncovered:
                            if idx not in vertex_cliques[u]:
                                covers_all = False
                                break
                        if covers_all:
                            best_rep = u
                            found_single = True
                            break
                            
                    if found_single:
                        # Swap v with best_rep
                        solution[i] = best_rep
                        # Update covered
                        for idx in vertex_cliques[best_rep]:
                            covered[idx] = True
                        # Since v was removed, and best_rep covers all previously covered by v?
                        # Not necessarily. We need to ensure best_rep covers everything v covered that is not covered by others.
                        # Actually, if we just swap, we need to re-verify coverage.
                        # But if best_rep covers all *uncovered* (which were uncovered by removing v), 
                        # and all other vertices are still there, then:
                        # 1. Cliques covered by others remain covered.
                        # 2. Cliques covered by v: 
                        #    - If they are in `uncovered`, they are now covered by best_rep.
                        #    - If they are NOT in `uncovered`, they were covered by others (so removing v didn't uncover them).
                        # So the swap is valid.
                        improved = True
                        break
                        
                    # If no single vertex works, try a greedy repair sequence?
                    # For simplicity and speed, we might skip multi-vertex swaps in this first pass.
                    # Or, we can just restore and move on.
                    covered[:] = covered_backup
                    continue

        return solution

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 2.0:
            # Emergency: return empty lists or best guess? 
            # Returning empty is invalid. 
            # We must return something valid. 
            # If we ran out of time, we return the current best or a trivial valid solution.
            # A trivial valid solution is all vertices that are in any edge.
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue

        edges = instance(n, p, seed)
        cliques = find_maximal_cliques(n, edges)
        sol = solve_set_cover(n, cliques)
        
        # Verify validity
        # A solution is valid if every maximal clique of size >= 2 has at least one vertex in sol.
        sol_set = set(sol)
        valid = True
        for c in cliques:
            if not any(v in sol_set for v in c):
                valid = False
                break
                
        if not valid:
            # Fallback to trivial solution
            touched = {x for e in edges for x in e}
            sol = sorted(touched)
            
        results.append(sorted(sol))
        
    return results
