import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting.
        adj: list of sets, adj[i] = set of neighbors of i
        Returns list of frozensets (each a maximal clique of size >= 2)
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u in P ∪ X to minimize branching
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            
            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])
            
            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_hit(cliques, n):
        """
        Greedy set cover: repeatedly pick vertex covering most uncovered cliques.
        cliques: list of frozensets
        Returns a set of vertices
        """
        if not cliques:
            return set()
        
        # For each vertex, count how many cliques it covers
        vertex_clique_count = [0] * n
        for c in cliques:
            for v in c:
                vertex_clique_count[v] += 1
        
        uncovered = set(range(len(cliques)))
        result = set()
        
        while uncovered:
            best_v = -1
            best_covered = 0
            for v in range(n):
                if v in result:
                    continue
                covered = 0
                for i in uncovered:
                    if v in cliques[i]:
                        covered += 1
                if covered > best_covered:
                    best_covered = covered
                    best_v = v
            if best_v == -1 or best_covered == 0:
                break
            result.add(best_v)
            uncovered = {i for i in uncovered if best_v not in cliques[i]}
        
        return result

    def try_improve(result, cliques, n, max_iter=300):
        """
        Local search: try removing a vertex and re-adding if needed, or swapping.
        Includes 2-for-1 swaps for stronger reduction.
        """
        result = set(result)
        if not result:
            return result
        
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            # Try removing each vertex
            for v in list(result):
                candidate = result - {v}
                valid = True
                for c in cliques:
                    if not (c & candidate):
                        valid = False
                        break
                if valid:
                    result = candidate
                    improved = True
                    break
        
        # Try 1-for-1 swaps
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            best_swap = None
            best_size = len(result)
            for v in result:
                candidate = result - {v}
                uncovered_cliques = [c for c in cliques if not (c & candidate)]
                if not uncovered_cliques:
                    continue
                for w in range(n):
                    if w in candidate:
                        continue
                    if all(w in c for c in uncovered_cliques):
                        new_size = len(candidate) + 1
                        if new_size < best_size:
                            best_size = new_size
                            best_swap = (v, w)
            if best_swap:
                v, w = best_swap
                result = result - {v} | {w}
                improved = True
                for v2 in list(result):
                    candidate = result - {v2}
                    valid = all(c & candidate for c in cliques)
                    if valid:
                        result = candidate
                        break
        
        # Try 2-for-1 swaps: remove two vertices, add one
        for _ in range(30):
            if time.time() - start_time > DEADLINE:
                break
            best_2for1 = None
            best_2for1_size = len(result)
            result_list = list(result)
            for i in range(len(result_list)):
                for j in range(i + 1, len(result_list)):
                    v1, v2 = result_list[i], result_list[j]
                    candidate = result - {v1} - {v2}
                    uncovered = [c for c in cliques if not (c & candidate)]
                    if not uncovered:
                        # Can remove both, no need to add
                        best_2for1 = (v1, v2, None)
                        best_2for1_size = len(candidate)
                        break
                    for w in range(n):
                        if w in candidate:
                            continue
                        if all(w in c for c in uncovered):
                            new_size = len(candidate) + 1
                            if new_size < best_2for1_size:
                                best_2for1_size = new_size
                                best_2for1 = (v1, v2, w)
                if best_2for1 is not None and best_2for1[2] is None:
                    break
                if time.time() - start_time > DEADLINE:
                    return result
            
            if best_2for1:
                v1, v2, w = best_2for1
                if w is None:
                    result = result - {v1} - {v2}
                else:
                    result = result - {v1} - {v2} | {w}
                improved = True
                # After 2-for-1, try to remove redundant vertices
                for v3 in list(result):
                    candidate = result - {v3}
                    if all(c & candidate for c in cliques):
                        result = candidate
                        break
        
        return result

    def branch_and_bound(cliques, n, initial_solution, deadline):
        """
        Branch-and-bound exact solver for minimum clique transversal.
        Uses a greedy upper bound and a disjoint-clique lower bound.
        """
        if not cliques:
            return set()
        
        # Precompute for each vertex, the set of clique indices it covers
        vertex_cliques = [set() for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(i)
        
        all_clique_indices = set(range(len(cliques)))
        best_solution = set(initial_solution)
        best_size = len(best_solution)
        
        # Lower bound: find a set of pairwise disjoint cliques
        # The minimum transversal must have at least one vertex from each disjoint clique
        def compute_lower_bound(uncovered_cliques):
            """Find maximum set of pairwise disjoint cliques from uncovered."""
            if not uncovered_cliques:
                return 0
            # Greedy: pick clique with smallest size first
            remaining = set(uncovered_cliques)
            disjoint_count = 0
            while remaining:
                # Pick the smallest clique
                min_clique = min(remaining, key=lambda i: len(cliques[i]))
                disjoint_count += 1
                # Remove all cliques that intersect with min_clique
                min_clique_set = cliques[min_clique]
                remaining = {i for i in remaining if not (cliques[i] & min_clique_set)}
            return disjoint_count
        
        # Branch and bound
        def bb(selected, uncovered_cliques, depth=0):
            nonlocal best_solution, best_size
            
            if time.time() - start_time > deadline:
                return
            
            if not uncovered_cliques:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return
            
            # Lower bound
            lb = compute_lower_bound(uncovered_cliques)
            if len(selected) + lb >= best_size:
                return
            
            # Pick a clique to branch on
            # Choose the clique with the fewest candidate vertices
            min_clique_idx = min(uncovered_cliques, key=lambda i: len(cliques[i]))
            clique = cliques[min_clique_idx]
            
            # Branch: pick a vertex from this clique
            for v in sorted(clique, key=lambda x: len(vertex_cliques[x])):
                new_selected = set(selected)
                new_selected.add(v)
                
                # Update uncovered cliques
                new_uncovered = set()
                for i in uncovered_cliques:
                    if v not in cliques[i]:
                        new_uncovered.add(i)
                
                bb(new_selected, new_uncovered, depth + 1)
                
                if time.time() - start_time > deadline:
                    return
            
            # Also branch on NOT picking any vertex from this clique? No, we must pick one.
            # Actually, we're forcing a choice from this clique, which is correct.
        
        # Start branch and bound
        # First, remove any redundant vertices from initial solution
        cleaned = set()
        for v in initial_solution:
            candidate = cleaned | {v}
            # Check if v is needed
            uncovered_without_v = set()
            for i, c in enumerate(cliques):
                if not (c & cleaned):
                    uncovered_without_v.add(i)
            if not uncovered_without_v:
                cleaned = candidate
            else:
                # Check if v covers all uncovered
                if all(v in cliques[i] for i in uncovered_without_v):
                    cleaned = candidate
        
        if len(cleaned) < best_size:
            best_size = len(cleaned)
            best_solution = cleaned
        
        # Run B&B with a time limit
        bb(set(), all_clique_indices)
        
        return best_solution

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy hit set
        hit = greedy_hit(cliques, n)
        
        # Local improvement
        hit = try_improve(hit, cliques, n)
        
        # Try branch-and-bound if time permits
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time > 5.0 and len(cliques) < 500:
            # Run B&B with a time limit
            bb_deadline = time.time() + min(remaining_time - 2.0, 30.0)
            hit_bb = branch_and_bound(cliques, n, hit, bb_deadline)
            if len(hit_bb) < len(hit):
                hit = hit_bb
        
        results.append(sorted(hit))
    
    return results
