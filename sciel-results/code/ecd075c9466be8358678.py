import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        import random
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
            # Pick vertex with max |P ∩ adj[u]|
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
            # Remove all cliques containing best_v from uncovered
            uncovered = {i for i in uncovered if best_v not in cliques[i]}
        
        return result

    def try_improve(result, cliques, n, max_iter=200):
        """
        Local search: try removing a vertex and re-adding if needed, or swapping.
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
                # Check if candidate still hits all cliques
                valid = True
                for c in cliques:
                    if not (c & candidate):
                        valid = False
                        break
                if valid:
                    result = candidate
                    improved = True
                    break
        
        # Try adding vertices to enable removals (swap heuristic)
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            best_swap = None
            best_size = len(result)
            for v in result:
                candidate = result - {v}
                # Find which cliques are now uncovered
                uncovered_cliques = [c for c in cliques if not (c & candidate)]
                if not uncovered_cliques:
                    continue
                # Try to add one vertex that covers all uncovered cliques
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
                # Continue improving
                for v2 in list(result):
                    candidate = result - {v2}
                    valid = all(c & candidate for c in cliques)
                    if valid:
                        result = candidate
                        break
        
        return result

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
        
        results.append(sorted(hit))
    
    return results
