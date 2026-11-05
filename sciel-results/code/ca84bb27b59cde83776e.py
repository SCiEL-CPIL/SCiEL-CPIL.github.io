"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivoting."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X with max |P ∩ N(u)|
            P_union_X = P | X
            if P_union_X:
                # Pick pivot with maximum degree in P
                best_u = max(P_union_X, key=lambda u: len(P & adj[u]))
                candidates = P - adj[best_u]
            else:
                candidates = P
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_hitting_set(adj, n, cliques, timeout=30.0):
        """Find minimum vertex set hitting all maximal cliques of size >= 2."""
        if not cliques:
            return set()
        
        # Build incidence: for each vertex, which cliques it covers
        # cliques is list of frozensets
        num_cliques = len(cliques)
        if num_cliques == 0:
            return set()
        
        # Greedy initialization
        uncovered = set(range(num_cliques))
        # For each vertex, compute which cliques it belongs to
        vertex_cliques = {v: [] for v in range(n)}
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(ci)
        
        solution = set()
        while uncovered:
            # Pick vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in solution:
                    continue
                count = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            solution.add(best_v)
            for ci in vertex_cliques[best_v]:
                if ci in uncovered:
                    uncovered.discard(ci)
        
        # Local search: try to remove vertices and re-add if needed
        # Try removing each vertex; if it causes uncovered cliques, try swapping
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 5:
            improved = False
            for v in list(solution):
                # Try removing v
                solution.discard(v)
                # Check if all cliques still covered
                new_uncovered = []
                for ci, c in enumerate(cliques):
                    if not (c & solution):
                        new_uncovered.append(ci)
                if not new_uncovered:
                    improved = True
                    continue
                # Try swapping: add a vertex that covers the most new_uncovered
                best_swap = None
                best_gain = -1
                for w in range(n):
                    if w in solution:
                        continue
                    gain = sum(1 for ci in new_uncovered if w in cliques[ci])
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = w
                if best_swap is not None and best_gain > 0:
                    solution.add(best_swap)
                    improved = True
                else:
                    # Can't fix by single swap, revert
                    solution.add(v)
        
        # Try adding vertices to cover any remaining (shouldn't happen)
        for ci, c in enumerate(cliques):
            if not (c & solution):
                # Pick a vertex from this clique
                for v in c:
                    solution.add(v)
                    break
        
        return solution

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if (time.time() - start_time) > DEADLINE - 10:
            # Not enough time, use greedy-only fallback
            adj = generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        
        adj = generate_graph(n, p, seed)
        
        # Enumerate maximal cliques
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        # Filter: only maximal cliques of size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Solve hitting set
        sol = solve_hitting_set(adj, n, cliques)
        results.append(sorted(sol))
    
    return results
# EVOLVE-BLOCK-END
