"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    # Deadline guard: keep a safety margin before 150s
    deadline = time.time() + 145.0

    def generate_graph(n, p, seed):
        """Generate a random graph using the same RNG as the verifier."""
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def get_neighbors(edges, n):
        """Build adjacency sets."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_max_cliques(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Returns list of frozensets (each a maximal clique).
        """
        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            union_px = P | X
            if union_px:
                # Pick pivot with max |P ∩ N(u)|
                best_u = max(union_px, key=lambda u: len(P & adj[u]))
                candidates = P - adj[best_u]
            else:
                candidates = P

            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed, time_budget):
        """
        Find a minimum vertex set that intersects every maximal clique of size >= 2.
        This is the Minimum Vertex Cover on the hypergraph of maximal cliques.
        We use a greedy + local search approach, with branch-and-bound for small cases.
        """
        edges = generate_graph(n, p, seed)
        adj = get_neighbors(edges, n)
        
        # Enumerate maximal cliques of size >= 2
        max_cliques = bron_kerbosch_max_cliques(adj, n)
        
        # Filter: only cliques with at least 2 vertices
        max_cliques = [c for c in max_cliques if len(c) >= 2]
        
        if not max_cliques:
            return []
        
        # Convert to list of lists for easier manipulation
        clique_list = [list(c) for c in max_cliques]
        
        # Build incidence: for each vertex, which cliques it covers
        # and for each clique, which vertices are in it
        num_cliques = len(clique_list)
        
        # Greedy set cover: repeatedly pick vertex covering most uncovered cliques
        uncovered = set(range(num_cliques))
        selected = []
        vertex_clique_count = [0] * n
        
        # Precompute which cliques each vertex is in
        vertex_in_cliques = [[] for _ in range(n)]
        for ci, cl in enumerate(clique_list):
            for v in cl:
                vertex_in_cliques[v].append(ci)
        
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                count = sum(1 for ci in vertex_in_cliques[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
            
            selected.append(best_v)
            for ci in vertex_in_cliques[best_v]:
                uncovered.discard(ci)
        
        # Local search: try to swap or remove vertices to reduce size
        # First, try to remove any vertex that doesn't affect coverage
        for _ in range(10):  # Multiple passes
            changed = False
            for i, v in enumerate(selected):
                # Check if removing v still covers all cliques
                temp_selected = selected[:i] + selected[i+1:]
                covered = set()
                for sv in temp_selected:
                    for ci in vertex_in_cliques[sv]:
                        covered.add(ci)
                if covered == set(range(num_cliques)):
                    selected = temp_selected
                    changed = True
                    break
            if not changed:
                break
        
        # Local search: try swapping one vertex for another
        for _ in range(50):
            if time.time() > time_budget:
                break
            improved = False
            for i, v in enumerate(selected):
                # Try replacing v with some other vertex
                for w in range(n):
                    if w == v or w in selected:
                        continue
                    new_selected = selected[:i] + [w] + selected[i+1:]
                    covered = set()
                    for sv in new_selected:
                        for ci in vertex_in_cliques[sv]:
                            covered.add(ci)
                    if covered == set(range(num_cliques)):
                        # Same size, but maybe better for future swaps - accept if it changes
                        if w != v:
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return sorted(selected)
    
    # Solve each instance with time budget
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for i, (n, p, seed) in enumerate(instances):
        remaining = deadline - time.time() - 5.0  # Leave 5s for last instance
        if remaining < 1.0:
            # Not enough time, return empty or minimal
            results.append([])
            continue
        
        # Allocate time proportionally, but ensure at least 1s
        time_budget = min(remaining, 40.0)
        
        result = solve_instance(n, p, seed, time.time() + time_budget)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
