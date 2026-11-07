"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    """
    Construct clique partitions for three split graphs.
    """
    start_time = time.time()
    DEADLINE = 120.0  # 150s wall-clock limit with margin

    def instance(n, k, p, seed):
        """
        Generate a split graph with a clique of size k and an independent set of size n-k.
        Cross edges are included with probability p.
        """
        rng = Random(seed)
        # Clique edges among first k vertices
        clique_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        # Cross edges between clique and independent set
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return sorted(clique_edges + cross_edges)

    def greedy_clique_partition(edges, n, k):
        """
        Find a clique partition using a greedy approach.
        Split graphs have a specific structure that can be exploited.
        """
        if not edges:
            return []
        
        # Build adjacency sets for faster lookup
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
        
        # For split graphs, the clique C = {0, ..., k-1} is a maximum clique.
        # We can use this to our advantage.
        
        # Strategy: 
        # 1. The clique C itself is one clique.
        # 2. For each vertex v in the independent set, if it has edges to C,
        #    we can form cliques {v} ∪ (neighbors of v in C) if those neighbors form a clique in C.
        #    But since C is a clique, any subset of C is also a clique.
        # 3. However, we need each edge to be in exactly one clique.
        
        # Better approach for split graphs:
        # - The clique C can be partitioned into smaller cliques.
        # - Each independent set vertex v can be added to a clique containing some of its neighbors in C.
        
        # Let's use a different strategy:
        # Since C is a clique, we can try to form cliques by taking subsets of C and adding independent set vertices.
        
        # First, let's see which edges we need to cover.
        # All edges within C must be covered.
        # Edges between C and I must be covered.
        
        # A simple valid partition:
        # - {0, 1, ..., k-1} covers all C-C edges.
        # - For each v in I with neighbors N(v) in C, if N(v) is non-empty, 
        #   we can't just add {v} ∪ N(v) because that would double-cover edges in N(v).
        
        # Actually, the constraint is that each edge must be in exactly one clique.
        # So we need a partition of the edge set into cliques.
        
        # Let me think differently. In a split graph:
        # - C is a clique
        # - I is an independent set
        # - Edges only exist within C and between C and I
        
        # A valid clique partition can be:
        # 1. {C} as one clique (covers all C-C edges)
        # 2. For each v in I, if v has neighbors in C, we need to cover edges (u,v) for u in N(v).
        #    But we can't just make {u,v} for each edge because that's not minimal.
        
        # Actually, we can make cliques of the form {v} ∪ S where S ⊆ N(v) and S is a clique (which it is since C is a clique).
        # But we need to partition the cross edges.
        
        # Let's use a greedy matching-like approach:
        # - Start with C as a clique.
        # - For each v in I, if v has neighbors, we can remove some neighbors from C's clique and form {v} ∪ S.
        
        # This is getting complex. Let's use a simpler valid approach first:
        # - {C} for all C-C edges
        # - For each edge (u,v) with u in C, v in I, make {u,v} a clique.
        
        # This is valid but not minimal. Let's try to do better.
        
        # Better: Group independent set vertices that share neighbors in C.
        
        # For now, let's implement a basic valid solution:
        cliques = []
        
        # The main clique
        if k >= 2:
            cliques.append(list(range(k)))
        
        covered_edges = set()
        for u in range(k):
            for v in adj[u]:
                if v >= k:  # v is in independent set
                    e = (u, v)
                    if e not in covered_edges:
                        cliques.append([u, v])
                        covered_edges.add(e)
        
        # Filter to only cliques with at least 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        
        # Verify: each edge is in exactly one clique
        # (This is guaranteed by construction above)
        
        return cliques

    # Generate instances and compute partitions
    specs = [(30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)]
    result = []
    
    for spec in specs:
        if time.time() > DEADLINE:
            # Return what we have so far (shouldn't happen for small graphs)
            break
        edges = instance(*spec)
        n, k, p, seed = spec
        cliques = greedy_clique_partition(edges, n, k)
        # Ensure all vertices are Python int
        cliques = [[int(v) for v in clique] for clique in cliques]
        result.append(cliques)
    
    # If we didn't finish all instances, fill with empty lists
    while len(result) < 3:
        result.append([])
    
    return result

# EVOLVE-BLOCK-END
