"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(edges, n, k):
        """
        Improved clique partition for split graphs.
        
        Strategy:
        1. Build the K clique (covers all edges within K).
        2. For cross edges (u in K, v in I), group them into larger cliques.
           A clique {u1, u2, ..., um, v} is valid iff all ui are pairwise adjacent
           in K (they are, since K is a clique) AND v is adjacent to all ui.
           So for each v in I, we can form cliques {S, v} where S is a subset
           of N_K(v) (neighbors of v in K).
        
        The key insight: instead of making {u, v} for each cross edge, we can
        group multiple K-neighbors of v into one clique with v.
        
        To minimize total cliques:
        - For each v in I, we need to cover all edges (u, v) where u in N_K(v).
        - Each clique {S, v} covers |S| such edges.
        - We want to maximize |S| to minimize the number of cliques per v.
        - Since K is a clique, any subset S of N_K(v) works.
        - So the best is: one clique {N_K(v), v} if N_K(v) is non-empty.
        - But wait: we also need every edge in K to be in exactly one clique.
          The K clique covers all K-internal edges. If we include K vertices
          in cross cliques, those K vertices appear in multiple cliques, but
          the edges within K are still only in the K clique. That's fine.
        
        Actually, let's reconsider. The constraint is:
        - Every listed set must be a clique.
        - Every edge must lie in exactly one listed clique.
        
        If I make the full K as one clique, and then for each v in I, make
        {N_K(v), v} as a clique, then:
        - Edges within K: covered by the K clique. ✓
        - Cross edges (u, v): covered by {N_K(v), v} since u in N_K(v). ✓
        - No edge is in two cliques because K-internal edges are only in the
          K clique, and cross edges are only in the corresponding {N_K(v), v}.
        
        But wait: what if N_K(v) has size 1? Then {N_K(v), v} is just {u, v},
        which is the same as the seed approach.
        
        What if N_K(v) is empty? Then v has no cross edges, so no clique needed.
        
        So the algorithm is:
        1. Add K as a clique.
        2. For each v in I (vertices k to n-1):
           - Compute N_K(v) = {u in [0, k) : (u, v) is an edge}
           - If |N_K(v)| >= 2, add {N_K(v), v} as a clique.
           - If |N_K(v)| == 1, add {u, v} as a clique (same as seed).
        
        This should reduce the number of cliques when vertices in I have many
        neighbors in K.
        """
        cliques = []
        
        # 1. The clique K
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Build adjacency for cross edges
        # cross_neighbors[v] = list of u in [0, k) such that (u, v) is an edge
        cross_neighbors = {v: [] for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                cross_neighbors[v].append(u)
        
        # 2. For each v in I, add a clique with its K-neighbors
        for v in range(k, n):
            neighbors = cross_neighbors[v]
            if len(neighbors) >= 2:
                cliques.append(neighbors + [v])
            elif len(neighbors) == 1:
                cliques.append([neighbors[0], v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        
        # Ensure types are correct: all vertices are Python int
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(v) for v in c])
        results.append(final_cliques)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return results
# EVOLVE-BLOCK-END
