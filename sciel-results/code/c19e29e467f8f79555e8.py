"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from collections import defaultdict

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)

        # For each edge, find the maximal clique containing it
        # In a split graph, the clique on the first k vertices is a maximal clique.
        # Each vertex v >= k is adjacent to some subset of [0, k).
        # The maximal clique containing edge (u, v) where v >= k is {u, v} union (common neighbors of u and v in [0, k)).
        # But wait, we need a partition of edges into cliques.
        # A simple approach: for each vertex v >= k, the edges incident to v form a star.
        # But we need cliques.
        # Let's use the fact that split graphs are chordal.
        # A chordal graph has a perfect elimination ordering.
        # For split graphs, one can use the fact that the independent set vertices have neighborhoods that are subsets of the clique.
        
        # Actually, a simpler approach for split graphs:
        # Each edge is either within the clique (k vertices) or between clique and independent set.
        # Edges within the clique: they all belong to the clique {0, ..., k-1}.
        # But we need to partition ALL edges into cliques.
        # The clique {0, ..., k-1} covers all edges among the first k vertices.
        # For edges between clique and independent set:
        # Each vertex v >= k is connected to some subset S_v of [0, k).
        # The edges (u, v) for u in S_v form a star centered at v.
        # But a star is not a clique unless |S_v| <= 1.
        # So we need to group these edges into cliques.
        
        # Key insight: In a split graph, any clique contains at most one vertex from the independent set.
        # So each clique is either entirely within the clique part, or consists of one independent set vertex plus some clique vertices.
        # Therefore, edges between clique and independent set must be in cliques of the form {v} ∪ S where S ⊆ [0, k).
        # For a fixed v, the edges (u, v) for u in S_v must be covered by cliques {v} ∪ S where S ⊆ S_v.
        # But each such clique can only contain edges (u, v) for u in S.
        # So we need to partition S_v into subsets, and for each subset S, create clique {v} ∪ S.
        # But wait, that's not quite right. The clique {v} ∪ S contains edges (u1, u2) for u1, u2 in S as well as (u, v) for u in S.
        # But edges (u1, u2) for u1, u2 in S are already covered by the big clique {0, ..., k-1}.
        # So we can't use them again.
        
        # Let me reconsider. We need to partition the edge set.
        # Edge (u, v) where u, v < k: covered by clique {0, ..., k-1}.
        # Edge (u, v) where u < k, v >= k: must be in some clique.
        # Since the clique can contain at most one independent set vertex, the clique must be {v} ∪ S where u ∈ S ⊆ [0, k).
        # But then edges (u1, u2) for u1, u2 ∈ S are in this clique, but they're already in {0, ..., k-1}.
        # Contradiction! So we can't have cliques that contain edges within the clique part if those edges are already covered.
        
        # So the cliques covering cross edges must be of the form {u, v} (just the edge) or... wait.
        # Actually, if we have clique {v} ∪ S, the edges in this clique are:
        # - (v, u) for u in S (cross edges)
        # - (u1, u2) for u1, u2 in S (clique edges)
        # If we've already assigned clique edges to {0, ..., k-1}, we can't assign them again.
        # So we need cliques that ONLY contain cross edges.
        # But a clique with only cross edges and no clique-clique edges must be of size at most 2 (one clique vertex, one independent set vertex).
        # Because if it has two clique vertices u1, u2 and one independent set vertex v, it contains edge (u1, u2) which is a clique edge.
        
        # Hmm, so for cross edges, we can only use 2-cliques (edges) or... 
        # Wait, can we use cliques like {u, v} where u < k, v >= k? Yes, that's just an edge.
        # So each cross edge must be its own clique? That would be a lot of cliques.
        
        # Let me reconsider the problem. Maybe I'm overcomplicating.
        # Let's just use the greedy approach: for each edge, if not covered, add a clique containing it.
        # But that's what the seed does (each edge is a clique).
        
        # A better approach: group cross edges by the independent set vertex.
        # For vertex v, let S_v be its neighbors in [0, k).
        # We can form cliques {u, v} for each u in S_v. That's |S_v| cliques.
        # Total cliques = C(k, 2) edges in clique part... wait no, the clique part is one clique.
        # So: 1 clique for the k-clique, plus sum_{v>=k} |S_v| cliques for cross edges.
        # Total = 1 + sum_{v>=k} |S_v|
        
        # But can we do better? If two cross edges (u1, v1) and (u2, v2) can be in the same clique...
        # No, because a clique can have at most one independent set vertex.
        # So each cross edge is in a clique with at most one other vertex (its endpoint in the clique part).
        # So the minimum number of cliques for cross edges is the number of cross edges, unless...
        # Wait, no. A clique {u, v} has exactly one edge (u, v). So each cross edge needs its own 2-clique.
        # Unless we can have larger cliques that include multiple cross edges.
        # But as argued, a clique with two independent set vertices is impossible (they're not adjacent).
        # A clique with one independent set vertex v and multiple clique vertices u1, u2 contains edge (u1, u2).
        # So if we use such a clique, we're also covering edge (u1, u2).
        
        # So the strategy is:
        # 1. Start with the big clique {0, ..., k-1} covering all clique-clique edges.
        # 2. For cross edges, we can either:
        #    a. Use 2-cliques {u, v} for each cross edge.
        #    b. Use larger cliques {v, u1, u2, ...} which also cover clique-clique edges among {u1, u2, ...}.
        #       But those clique-clique edges are already covered by the big clique.
        #       So this doesn't help unless we DON'T use the big clique.
        
        # Alternative: Don't use the big clique. Instead, cover everything with smaller cliques.
        # This is getting complex. Let's just go with: 1 big clique + one 2-clique per cross edge.
        
        cliques = []
        # Big clique
        cliques.append(list(range(k)))
        
        # For each cross edge, add a 2-clique
        for (u, v) in edges:
            if u < k and v >= k:
                cliques.append([u, v])
            elif v < k and u >= k:
                cliques.append([v, u])
        
        return cliques

    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        results.append(partition_cliques(*spec))
    
    return results
# EVOLVE-BLOCK-END
