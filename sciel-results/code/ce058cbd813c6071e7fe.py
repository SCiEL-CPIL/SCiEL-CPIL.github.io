"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """Find a minimal clique partition for a split graph using a
        structural approach: since C is a clique, we can cover C-C edges
        with large cliques, and each I-vertex can be paired with a clique
        in its C-neighborhood.
        
        Key insight: In a split graph with clique C and independent set I:
        - Any clique has at most one vertex from I
        - A clique is either a subset of C, or {i} ∪ S where i ∈ I and S ⊆ C is a clique
        
        Strategy:
        1. For each I-vertex i, we must cover edges (i, c) for c ∈ N_C(i).
           These edges can only be covered by cliques containing i (since i is in I).
           So we need to partition N_C(i) into cliques (subsets of C that are cliques,
           which they all are since C is a clique), and pair each with i.
           But wait - we can also have cliques that are just subsets of C.
           
        Actually, the key constraint is: every edge (i, c) where i ∈ I and c ∈ C
        must be in exactly one clique. Since i ∈ I, any clique containing edge (i,c)
        must contain i and c, and any other vertices in that clique must be in C
        (since no other I vertex can join). So cliques containing i are of form
        {i} ∪ S where S ⊆ C is a clique (always true) and S ⊆ N_C(i).
        
        So for each i ∈ I, we need to partition N_C(i) into subsets S_1, ..., S_m
        such that each {i} ∪ S_j is a clique (always true since C is a clique and
        S_j ⊆ N_C(i)), and these cliques cover all edges (i, c) for c ∈ N_C(i).
        Each edge (i,c) is covered by exactly one clique containing i.
        
        Additionally, C-C edges: every edge (u,v) with u,v ∈ C must be in exactly
        one clique. These can be in cliques {i} ∪ S (if u,v ∈ S for some i) or in
        cliques that are subsets of C.
        
        The problem reduces to:
        - For each i ∈ I, choose a partition of N_C(i) into cliques S_{i,1}, ..., S_{i,m_i}
        - The remaining C-C edges (those not covered by any {i} ∪ S_{i,j}) must be
          partitioned into cliques within C.
        
        To minimize total cliques, we want to maximize the C-C edges covered by
        the {i} ∪ S cliques, and use as few cliques as possible overall.
        
        A simple approach: for each i, use one clique {i} ∪ N_C(i) if N_C(i) is
        non-empty. This covers all edges from i to C, and also covers all C-C edges
        within N_C(i). Then handle remaining C-C edges.
        
        But this might not be optimal. Let's think more carefully.
        
        Actually, since C is a clique, any subset of C is a clique. So for each i ∈ I,
        we can cover all edges (i, c) for c ∈ N_C(i) with a single clique {i} ∪ N_C(i).
        This is valid and covers |N_C(i)| edges from i plus all C-C edges within N_C(i).
        
        After doing this for all i, the remaining edges are C-C edges (u,v) where
        u and v are not both in N_C(i) for any i. We need to partition these into
        cliques within C.
        
        Since C is a clique, we can cover all remaining C-C edges with a single
        clique C' where C' is the set of C vertices that have at least one uncovered
        C-C edge. But we need to be careful: if we use one big clique, it covers
        all C-C edges among its vertices.
        
        Let me implement this approach and see how it does.
        """
        edges = instance(n, k, p, seed)
        edge_set = set()
        for e in edges:
            if e[0] < e[1]:
                edge_set.add(e)
            else:
                edge_set.add((e[1], e[0]))
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        uncovered = set(edge_set)
        cliques = []
        
        # Phase 1: For each I-vertex, create clique {i} ∪ N_C(i)
        # Process in order of decreasing |N_C(i)| to cover more C-C edges early
        i_order = sorted(I, key=lambda i: -len(adj[i] & set(C)))
        
        for i in i_order:
            if time.time() > deadline:
                break
            # Get C-neighbors of i that have uncovered edges to i
            c_neighbors = [c for c in C if (min(i, c), max(i, c)) in uncovered]
            if not c_neighbors:
                continue
            
            # Create clique {i} ∪ c_neighbors
            full_clique = [i] + c_neighbors
            cliques.append(full_clique)
            
            # Remove all edges within this clique
            vlist = full_clique
            for idx in range(len(vlist)):
                for jdx in range(idx + 1, len(vlist)):
                    a, b = vlist[idx], vlist[jdx]
                    uncovered.discard((min(a, b), max(a, b)))
        
        # Phase 2: Handle remaining C-C edges
        remaining_cc = [(u, v) for (u, v) in uncovered if u < k and v < k]
        if remaining_cc:
            # Get the set of C vertices involved in remaining edges
            cc_vertices = set()
            for u, v in remaining_cc:
                cc_vertices.add(u)
                cc_vertices.add(v)
            
            # Since C is a clique, all pairs in cc_vertices are edges in the original graph.
            # We can cover all remaining C-C edges with one clique: cc_vertices
            # But we need to verify these are all still uncovered.
            # Actually, since C is a complete graph, any subset is a clique.
            # The remaining edges are exactly the C-C edges not yet covered.
            # We can use one clique containing all cc_vertices to cover all of them.
            
            full_clique = list(cc_vertices)
            if len(full_clique) >= 2:
                cliques.append(full_clique)
                # Remove all edges within this clique
                for idx in range(len(full_clique)):
                    for jdx in range(idx + 1, len(full_clique)):
                        a, b = full_clique[idx], full_clique[jdx]
                        uncovered.discard((min(a, b), max(a, b)))
        
        # Phase 3: Any remaining edges (should be empty, but handle generically)
        while uncovered:
            if time.time() > deadline:
                break
            u, v = next(iter(uncovered))
            clique = {u, v}
            # Try to expand
            candidates = [x for x in range(n) if x != u and x != v]
            for c in candidates:
                if all((min(c, x), max(c, x)) in uncovered for x in clique):
                    clique.add(c)
            
            cliques.append(list(clique))
            vlist = list(clique)
            for idx in range(len(vlist)):
                for jdx in range(idx + 1, len(vlist)):
                    a, b = vlist[idx], vlist[jdx]
                    uncovered.discard((min(a, b), max(a, b)))
        
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        cliques = solve_split(*spec)
        result.append(cliques)

    return result
# EVOLVE-BLOCK-END
