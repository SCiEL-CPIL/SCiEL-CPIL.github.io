"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        """
        Find a clique partition of the split graph with:
        - Clique C = {0, ..., k-1}
        - Independent set I = {k, ..., n-1}
        - Cross edges between C and I with probability p
        
        Strategy:
        1. The clique C must be covered by cliques. Since C is a complete graph,
           we can partition it into cliques. But note: we want to minimize total cliques.
        2. Each vertex in I has some neighborhood N(v) in C. The edges incident to v 
           are (u, v) for u in N(v). These edges must be covered by cliques that contain v.
           Since I is independent, any clique containing v can only contain v and vertices from N(v).
           Moreover, since N(v) is a subset of C (which is a clique), N(v) ∪ {v} is a clique.
        3. So for each v in I, the edges (u, v) for u in N(v) must be covered by cliques 
           that contain v. We can group these edges into cliques: each such clique is 
           {v} ∪ S where S ⊆ N(v) and S is a clique in C (which it always is since C is complete).
           But we need to cover all edges (u,v) for u in N(v) with disjoint edge sets.
           
        Actually, let's think more carefully:
        - Edges within C: must be covered by cliques that are subsets of C.
        - Edges between C and I: for each v in I, edges (u,v) for u in N(v). These must be 
          covered by cliques containing v. Each such clique can be {v} ∪ S_v where S_v ⊆ N(v).
          The edge (u,v) is in clique {v} ∪ S_v iff u ∈ S_v.
          We need to partition N(v) into subsets S_{v,1}, S_{v,2}, ..., S_{v,m_v} such that 
          each {v} ∪ S_{v,j} is a clique (always true) and the edges are disjoint.
          Since the edges incident to v are disjoint from edges incident to other vertices in I,
          we can handle each v independently for the cross edges.
        
        However, we also need to cover the edges within C. The cliques used to cover cross edges 
        (like {v} ∪ S_{v,j}) also cover some edges within C (the edges within S_{v,j}). 
        So we need to coordinate.
        
        Better approach:
        - Start with the full clique C. We need to partition all edges of the graph.
        - The edges within C form a complete graph K_k. We need to cover these edges with cliques.
        - For each v in I, the edges (u,v) for u in N(v) must be covered.
        
        Key insight: Any clique in the partition that contains a vertex v from I must be of the form 
        {v} ∪ S where S ⊆ N(v) ⊆ C. Such a clique covers:
        - The cross edges (u,v) for u in S
        - The edges within S (which are edges of C)
        
        Cliques that don't contain any vertex from I are subsets of C.
        
        So we can think of it as:
        1. For each v in I, decide how to partition N(v) into groups. Each group S gives a clique {v}∪S.
        2. The remaining edges within C (those not covered by any {v}∪S) must be covered by cliques that are subsets of C.
        
        To minimize the total number of cliques, we want to maximize the coverage of C's edges by the cross cliques.
        
        For a fixed v, if we partition N(v) into groups S_1, ..., S_m, the number of cliques contributed 
        for v is m, and the edges of C covered are the union of edges within each S_i.
        
        The total number of cliques = (sum over v in I of m_v) + (number of cliques needed to cover remaining edges of C).
        
        Since the remaining edges of C form a graph on k vertices, and we need to cover them with cliques, 
        the minimum number of cliques to cover a graph G is at least the maximum degree / (something)... 
        Actually, covering edges with cliques is equivalent to edge coloring where each color class is a clique.
        
        For a complete graph K_k, the minimum number of cliques to cover all edges is... well, we can use 
        large cliques. But if some edges are already covered, we need to cover the rest.
        
        Let's try a simpler heuristic:
        - For each v in I, put all of N(v) into one clique: {v} ∪ N(v). This uses 1 clique per v in I 
          (only for v with non-empty N(v)), and covers all cross edges plus all edges within N(v).
        - Then cover the remaining edges of C with additional cliques.
        
        The remaining edges of C are those edges (u,w) in C where not both u and w are in N(v) for any v 
        whose clique {v}∪N(v) we used. Actually, an edge (u,w) in C is covered by clique {v}∪N(v) iff 
        u,w ∈ N(v). So edge (u,w) is covered if there exists some v such that u,w ∈ N(v).
        
        Let U = union of all N(v) for v in I. Edges within U that are "covered" are those (u,w) where 
        there exists v with u,w ∈ N(v). But actually, we need to track which specific edges are covered.
        
        This is getting complex. Let me try a different approach:
        
        Greedy approach:
        1. Build the graph.
        2. Use a greedy edge-coloring-like algorithm where each color class must be a clique.
        
        Actually, let's use a simpler strategy that should work well for split graphs:
        
        Strategy:
        - The clique C = {0,...,k-1} is a complete graph.
        - For each v in I, N(v) is a subset of C.
        - Create cliques as follows:
          a) For each v in I with N(v) non-empty, create clique {v} ∪ N(v). 
             This covers all edges (u,v) for u in N(v) and all edges within N(v).
          b) After doing this for all v, some edges within C may still be uncovered.
             Cover the remaining edges of C using a greedy clique cover.
        
        But this might not be optimal because using one big clique per v might not be best.
        
        Alternative: Since the problem asks for a good solution (not necessarily optimal), 
        and the seed just returns edges as cliques (which is very bad), any reasonable clique partition 
        should be much better.
        
        Let me implement the simple strategy above and see.
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Vertices in I = {k, ..., n-1}
        I = list(range(k, n))
        C = list(range(k))
        
        cliques = []
        covered_edges = set()
        
        # Step 1: For each v in I, create clique {v} ∪ N(v)
        for v in I:
            neighbors = sorted(adj[v])
            if not neighbors:
                continue
            clique = [v] + neighbors
            # Check that this is a clique
            # All pairs in clique must be edges
            valid = True
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    a, b = clique[i], clique[j]
                    if b not in adj[a]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                cliques.append(clique)
                # Mark edges as covered
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        covered_edges.add((min(a,b), max(a,b)))
        
        # Step 2: Cover remaining edges of C
        remaining_edges = []
        for u, v in edges:
            edge = (min(u,v), max(u,v))
            if edge not in covered_edges:
                remaining_edges.append(edge)
        
        # Greedy clique cover for remaining edges (all within C)
        remaining_set = set(remaining_edges)
        while remaining_set:
            # Pick an edge
            e = next(iter(remaining_set))
            u, v = e
            # Find the largest clique containing this edge from remaining edges
            # Start with {u, v}
            clique = {u, v}
            # Try to add more vertices
            changed = True
            while changed:
                changed = False
                for w in C:
                    if w in clique:
                        continue
                    # Check if w is connected to all in clique via remaining edges
                    can_add = True
                    for x in clique:
                        edge = (min(w,x), max(w,x))
                        if edge not in remaining_set:
                            can_add = False
                            break
                    if can_add:
                        clique.add(w)
                        changed = True
            
            clique_list = sorted(clique)
            cliques.append(clique_list)
            # Remove covered edges
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    edge = (min(a,b), max(a,b))
                    remaining_set.discard(edge)
        
        # Convert to lists of ints
        result = []
        for clique in cliques:
            result.append([int(v) for v in clique])
        
        return result
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(clique_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
