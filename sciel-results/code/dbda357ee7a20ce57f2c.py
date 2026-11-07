"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_instance(n, k, p, seed, time_limit):
        """
        Find a minimal clique cover for a split graph.
        
        In a split graph with clique C (size k) and independent set I (size n-k),
        each edge in I has both endpoints in C. Edges within C form a complete subgraph.
        
        Key insight: Each vertex in I can be associated with the subset of C it connects to.
        Two vertices in I cannot be in the same clique (since I is independent).
        So each vertex in I must be in a clique that also contains all its neighbors in C.
        
        However, multiple vertices in I that share the same neighbor set in C can be 
        "covered" by the same clique if we group them. But wait - they can't be in the same
        clique because they're not connected to each other.
        
        Let me reconsider: A clique can contain:
        - Any subset of C
        - Any subset of I where all vertices in the subset have the same neighbor set 
          that is a subset of the C-vertices in the clique
        
        Actually, for a clique to be valid:
        - All pairs within the clique must be edges
        - For vertices in I, they must be connected to all other vertices in the clique
        
        So if we have a clique containing some vertices from C (call it S ⊆ C) and some 
        from I (call it T ⊆ I), then:
        - All pairs in S must be edges (always true since C is a clique)
        - All pairs in T must be edges (never true if |T| > 1 since I is independent)
        - Each vertex in T must be connected to all vertices in S
        
        Therefore, each clique can contain at most one vertex from I!
        
        Wait, that's not right either. Let me re-examine.
        
        If T ⊆ I and |T| ≥ 2, then two vertices in T are not connected (since I is 
        independent), so they can't be in the same clique.
        
        So every clique contains at most one vertex from I.
        
        Therefore, the clique cover consists of:
        1. Cliques that cover edges within C (the complete subgraph on C)
        2. Cliques that cover edges between C and I
        
        For edges within C: Since C is a clique, we need to cover all C(k,2) edges.
        The minimum number of cliques to cover all edges of a complete graph K_k is 
        ⌈(k-1)/2⌉ if k is odd, k/2 if k is even? No, that's for edge coloring.
        
        Actually, for covering edges of K_k with cliques (where each clique is a 
        complete subgraph), the minimum number is k-1 (by taking stars centered at 
        each vertex except one, or by other constructions).
        
        Wait, no. If we take cliques that are edges (size 2), we need C(k,2) cliques.
        But we can use larger cliques. The minimum number of cliques to cover all 
        edges of K_n is n-1 (this is a known result - you can achieve it with 
        {n-1} cliques by a proper construction).
        
        Actually, the minimum number of complete subgraphs (cliques) needed to cover 
        all edges of K_n is n-1. This is achieved by taking for each i from 1 to n-1, 
        the clique {i, i+1, ..., n}.
        
        Hmm, but that's not minimal in general. Let me think again.
        
        For K_3: need 2 cliques (e.g., {1,2} and {1,3} cover all 3 edges)
        For K_4: need 3 cliques? {1,2,3}, {1,4}, {2,4} - no, {1,2,3} covers edges 
        (1,2),(1,3),(2,3), then we need (1,4),(2,4),(3,4). We can use {1,2,4} and 
        {1,3,4} and {2,3,4}? That's 4 cliques for the remaining edges, but they 
        overlap.
        
        Actually, the problem is: cover all edges of K_k with cliques, where each 
        edge must be in exactly one clique. This is an edge partition into cliques.
        
        The minimum number of cliques to partition the edges of K_n is ⌈n/2⌉ if n 
        is even, and (n-1)/2 if n is odd? No...
        
        Let me just use a greedy approach: repeatedly find the largest clique in 
        the remaining graph.
        
        But for split graphs, there's a better approach.
        
        Let me reconsider the structure:
        - C is a clique (all edges within C exist)
        - I is an independent set (no edges within I)
        - Cross edges exist with probability p
        
        Every edge is either within C or between C and I.
        
        For edges within C: We need to partition these C(k,2) edges into cliques.
        Since all vertices in C are mutually adjacent, any subset of C forms a clique.
        The minimum number of cliques to partition all edges of K_k is k-1.
        Construction: For i = 1, 2, ..., k-1, take the clique {i, i+1, i+2, ..., k}.
        Each edge (j, l) with j < l appears in exactly one clique: the one starting at j.
        
        For edges between C and I: Each vertex v in I is connected to some subset S_v ⊆ C.
        The edges incident to v are (u, v) for u ∈ S_v.
        We need to cover these edges with cliques. Each such clique must contain v 
        and some subset T ⊆ S_v. Since v is only connected to vertices in S_v, 
        the clique must be {v} ∪ T where T ⊆ S_v.
        
        But we also need to ensure that edges within C are not double-counted.
        
        Actually, the cliques for covering cross edges can also include multiple 
        vertices from C. But they can only include one vertex from I (since I is 
        independent).
        
        So the strategy is:
        1. Partition the edges of K_k (within C) into k-1 cliques using the 
           construction above.
        2. For each vertex v in I, the edges (u, v) for u ∈ S_v need to be covered.
           But these edges might already be covered by the cliques from step 1 if 
           those cliques include v. But they don't, since step 1 only uses vertices 
           from C.
           
        Wait, I think I'm overcomplicating this. Let me re-read the problem.
        
        "Every listed set must be a clique with at least two vertices and every edge 
        must lie in exactly one listed clique."
        
        So we need an edge partition into cliques.
        
        For a split graph, the edges are:
        - All edges within C (complete graph K_k)
        - Edges between C and I (each v in I connected to S_v ⊆ C)
        
        Since there are no edges within I, and each clique can contain at most one 
        vertex from I, the cliques that contain vertices from I can only be of the 
        form {v} ∪ T where v ∈ I and T ⊆ S_v.
        
        The cliques that don't contain any vertex from I are subsets of C.
        
        So the problem decomposes:
        1. Cover all edges within C using cliques that are subsets of C.
        2. Cover all cross edges (between C and I) using cliques that contain exactly 
           one vertex from I.
        
        For part 1: The minimum number of cliques to partition all edges of K_k is 
        k-1 (using the construction: for i=1..k-1, clique {i, i+1, ..., k}).
        
        For part 2: For each vertex v in I, we need to cover the edges (u, v) for 
        u ∈ S_v. Since all these edges share the common vertex v, we can cover them 
        all with a single clique {v} ∪ S_v (if S_v is non-empty). If S_v is empty, 
        no edges to cover.
        
        But wait, can we combine multiple vertices from I into one clique? No, because 
        they're not connected to each other.
        
        So the total number of cliques is:
        - (k-1) for covering K_k
        - (number of vertices in I with non-empty S_v) for covering cross edges
        
        But this might not be minimal. We might be able to cover some cross edges 
        using cliques that also cover some edges within C, reducing the total count.
        
        Actually, no - if a clique contains a vertex v from I and some vertices T from 
        C, it covers edges (v, u) for u ∈ T, but it also covers edges within T (which 
        are edges of K_k). If those edges within T are already covered by the cliques 
        from part 1, we'd be double-covering them, which is not allowed.
        
        So we need to be more careful. Let me think of it as: we need to partition 
        all edges into cliques. Each clique is a set of vertices where all pairs are 
        edges.
        
        Alternative approach: Use a greedy algorithm that repeatedly finds a 
        (maximal) clique in the remaining edge set.
        
        Given the time constraint, let me implement a reasonable heuristic:
        
        1. Build the adjacency list.
        2. Maintain the set of uncovered edges.
        3. Repeatedly find a clique that covers many uncovered edges.
        4. Remove those edges from the uncovered set.
        5. Continue until no edges remain.
        
        For efficiency, we can use the fact that split graphs have a simple structure.
        
        Let me implement a simpler approach:
        
        For the edges within C: Use the k-1 clique construction.
        For the cross edges: For each vertex v in I, if it has neighbors S_v in C, 
        we need to cover the edges (u, v) for u in S_v. 
        
        But we can potentially merge some of these with the C-cliques. However, since 
        the C-cliques already cover all edges within C, adding a vertex from I to a 
        C-clique would create new edges (between the I-vertex and the C-vertices in 
        the clique) but would not create new edges within C (since those are already 
        covered).
        
        Actually, I think the cleanest approach is:
        
        1. Start with the k-1 cliques that cover K_k: {0, 1, ..., k-1}, {1, 2, ..., 
           k-1}, ..., {k-2, k-1}. Wait, that's not right.
           
           Let me use: for i in range(k-1): clique = [i, i+1, ..., k-1]
           Edge (j, l) with j < l is covered by clique j (since j is the smallest 
           index, and the clique starting at j contains all vertices from j to k-1).
           Actually, edge (j, l) is in clique j if j < l, and in clique j+1, j+2, 
           ..., l-1 as well. So it's covered multiple times!
           
           I need a partition, not just a cover.
           
           For a partition of K_k edges into cliques, one standard construction is:
           If k is even, k = 2m, use m disjoint perfect matchings (1-factors). Each 
           matching is a set of m disjoint edges, so each "clique" is just an edge. 
           That uses k(k-1)/2 cliques, which is bad.
           
           Better: Use larger cliques. For K_4, we can partition edges into:
           {0,1,2}, {0,3}, {1,3}, {2,3} - that's 4 cliques, but {0,1,2} covers 3 
           edges, and the others cover 1 each. Total edges = 6, so 3+1+1+1=6. ✓
           
           Actually, the minimum number of cliques to partition the edges of K_n is 
           ⌈(n-1)/2⌉ if we allow cliques of any size? No, that's for edge coloring.
           
           I recall that the minimum number of complete subgraphs needed to partition 
           the edges of K_n is n-1. This is achieved by:
           For i = 0, 1, ..., n-2: clique_i = {i, i+1, i+2, ..., n-1}
           But as I noted, this doesn't partition the edges; each edge is in multiple 
           cliques.
           
           Let me look up: The edge set of K_n can be partitioned into n-1 cliques 
           iff... actually I'm not sure this is always possible with n-1 cliques.
           
           For K_3: 3 edges. Can we partition into 2 cliques? {0,1} and {0,2} - 
           but then edge (1,2) is uncovered. {0,1,2} covers all 3 edges in 1 clique. 
           So for K_3, we need 1 clique (the whole triangle).
           
           For K_4: 6 edges. Can we partition into 2 cliques? Each clique with 4 
           vertices covers 6 edges, but we only have 4 vertices total. If we use 
           {0,1,2} (3 edges) and {0,3}, {1,3}, {2,3} (1 edge each), that's 4 cliques. 
           Can we do 3? {0,1,2}, {0,3,?} - {0,3} only covers 1 edge. 
           
           I think for K_4, the minimum is 3: {0,1,2}, {1,2,3}, {0,3} - but 
           {0,1,2} covers (0,1),(0,2),(1,2), {1,2,3} covers (1,2),(1,3),(2,3) - 
           but (1,2) is in both! Not a partition.
           
           OK, I'll just use a greedy heuristic. Given the small sizes (k up to 30), 
           this should be fast enough.
        
        Let me implement a greedy algorithm:
        1. Build adjacency from edges.
        2. Maintain uncovered_edges as a set.
        3. While uncovered_edges is not empty:
           a. Find a clique C such that all pairs in C are in uncovered_edges 
              (or at least many are).
           b. Actually, for a valid clique in the partition, we need that all pairs 
              in C are edges in the original graph, and we remove those edges from 
              uncovered_edges. But we should only remove edges that are still 
              uncovered.
           c. To maximize efficiency, find a clique that covers the most uncovered 
              edges.
        
        For simplicity and speed, I'll use this heuristic:
        - Always include high-degree vertices in the current clique.
        - Start with a vertex, greedily add neighbors that are connected to all 
          current clique members.
        
        Let me code this up.
        """
        start_time = time.time()
        
        # Build adjacency
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Uncovered edges
        uncovered = set(edge_set)
        cliques = []
        
        while uncovered:
            if time.time() - start_time > time_limit:
                # Fallback: use individual edges
                for e in uncovered:
                    cliques.append(list(e))
                break
            
            # Find a good clique
            # Strategy: pick a vertex with high degree in uncovered graph
            # Then greedily build a clique
            
            # Compute degrees in uncovered graph
            uncov_deg = {i: 0 for i in range(n)}
            for u, v in uncovered:
                uncov_deg[u] += 1
                uncov_deg[v] += 1
            
            # Start with the vertex of highest uncovered degree
            start = max(range(n), key=lambda x: uncov_deg[x])
            if uncov_deg[start] == 0:
                break
            
            # Greedily build a clique starting from start
            current = {start}
            candidates = set(adj[start]) & {v for v in adj[start] if uncov_deg[v] > 0}
            
            # Sort candidates by uncovered degree
            candidates = sorted(candidates, key=lambda x: -uncov_deg[x])
            
            for v in candidates:
                # Check if v is connected to all in current
                if all(v in adj[u] for u in current):
                    current.add(v)
            
            # Convert to list
            clique_list = sorted(current)
            cliques.append(clique_list)
            
            # Remove covered edges from uncovered
            covered = set()
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    e = (min(clique_list[i], clique_list[j]), 
                         max(clique_list[i], clique_list[j]))
                    if e in uncovered:
                        covered.add(e)
            
            uncovered -= covered
        
        return cliques
    
    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        cliques = solve_instance(*spec, time_limit=30)
        # Ensure types are int and list
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
    
    return results
# EVOLVE-BLOCK-END
