"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        if time.time() > deadline:
            # Fallback: return trivial solution (edges) if time runs out
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]
            
        edges = instance(n, k, p, seed)
        if not edges:
            return []
            
        # 1. Generate all maximal cliques (or just all cliques of size >= 2)
        # In a split graph, vertices are partitioned into C (clique) and I (independent set).
        # Any clique is a subset of C plus any subset of I that is fully connected to that subset of C.
        # Since I is independent, a clique can contain at most 1 vertex from I.
        # So cliques are:
        # a) Subsets of C of size >= 2
        # b) Subsets of C of size >= 1 plus one vertex from I that is connected to all in the subset.
        
        # Let's generate candidate cliques.
        # To keep ILP small, we should probably just use maximal cliques?
        # No, non-maximal cliques might be needed for optimal packing?
        # Actually, for edge partitioning into cliques, using maximal cliques is sufficient?
        # Not necessarily. But for split graphs, the structure is simple.
        # Let's generate all cliques of size >= 2.
        # Number of cliques in C: 2^k - k - 1. For k=30, this is ~10^9. Too many.
        
        # We need a smarter set of candidate cliques.
        # Observation: In a split graph, any edge (u,v) with u,v in C is covered by some clique.
        # Any edge (u,v) with u in C, v in I is covered by some clique.
        
        # Let's restrict candidate cliques to "maximal" ones?
        # Or maybe just cliques of size 2 and 3?
        # The seed program uses size 2. 
        # The Erdős-Ordman-Zalcstein bound suggests O(n^2) but we want minimal.
        # For split graphs, it is known that the clique partition number is related to the number of edges in the bipartite part?
        
        # Let's try a heuristic or a restricted ILP.
        # If we use all maximal cliques, how many are there?
        # A maximal clique is C' subset C, and I' subset I such that I' is the set of vertices in I connected to all in C'.
        # For a fixed C', there is only 1 such maximal clique (containing C' and all possible I vertices).
        # So the number of maximal cliques is at most 2^k. Still too big for k=30.
        
        # Alternative: Greedy + Local Search?
        # Or ILP with a restricted set of cliques.
        # Let's try to find a good set of cliques.
        # Since the graph is a split graph, the "hard" part is covering the edges between C and I.
        # Edges within C can be covered by large cliques.
        
        # Let's try a simple greedy approach first to see if it beats the seed.
        # Seed: #edges.
        # Greedy: Pick largest clique, remove edges, repeat.
        
        # Actually, let's look at the structure.
        # Vertices 0..k-1 are the clique C.
        # Vertices k..n-1 are the independent set I.
        
        # We can cover all edges within C with just 1 clique: C itself.
        # Then we are left with edges between C and I.
        # Let E_CI be the set of edges (u, v) with u in C, v in I.
        # We need to partition E_CI into cliques.
        # A clique containing an edge (u, v) with u in C, v in I can also contain other vertices from C connected to v.
        # So for each v in I, let N(v) be the set of neighbors in C.
        # The edges incident to v are {(u, v) : u in N(v)}.
        # These edges can be covered by cliques of the form {S, v} where S is a clique in C and S subset N(v).
        # Since C is a clique, any S subset N(v) is a clique.
        # So for each v, we need to partition the star centered at v (restricted to N(v)) into sets {S, v}.
        # This is equivalent to partitioning the set N(v) into subsets S_1, ..., S_m such that each S_j forms a clique with v.
        # Since any subset of C is a clique, we can just group the neighbors of v arbitrarily?
        # Wait, the cliques must be disjoint in terms of edges.
        # Edges incident to different v's are disjoint (since I is independent, no edge between v1, v2).
        # So the problem decomposes by vertex v in I!
        
        # For each v in I:
        # We have edges (u, v) for u in N(v).
        # We can cover these edges with cliques {u1, u2, ..., um, v}.
        # Each such clique covers edges (u1, v), ..., (um, v).
        # To minimize the number of cliques for this v, we should use as large a clique as possible?
        # If we use one clique {N(v), v}, it covers all edges incident to v.
        # This uses 1 clique per v (if N(v) is not empty).
        # Total cliques = 1 (for C) + sum_{v in I, N(v) != empty} 1.
        
        # Is this valid?
        # Clique 1: C = {0, ..., k-1}. Covers all edges within C.
        # For each v in I with N(v) non-empty:
        #   Clique v: N(v) U {v}.
        #   This is a clique because C is a clique and all u in N(v) are connected to v.
        #   It covers edges (u, v) for all u in N(v).
        #   Does it cover any other edges? No, because I is independent, so no edges between v and other I vertices.
        #   And edges within N(v) are already covered by Clique 1.
        #   Wait, the requirement is "every edge must lie in exactly one listed clique".
        #   If we list C, it covers all edges (u1, u2) with u1, u2 in C.
        #   If we list N(v) U {v}, it covers edges (u, v) for u in N(v).
        #   Does it cover edges (u1, u2) for u1, u2 in N(v)? Yes, because they are in the clique.
        #   But those edges are ALSO in Clique 1.
        #   This violates "exactly one".
        
        # So we cannot use C as a clique if we use larger cliques that include subsets of C.
        # We must partition the edges.
        
        # Let's rethink.
        # Edges within C: E_CC.
        # Edges between C and I: E_CI.
        # No edges within I.
        
        # If we use a clique K that contains some vertices from C and some from I, it covers edges within K.
        # If K contains v in I and S subset C, it covers:
        # - Edges (u1, u2) for u1, u2 in S.
        # - Edges (u, v) for u in S.
        
        # We need to cover all E_CC and E_CI exactly once.
        
        # Strategy:
        # 1. Handle E_CI first?
        # For each v in I, let N(v) be neighbors in C.
        # We must cover edges (u, v) for u in N(v).
        # Any clique covering (u, v) must contain u and v. It may contain other vertices from C.
        # Let's assign each edge (u, v) to a clique.
        # Since edges incident to different v's are disjoint, we can solve for each v independently?
        # No, because the cliques might share vertices from C, and thus cover edges within C.
        # If a clique K_v contains S_v subset C and v, it covers edges within S_v.
        # These edges within S_v must NOT be covered by any other clique.
        # So the sets S_v (for v in I) must be such that the edges within them are disjoint?
        # Actually, the union of cliques must partition the edges.
        
        # Let's try this:
        # For each v in I, we choose a subset S_v of N(v).
        # We form a clique K_v = S_v U {v}.
        # This covers edges (u, v) for u in S_v, and edges within S_v.
        # We need to cover all edges (u, v) for u in N(v). So union of S_v must cover N(v)? No, each edge (u,v) must be in exactly one clique.
        # If we put u in S_v, the edge (u,v) is covered by K_v.
        # Can (u,v) be covered by another clique? Only if another clique contains u and v.
        # But v is only in K_v (if we restrict v to be in only one clique? No, v can be in multiple cliques, but edges must be partitioned).
        # If v is in K_v1 and K_v2, then edges (u, v) are partitioned between them.
        # But typically, it's best to group all neighbors of v into one clique to minimize the count?
        # If we use one clique K_v = N(v) U {v} for each v, we cover all edges incident to v.
        # But we also cover edges within N(v).
        # The edges within N(v) are a subset of E_CC.
        # If we do this for all v, we cover many edges in E_CC multiple times (once for each v whose N(v) contains them).
        # This is invalid.
        
        # So we must be careful with E_CC.
        # Maybe we should cover E_CC with cliques that do NOT contain any vertex from I?
        # If we cover all E_CC with cliques consisting only of vertices from C, then we can freely use cliques containing I vertices to cover E_CI, as long as we don't double cover E_CC.
        # But if a clique K_v = S_v U {v} contains S_v subset C, it covers edges within S_v.
        # If those edges were already covered by a C-only clique, it's a conflict.
        # So, if we decide to cover E_CC entirely with C-only cliques, then we cannot use any clique that contains vertices from C and I, because it would cover some C-C edges again.
        # Wait, if K_v covers edges within S_v, and those edges are already covered, it's a violation.
        # So, if we want to use cliques with I vertices, we must NOT cover the edges within S_v elsewhere.
        # This means the C-C edges covered by K_v must be removed from the pool of C-C edges to be covered by C-only cliques.
        
        # This looks like a complex interaction.
        # However, note that for each v, S_v is a subset of N(v).
        # The edges covered within C by K_v are all pairs in S_v.
        # To minimize cliques, we want S_v to be large?
        # If S_v = N(v), we cover all edges incident to v with 1 clique.
        # The cost is that we "use up" the edges within N(v).
        
        # Let's try a simple strategy:
        # 1. For each v in I, let S_v = N(v).
        # 2. Create clique K_v = S_v U {v}.
        # 3. Let E_used_CC be the set of edges within C that are covered by any K_v.
        #    E_used_CC = union of { (u1, u2) : u1, u2 in S_v } for all v.
        # 4. The remaining edges in E_CC are E_rem_CC = E_CC \ E_used_CC.
        # 5. We need to cover E_rem_CC with cliques consisting only of vertices from C.
        #    Since C is a clique, we can cover any set of edges in a clique graph with... well, we need to partition E_rem_CC into cliques.
        #    But E_rem_CC is a subgraph of the complete graph on C.
        #    We can just cover each remaining edge with a clique of size 2?
        #    Or find a better partition.
        #    Given time constraints, covering each remaining edge with a size-2 clique is a valid fallback.
        #    Number of cliques = |I_nonempty| + |E_rem_CC|.
        
        # Let's estimate the size.
        # |I_nonempty| is at most n-k.
        # |E_rem_CC| is at most k(k-1)/2.
        # Total <= n-k + k^2/2.
        # Seed: |E| = k(k-1)/2 + p*k*(n-k).
        # For n=90, k=30, p=0.7:
        # Seed: 435 + 0.7*30*60 = 435 + 1260 = 1695.
        # New: 60 + 435 = 495.
        # Huge improvement!
        
        # Is it valid?
        # Cliques K_v = N(v) U {v}.
        # Edges covered:
        # - (u, v) for u in N(v). All such edges are covered exactly once (by K_v).
        # - (u1, u2) for u1, u2 in N(v). These are covered by K_v.
        # Are they covered by any other K_w?
        # If u1, u2 are in N(v) and N(w), then the edge (u1, u2) is covered by both K_v and K_w.
        # This is a violation!
        
        # So we cannot just take S_v = N(v) for all v.
        # We need the sets S_v to be such that the edges within them are disjoint?
        # Or more generally, the cliques K_v must not share edges.
        # Since K_v and K_w share vertices in C, they share edges if they share at least 2 vertices in C.
        # So we need S_v and S_w to intersect in at most 1 vertex for all v != w.
        # This is a packing problem.
        
        # Given the complexity, let's stick to a simpler valid strategy:
        # Use only cliques of size 2 for E_CI?
        # That's the seed.
        
        # How about:
        # For each v in I, cover edges (u, v) with cliques {u, v}.
        # This is the seed.
        
        # Can we do better?
        # If we use a clique {u1, u2, v}, we cover (u1, v), (u2, v) and (u1, u2).
        # We save 1 clique (instead of 2 for the edges, we use 1).
        # But we "consume" the edge (u1, u2) which might be needed elsewhere.
        # If we don't need to cover (u1, u2) separately (i.e., it's not in E_rem_CC), it's fine.
        
        # Let's try a greedy approach:
        # 1. Start with all edges uncovered.
        # 2. While edges remain:
        #    - Find a clique that covers the most uncovered edges.
        #    - Add it to the partition.
        #    - Mark those edges as covered.
        # This is NP-hard generally, but for split graphs, maybe fast enough?
        # Or just a heuristic.
        
        # Given the time limit, let's try the "Star + Remainder" approach but fix the conflict.
        # Conflict: (u1, u2) covered by K_v and K_w.
        # Solution: When forming K_v, only include edges (u1, u2) if they haven't been "reserved" by another K_w?
        # This is getting complicated.
        
        # Alternative:
        # Cover all E_CI with size-2 cliques {u, v}.
        # Cover all E_CC with size-2 cliques {u, u'}.
        # This is the seed.
        
        # Let's try to merge some E_CI edges.
        # For each v, if we have many neighbors, we can group them.
        # If we group u1, u2 with v, we use 1 clique for 3 edges.
        # Previously 3 cliques.
        # Savings: 2 cliques.
        # Cost: Edge (u1, u2) is now covered. We must ensure it's not covered again.
        # If we remove (u1, u2) from the E_CC pool, we are fine.
        
        # Algorithm:
        # 1. Initialize E_CC_remaining = all edges in C.
        # 2. Initialize E_CI_remaining = all edges between C and I.
        # 3. Result = []
        # 4. For each v in I:
        #    Let N(v) be neighbors in C.
        #    While N(v) is not empty:
        #       Pick a subset S of N(v) such that all edges within S are in E_CC_remaining.
        #       To maximize savings, pick S as large as possible?
        #       If we pick S, we form clique K = S U {v}.
        #       Remove edges (u, v) for u in S from E_CI_remaining.
        #       Remove edges within S from E_CC_remaining.
        #       Add K to Result.
        #       N(v) = N(v) \ S.
        # 5. After processing all v, cover remaining E_CC_remaining with size-2 cliques.
        # 6. Cover remaining E_CI_remaining (should be empty if we processed all v) with size-2 cliques.
        
        # How to pick S?
        # If we pick S = N(v), we might remove many E_CC edges.
        # But if some edge in N(v) was already removed by a previous v', we can't include both endpoints in S?
        # No, if edge (u1, u2) is removed, we can still include u1 and u2 in S?
        # If we include u1 and u2 in S, the clique K covers edge (u1, u2).
        # But (u1, u2) is already covered. Violation.
        # So S must be a clique in the graph of "available" E_CC edges?
        # i.e., for all u1, u2 in S, (u1, u2) must be in E_CC_remaining.
        
        # So S must be a clique in the subgraph of C induced by E_CC_remaining.
        # And S must be a subset of N(v).
        
        # Greedy: For each v, find the largest clique S in N(v) that is also a clique in E_CC_remaining.
        # This is a Max Clique problem on a small graph (size <= k).
        # k=30. Max clique is fast.
        
        # Let's implement this.
        
        edges_cc = [(u, v) for u in range(k) for v in range(u+1, k)]
        edges_ci = [(u, v) for v in range(k, n) for u in range(k) if (u, v) in set(edges) or (v, u) in set(edges)]
        # Wait, instance() returns sorted edges.
        # Let's build adjacency sets for efficiency.
        
        adj_cc = {u: set() for u in range(k)}
        adj_ci = {v: set() for v in range(k, n)}
        
        for u, v in edges:
            if u < k and v < k:
                adj_cc[u].add(v)
                adj_cc[v].add(u)
            elif u < k and v >= k:
                adj_ci[v].add(u)
            elif v < k and u >= k:
                adj_ci[u].add(v)
            else:
                # Should not happen in split graph
                pass
                
        E_cc_remaining = set(edges_cc)
        E_ci_remaining = set((u, v) for v in range(k, n) for u in adj_ci[v])
        
        cliques = []
        
        # Process each v in I
        for v in range(k, n):
            neighbors = list(adj_ci[v])
            if not neighbors:
                continue
                
            # Find largest clique S in neighbors such that all edges within S are in E_cc_remaining
            # Greedy: try to add vertices one by one?
            # Or just use a simple heuristic: sort neighbors by degree in E_cc_remaining?
            
            # Simple heuristic: 
            # Start with S = []
            # Iterate through neighbors, if u can be added to S (i.e., connected to all in S in E_cc_remaining), add it.
            # This finds a clique, but not necessarily the largest.
            
            S = []
            for u in neighbors:
                can_add = True
                for u2 in S:
                    if (min(u, u2), max(u, u2)) not in E_cc_remaining:
                        can_add = False
                        break
                if can_add:
                    S.append(u)
                    
            if S:
                clique = S + [v]
                cliques.append(clique)
                # Remove covered edges
                # Edges (u, v) for u in S
                for u in S:
                    E_ci_remaining.discard((u, v))
                # Edges within S
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        E_cc_remaining.discard((min(S[i], S[j]), max(S[i], S[j])))
                        
            # Handle remaining neighbors of v (if any, due to greedy failure to find a clique)
            # If S was not all neighbors, some edges (u, v) remain.
            # We must cover them.
            # Since we can't form a clique with them and others (due to E_cc constraints),
            # we can use size-2 cliques {u, v}.
            for u in neighbors:
                if (u, v) in E_ci_remaining:
                    cliques.append([u, v])
                    E_ci_remaining.discard((u, v))
                    # Note: This does NOT remove any E_cc edges, as size-2 clique {u, v} covers no C-C edges.
                    
        # Cover remaining E_cc_remaining
        for u, v in E_cc_remaining:
            cliques.append([u, v])
            
        # Cover any remaining E_ci_remaining (should be empty)
        for u, v in E_ci_remaining:
            cliques.append([u, v])
            
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed-like solution
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)
            
    return results
# EVOLVE-BLOCK-END
