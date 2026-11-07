import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency sets for fast lookup
        # adj[u] is a set of v such that (u,v) is an edge
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # 1. Identify the core clique C = {0, ..., k-1}
        # All edges within C must be covered.
        # We can cover all internal edges with the single clique C itself.
        # However, we can potentially "merge" some vertices from the independent set I
        # into C if they are universal to C (connected to all of C).
        # If a vertex v in I is connected to all u in C, then C U {v} is a clique.
        # Adding v to C covers the edges (u, v) for all u in C with 1 clique instead of |C| cliques.
        # This is a huge win.
        
        # Let's find all vertices in I that are universal to C.
        # Universal to C means adj[v] contains all of {0, ..., k-1}.
        # Since we only care about edges to C for the "merge" decision, and edges to I don't exist (I is independent),
        # a vertex v in I is universal to C iff |adj[v] & C_set| == k.
        
        C_set = set(range(k))
        I_vertices = list(range(k, n))
        
        universal_vertices = []
        non_universal_vertices = []
        
        for v in I_vertices:
            # Check if v is connected to all u in C
            # adj[v] is a set of neighbors.
            # We need C_set subset of adj[v]
            if len(adj[v] & C_set) == k:
                universal_vertices.append(v)
            else:
                non_universal_vertices.append(v)
        
        # Strategy:
        # We have a base clique C.
        # We have a set U of universal vertices (connected to all C).
        # We have a set N of non-universal vertices.
        
        # Edges to cover:
        # 1. Internal C edges: Covered by C.
        # 2. Edges between C and U: Since every u in U is connected to all c in C, 
        #    the set C U U is a clique. We can cover all these edges with one clique: C U U.
        # 3. Edges between C and N: For each v in N, the neighbors in C form a subset S_v of C.
        #    The edges are (u, v) for u in S_v.
        #    Since v is not in C, and v is not connected to all C, we cannot just add v to the main clique.
        #    Can we group multiple vertices from N?
        #    If v1, v2 in N are both connected to the same subset S of C, and v1 is NOT connected to v2 (which is true since N subset I and I is independent),
        #    then {u1, ..., um, v1, v2} is NOT a clique because v1-v2 is not an edge.
        #    So we cannot group vertices from N into the same clique as each other.
        #    Can we group a vertex v in N with a subset of C?
        #    Yes, the set S_v U {v} is a clique (since S_v subset C and v connected to all in S_v).
        #    But wait, the edges (u, v) for u in S_v are covered by the clique S_v U {v}.
        #    However, we must ensure that no edge is covered twice.
        #    The edges (u, v) are only covered by cliques containing both u and v.
        #    If we use the big clique C U U, it covers edges between C and U, and internal C.
        #    It does NOT cover edges between C and N, because N is not in C U U.
        #    So for each v in N, we need to cover the edges (u, v) for u in S_v.
        #    The minimal way is to create a clique for each v?
        #    Actually, we can just create one clique per v in N: S_v U {v}.
        #    Size of this clique is |S_v| + 1.
        #    Is it possible to do better?
        #    Suppose we have two vertices v1, v2 in N.
        #    Can we cover edges of v1 and v2 with fewer cliques?
        #    Each clique covering an edge (u, v1) must contain u and v1.
        #    Since v1 is only connected to S_v1 in C, any clique containing v1 can only contain vertices from S_v1 U {v1}.
        #    Similarly for v2.
        #    Since v1 and v2 are not adjacent, they cannot be in the same clique.
        #    Therefore, the cliques covering edges incident to v1 are disjoint from those covering edges incident to v2 (in terms of the "v" slot).
        #    Actually, a clique can contain vertices from C, but it can contain at most one vertex from N (since N is independent).
        #    So, for each v in N, we need a set of cliques that cover all edges (u, v) for u in S_v.
        #    And any such clique must be a subset of S_v U {v}.
        #    To cover all edges (u, v) for u in S_v, we can simply use the single clique S_v U {v}.
        #    This covers all |S_v| edges with 1 clique.
        #    Could we cover them with fewer? No, 1 is the minimum if |S_v| > 0.
        #    If S_v is empty, v is isolated (no edges to C), so 0 cliques needed for v.
        #    
        #    So the total number of cliques is:
        #    1 (for C U U) + sum(1 for v in N if S_v is non-empty)
        #    
        #    Wait, is C U U always a clique?
        #    Yes, C is a clique. U vertices are connected to all C. U vertices are not connected to each other (since they are in I).
        #    Wait! If u1, u2 in U, are they connected?
        #    U subset I. I is an independent set. So u1 is NOT connected to u2.
        #    Therefore, C U U is NOT a clique if |U| > 1!
        #    
        #    Correction:
        #    If u1, u2 in U, edge (u1, u2) does not exist.
        #    So we cannot have a clique containing both u1 and u2.
        #    So we cannot merge multiple universal vertices into one clique with C.
        #    
        #    Let's re-evaluate.
        #    We have C (clique).
        #    We have U (universal to C, independent set).
        #    Edges between C and U: (c, u) for all c in C, u in U.
        #    Can we cover these efficiently?
        #    Any clique containing an edge (c, u) must contain c and u.
        #    It can also contain other c' in C (since C is clique).
        #    It cannot contain another u' in U (since u-u' is non-edge).
        #    So a clique can be of the form S U {u} where S subset C.
        #    To cover all edges (c, u) for a fixed u, we need to cover all c in C.
        #    We can use one clique C U {u}. This covers all k edges (c, u) for c in C.
        #    So for each u in U, we need 1 clique: C U {u}.
        #    Total for U: |U| cliques.
        #    
        #    What about internal C edges?
        #    They are covered by the clique C itself.
        #    Do the cliques C U {u} cover internal C edges?
        #    Yes, C U {u} contains C, so it covers all internal C edges.
        #    So if we use |U| cliques of the form C U {u}, they cover:
        #    - All internal C edges (multiple times, but that's fine? NO. "Every edge must lie in EXACTLY one listed clique".)
        #    
        #    Constraint: "Every edge must lie in exactly one listed clique."
        #    This is a partition of the edge set into cliques.
        #    
        #    So we cannot have overlapping edges.
        #    
        #    Let's refine.
        #    We need to partition E into cliques.
        #    
        #    Case 1: U is empty.
        #    Edges: Internal C, and C-N edges.
        #    We can use C to cover internal C edges.
        #    Then for each v in N with S_v non-empty, we use S_v U {v} to cover edges (u, v).
        #    Do these cliques overlap on edges?
        #    C covers (u, w) for u, w in C.
        #    S_v U {v} covers (u, v) for u in S_v.
        #    These are disjoint edge sets (one is internal C, one is cross C-N).
        #    So this is a valid partition.
        #    Count: 1 (for C) + |{v in N : S_v != empty}|.
        #    
        #    Case 2: U is non-empty.
        #    Edges: Internal C, C-U edges, C-N edges.
        #    We cannot use C alone because C-U edges are not covered by C.
        #    We cannot use C U {u} for all u because they would overlap on internal C edges.
        #    
        #    We need to partition the edges.
        #    Let's look at the structure.
        #    The edges incident to a vertex u in U are only to C.
        #    The edges incident to a vertex v in N are only to C.
        #    The edges incident to vertices in C are to C, U, and N.
        #    
        #    Consider the set of edges incident to U. Let E_U be these edges.
        #    Each edge in E_U is of the form (c, u).
        #    Any clique covering (c, u) must contain c and u.
        #    It cannot contain any other u' in U.
        #    It can contain other c' in C.
        #    So cliques covering E_U are of the form S U {u} where S subset C.
        #    
        #    Similarly for N.
        #    
        #    The internal C edges E_C must be covered by cliques that are subsets of C.
        #    Let's call the cliques covering E_C as "C-cliques".
        #    Let's call the cliques covering E_U as "U-cliques".
        #    Let's call the cliques covering E_N as "N-cliques".
        #    
        #    A U-clique S U {u} contains edges (c, u) for c in S, AND edges (c1, c2) for c1, c2 in S.
        #    So a U-clique also covers some internal C edges!
        #    Specifically, it covers the edges induced by S.
        #    
        #    So we can "steal" internal C edges into U-cliques to avoid using a separate C-clique for those edges.
        #    
        #    Goal: Minimize total number of cliques.
        #    
        #    Let's define the problem for a fixed u in U.
        #    We need to cover all edges (c, u) for c in C.
        #    We can use one clique C U {u}. This covers all (c, u) and all internal edges in C.
        #    If we do this for all u in U, we have |U| cliques.
        #    These cliques cover ALL internal C edges (multiple times).
        #    This violates the "exactly one" rule.
        #    
        #    We must partition the internal C edges among the U-cliques (and potentially a leftover C-clique).
        #    
        #    Let C = {0, ..., k-1}.
        #    We have |U| "slots" (one for each u in U).
        #    We can assign each internal edge (c1, c2) to exactly one u in U.
        #    If we assign edge (c1, c2) to u, then the clique for u must contain c1, c2, and u.
        #    Since the clique for u must also cover all edges (c, u) for c in C, the clique for u is always C U {u} ?
        #    No. The clique for u is determined by the set of internal edges assigned to it?
        #    No, the clique for u must cover ALL edges (c, u). So it MUST contain all c in C.
        #    So the clique for u is ALWAYS C U {u}.
        #    
        #    Wait. If the clique for u is C U {u}, it covers ALL internal edges in C.
        #    If we have multiple u's, each clique C U {u} covers ALL internal edges.
        #    This means every internal edge is covered |U| times.
        #    This is invalid.
        #    
        #    So we cannot use C U {u} for multiple u's.
        #    
        #    We must use smaller cliques for U.
        #    For a fixed u, we can partition the set of neighbors in C (which is all C) into subsets S_1, ..., S_m.
        #    And use cliques S_1 U {u}, ..., S_m U {u}.
        #    This covers all (c, u) edges.
        #    And it covers internal edges within each S_i.
        #    The internal edges NOT within any S_i are not covered by this u's cliques.
        #    
        #    So, for each u in U, we choose a partition of C into subsets.
        #    The union of all internal edges covered by all U-cliques must be a subset of E_C.
        #    The remaining internal edges (if any) must be covered by "pure" C-cliques (subsets of C).
        #    
        #    To minimize the number of cliques:
        #    1. For each u in U, we want to minimize the number of subsets in the partition of C.
        #       The minimum is 1 (partition into {C}). This uses 1 clique C U {u}.
        #       This covers all internal edges in C.
        #       If we do this for one u, all internal edges are covered.
        #       For other u's, we cannot use C U {u} because it would duplicate internal edges.
        #       So for other u's, we must use partitions that cover NO internal edges?
        #       That means the subsets S_i must be singletons.
        #       If S_i = {c}, the clique is {c, u}. It covers edge (c, u) and no internal edges (since size 2 has no internal pairs).
        #       So for a u that doesn't "own" the internal edges, we must use k cliques of size 2: {c, u} for each c in C.
        #    
        #    So the strategy is:
        #    - Pick one u* in U (if U is non-empty) to "own" the internal C edges.
        #      Use clique C U {u*}. Cost: 1 clique. Covers all internal C edges and all (c, u*) edges.
        #    - For each other u in U \ {u*}, use k cliques {c, u}. Cost: k cliques. Covers all (c, u) edges. No internal edges covered.
        #    - Total for U: 1 + (|U|-1)*k.
        #    - Total for N: For each v in N, use 1 clique S_v U {v} (if S_v non-empty). Cost: 1 per v.
        #      These cover (u, v) edges and internal edges in S_v.
        #      Wait, do these overlap with internal edges covered by C U {u*}?
        #      Yes! C U {u*} covers ALL internal edges in C.
        #      S_v U {v} covers internal edges in S_v.
        #      Since S_v subset C, the internal edges in S_v are a subset of internal edges in C.
        #      So they are ALREADY covered by C U {u*}.
        #      This would be a duplicate!
        #      
        #    So we cannot use S_v U {v} if its internal edges are already covered.
        #    We must ensure that the internal edges covered by N-cliques are DISJOINT from those covered by U-cliques (and C-cliques).
        #    
        #    This becomes a complex packing problem.
        #    
        #    Let's simplify.
        #    Since k is small (10, 20, 30), we can potentially do better with a greedy or ILP approach, but time is limited.
        #    
        #    Alternative simple strategy:
        #    1. Cover all internal C edges with one clique C.
        #    2. For each u in U, cover edges (c, u) with k cliques {c, u}.
        #    3. For each v in N, cover edges (c, v) with k_v cliques {c, v} where c in S_v.
        #       (Using singletons avoids internal edge conflicts).
        #    Total: 1 + |U|*k + sum(|S_v| for v in N).
        #    
        #    This is valid (no overlaps).
        #    But is it optimal?
        #    Using C U {u*} saves k-1 cliques for u* (1 instead of k) but "consumes" all internal edges.
        #    If we consume internal edges, we can't use S_v U {v} for N (must use singletons).
        #    If we don't consume internal edges (use singletons for U), we can use S_v U {v} for N.
        #    
        #    Let's compare costs.
        #    Option A (No C-clique, one U-owner):
        #    - 1 clique C U {u*}.
        #    - (|U|-1)*k cliques for other U.
        #    - For N, must use singletons: sum(|S_v|).
        #    Total: 1 + (|U|-1)k + sum(|S_v|).
        #    
        #    Option B (C-clique, no U-owner):
        #    - 1 clique C.
        #    - |U|*k cliques for U.
        #    - For N, can use S_v U {v}: count is |{v: S_v != empty}|.
        #    Total: 1 + |U|k + |N_nonempty|.
        #    
        #    Option C (Hybrid?):
        #    Maybe we can split internal edges?
        #    Given the time, let's implement Option A and Option B and pick the minimum.
        #    Also consider Option D: No C-clique, no U-owner (all singletons for U and N).
        #    Total: sum(|S_v| for v in N) + |U|*k. (Internal edges not covered? No, must be covered. If no C-clique and no U-clique covers them, invalid.)
        #    So internal edges MUST be covered.
        #    
        #    Let's calculate S_v sizes.
        #    S_v is the set of neighbors of v in C.
        #    
        #    Let's code this up.
    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Fallback: return something valid? No, must be valid.
            # If we timeout, we fail. But these are fast.
            pass
            
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C_set = set(range(k))
        I_vertices = list(range(k, n))
        
        U = []
        N = []
        S_N = {}
        
        for v in I_vertices:
            neighbors_in_C = adj[v] & C_set
            if len(neighbors_in_C) == k:
                U.append(v)
            else:
                N.append(v)
                S_N[v] = neighbors_in_C
        
        # Calculate costs for Option A and Option B
        
        # Option A: One U-vertex owns C.
        # Cost = 1 + (len(U)-1)*k + sum(len(S_N[v]) for v in N)
        # Note: If U is empty, we can't do Option A.
        if len(U) > 0:
            cost_A = 1 + (len(U) - 1) * k + sum(len(S_N[v]) for v in N)
        else:
            cost_A = float('inf')
            
        # Option B: C owns internal edges.
        # Cost = 1 + len(U)*k + len([v for v in N if len(S_N[v]) > 0])
        cost_B = 1 + len(U) * k + len([v for v in N if len(S_N[v]) > 0])
        
        # Option C: No C-clique, all U singletons.
        # Internal edges must be covered. If U is empty and we don't use C, who covers internal edges?
        # N-cliques S_v U {v} cover internal edges in S_v.
        # Do they cover ALL internal edges?
        # Only if the union of S_v covers all pairs in C.
        # This is unlikely and hard to check/construct optimally.
        # Let's stick to A and B.
        
        best_cost = min(cost_A, cost_B)
        
        if cost_A < cost_B:
            # Construct Option A
            cliques = []
            u_star = U[0]
            # 1. C U {u_star}
            cliques.append(list(C_set) + [u_star])
            
            # 2. Other U vertices: singletons {c, u}
            for u in U[1:]:
                for c in range(k):
                    cliques.append([c, u])
            
            # 3. N vertices: singletons {c, v} for c in S_N[v]
            for v in N:
                for c in S_N[v]:
                    cliques.append([c, v])
                    
        else:
            # Construct Option B
            cliques = []
            # 1. C
            cliques.append(list(C_set))
            
            # 2. U vertices: singletons {c, u}
            for u in U:
                for c in range(k):
                    cliques.append([c, u])
            
            # 3. N vertices: S_N[v] U {v}
            for v in N:
                if len(S_N[v]) > 0:
                    clique_v = list(S_N[v]) + [v]
                    cliques.append(clique_v)
                    
        # Convert to int
        for i, c in enumerate(cliques):
            cliques[i] = [int(x) for x in c]
            
        results.append(cliques)
        
    return results
