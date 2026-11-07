"""Strategy: Global set-partitioning via core decomposition and min-cost flow assignment."""

def construct():
    import time
    import random
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_cost_flow

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve(n, k, p, seed, deadline):
        rng = random.Random(seed)
        
        # 1. Build graph
        # Core vertices: 0..k-1
        # Independent vertices: k..n-1
        adj_core = [set() for _ in range(k)]
        for u in range(k):
            for v in range(u + 1, k):
                adj_core[u].add(v)
                adj_core[v].add(u)
                
        adj_cross = [set() for _ in range(n)]
        for i in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj_cross[i].add(u)
                    adj_core[u].add(i) # Not needed for core logic, but for checking
                    # Actually, we just need to know which core vertices connect to i
                    # Let's store neighbors of i in the core
                    pass
        
        # Re-do cross adjacency properly
        core_neighbors_of_i = [set() for _ in range(n)]
        for i in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    core_neighbors_of_i[i].add(u)

        # 2. Decompose Core into disjoint cliques
        # We want to partition the set of core vertices {0..k-1} into disjoint cliques.
        # This is equivalent to finding a clique partition of the complete graph K_k.
        # Wait, the core IS a clique. So any subset is a clique.
        # The constraint is that we want to cover the *edges* of the core with these cliques?
        # No, the task says: "every edge must lie in exactly one listed clique".
        # The core edges must be covered. Since the core is a clique, if we partition the core vertices
        # into sets C_1, C_2, ..., C_m, the edges within each C_j are covered by clique C_j.
        # But what about edges between C_j and C_l? They are core edges too!
        # If we use clique C_j, it covers edges (u,v) where u,v in C_j.
        # It does NOT cover edges (u,v) where u in C_j, v in C_l.
        # So a simple partition of vertices into cliques does NOT cover all core edges unless m=1.
        # If m=1, we have one clique of size k. It covers all core edges.
        # But then we can't attach any I-vertex to it unless the I-vertex is connected to ALL core vertices.
        
        # Let's re-read the requirement: "Every listed set must be a clique... and every edge must lie in exactly one listed clique."
        # This is a clique partition (edge partition).
        
        # Strategy Revision:
        # We are partitioning the edge set E(G) into cliques.
        # For split graphs, a known optimal structure often involves:
        # 1. A "large" clique consisting of the core vertices that are "universal" or highly connected.
        # 2. For each I-vertex i, if N(i) is not a subset of the "large" clique, we might need to form cliques involving i.
        
        # Let's look at the structure of an optimal solution for split graphs.
        # If we take the entire core C as one clique, we cover all core edges. Cost: 1 clique.
        # Then for each i in I, we need to cover edges (i, u) for u in N(i).
        # If N(i) is empty, no cliques needed for i.
        # If N(i) is not empty, we can form cliques {i} U S where S is a subset of N(i).
        # Since I is an independent set, any clique containing i can contain at most one vertex from I (which is i).
        # So any clique containing i is of the form {i} U S, where S is a clique in C (which is always true).
        # So we just need to cover the set of edges { (i, u) : u in N(i) } using cliques of the form {i} U S.
        # Each such clique covers |S| edges incident to i.
        # To minimize the number of cliques for i, we want to cover N(i) with as few cliques as possible?
        # No, the cliques are just sets. If we pick S = N(i), we use 1 clique {i} U N(i) and cover all edges incident to i.
        # Is {i} U N(i) a clique? Yes, because N(i) is a subset of C, and C is a clique.
        # So, if we use the whole core C as one clique, and for each i with N(i) non-empty we use {i} U N(i), 
        # do we have conflicts?
        # "Every edge must lie in exactly one listed clique."
        # Core edges are covered by C.
        # Cross edges (i, u) are covered by {i} U N(i).
        # Are any edges covered twice?
        # Core edges are only in C.
        # Cross edges are only in {i} U N(i) for their specific i.
        # So this is a valid partition!
        # Number of cliques = 1 (for C) + |{i in I : N(i) != empty}|.
        
        # Can we do better?
        # What if we don't use the whole core C?
        # Suppose we split C into C1 and C2.
        # Then edges within C1 are covered by C1, edges within C2 by C2.
        # Edges between C1 and C2 are NOT covered by C1 or C2.
        # We would need additional cliques to cover the edges between C1 and C2.
        # Since C is a clique, the edges between C1 and C2 form a complete bipartite graph K_{|C1|, |C2|}.
        # To partition the edges of K_{a,b} into cliques, we need at least max(a,b) cliques?
        # Actually, a clique in a bipartite graph has size at most 2. So we need ab cliques of size 2?
        # No, a clique in the original graph G must be a set of mutually adjacent vertices.
        # In the subgraph induced by C1 U C2, any clique can have at most one vertex from C1 and one from C2?
        # No, vertices in C1 are adjacent to each other. Vertices in C2 are adjacent to each other.
        # Vertices in C1 are adjacent to vertices in C2.
        # So C1 U C2 is a clique!
        # So if we split C into C1 and C2, we could just use C1 U C2 as a clique?
        # But then we are back to C.
        
        # The only way to "save" cliques is if we can merge the "i-cliques" with the core cliques?
        # No, an i-clique {i} U S contains i. The core clique C does not contain i.
        # They are disjoint sets of vertices? No, they share S.
        # But the condition is on EDGES.
        # Edge (i, u) is in {i} U S.
        # Edge (u, v) with u,v in S is in C.
        # These are distinct edges.
        
        # So the count is 1 + sum_{i in I, N(i)!=empty} 1.
        # Is it possible to reduce this?
        # What if N(i) is small?
        # If N(i) = {u}, we use clique {i, u}.
        # If N(i) = {u, v}, we use clique {i, u, v}.
        # In both cases, 1 clique.
        # So for each i with N(i) != empty, we pay 1.
        
        # Total cliques = 1 + |{i : N(i) != empty}|.
        
        # Can we avoid the "1" for the core?
        # If we don't use C, we must cover core edges some other way.
        # If we cover core edges using 2-cliques, we need k(k-1)/2 cliques. That's worse.
        # If we cover core edges using larger cliques, we still need to cover all edges.
        # Since C is a clique, any clique partition of C's edges requires at least 1 clique (the whole C).
        # So the "1" is fixed if we want to cover core edges efficiently.
        
        # So the problem reduces to: Minimize |{i : N(i) != empty}|?
        # No, we can't change the graph. The graph is fixed.
        # So the number of cliques is exactly 1 + |{i : N(i) != empty}|?
        
        # Wait, is it possible to cover cross edges in a way that shares cliques?
        # No, because each cross edge (i, u) involves a unique i.
        # A clique containing (i, u) must contain i.
        # Since I is independent, a clique can contain at most one i.
        # So cliques covering cross edges are disjoint in terms of I-vertices.
        # So each i requires at least one clique if N(i) != empty.
        
        # So the minimum number of cliques is exactly 1 + |{i : N(i) != empty}|.
        
        # Let's verify this logic.
        # Instance 1: n=30, k=10.
        # If this logic holds, the score should be very predictable.
        # Let's calculate |{i : N(i) != empty}| for the seed.
        
        # If this is true, the current program (score 1.6183) is likely NOT achieving this, 
        # or the current program is doing something else.
        # Let's check the current program's behavior.
        # It merges universal neighbors.
        # If the current program is stuck at 1.6183, and the theoretical minimum is 1 + count,
        # maybe the current program is producing MORE cliques than necessary?
        
        # Let's test the hypothesis: Optimal = 1 + count_non_empty_neighbors.
        
        # Let's implement this simple strategy and see.
        
        count_non_empty = 0
        for i in range(k, n):
            if core_neighbors_of_i[i]:
                count_non_empty += 1
        
        cliques = []
        
        # 1. Add the core clique
        cliques.append(list(range(k)))
        
        # 2. For each i with non-empty neighbors, add {i} U N(i)
        for i in range(k, n):
            if core_neighbors_of_i[i]:
                S = list(core_neighbors_of_i[i])
                cliques.append([i] + S)
        
        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 100

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = _solve(n, k, p, seed, deadline)
        results.append(cliques)

    return results
