import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    n = 20
    
    # Precompute all disjoint triples for exact K3,3 check
    # A K3,3 is formed by two disjoint sets A, B of size 3 such that all edges between them exist.
    # We precompute all pairs of disjoint triples (A, B).
    # There are C(20, 3) * C(17, 3) / 2? No, order of A and B doesn't matter for the subgraph, 
    # but for counting violations, we can just check all pairs (A, B) where A and B are disjoint.
    # Actually, it's easier to just iterate over all pairs of triples.
    # Total triples: 1140. Pairs of disjoint triples is large, but we can optimize.
    
    # Alternative exact check:
    # A graph contains K3,3 iff there exist 6 distinct vertices v1..v6 such that 
    # edges (v1,v4), (v1,v5), (v1,v6), (v2,v4), (v2,v5), (v2,v6), (v3,v4), (v3,v5), (v3,v6) all exist.
    # This is equivalent to: For any partition of 6 vertices into two sets of 3, say A and B,
    # if the subgraph induced by A U B is a complete bipartite graph K3,3 between A and B, we have a violation.
    
    # We can precompute all such "K3,3 patterns".
    # Number of ways to choose 6 vertices: C(20,6) = 38760.
    # For each set of 6, there are C(6,3)/2 = 10 ways to split into two sets of 3.
    # So ~387,600 patterns.
    # Checking each pattern against the adjacency matrix takes 9 edge checks.
    # Total ops per full check: ~3.5M.
    # In Python, this is too slow for 120s if we do many steps.
    # We need a faster check or a smarter search.
    
    # Let's use the property:
    # G is K3,3-free iff for every pair of vertices u, v, the number of common neighbours is not "too high" in a specific way?
    # No, the "max common neighbours <= 2" is sufficient but not necessary.
    # Example: A 5-cycle is K3,3-free. Max common neighbours is 1.
    # Example: Complete bipartite K2,18 is K3,3-free. Max common neighbours is 2 (for the two vertices in the part of size 2).
    # Wait, in K2,18, the two vertices in the small part have 18 common neighbours.
    # So "max common neighbours <= 2" is FALSE for K2,18.
    # My previous analysis in the prompt was flawed!
    # "If u and v have 3 common neighbours x, y, z, then {u, v} plus x, y, z doesn't form K3,3 directly."
    # Correct. K2,18 has 36 edges. The greedy with "max common neighbours <= 2" would produce a much sparser graph, likely around 20-30 edges.
    # The current program score is 1.6389. If seed is K2,18 (36 edges), then 1.6389 * 36 = 59 edges?
    # Or is the seed something else?
    # "The evaluator measures the raw value... search score is value/seed".
    # If seed is 36, and score is 1.63, value is ~59.
    # Max edges in K3,3-free graph on 20 vertices?
    # ex(20, K3,3) is known to be around 50-60?
    # Let's check known values. ex(n, K3,3) ~ 1/2 n^2? No, K3,3 has 6 vertices, 9 edges.
    # Turan number T(n, 3) is ~ n^2/3.
    # K3,3 is bipartite.
    # Known: ex(n, K3,3) <= 1/2 n^2 for large n?
    # For n=20, max edges is likely around 50-60.
    
    # The current program uses "max common neighbours <= 2".
    # This constraint forbids K2,18!
    # In K2,18, the two vertices in the part of size 2 have 18 common neighbours.
    # So the current program CANNOT find K2,18 if it starts from empty and uses that constraint.
    # It starts with K2,18 as baseline.
    # Then it tries to improve.
    # If the baseline is 36, and it finds 59, that's a big jump.
    
    # My new strategy: Simulated Annealing with EXACT K3,3 check.
    # To make the exact check fast, we can use a heuristic check first, or optimize the exact check.
    # Actually, we can use the following necessary condition for K3,3:
    # If G contains K3,3, then there exist 3 vertices u, v, w such that |N(u) ∩ N(v) ∩ N(w)| >= 3?
    # No.
    
    # Let's stick to a robust local search with exact checking, but optimized.
    # We can check for K3,3 by iterating over all pairs of vertices (u, v) and looking at common neighbours.
    # If |N(u) ∩ N(v)| >= 3, let C = N(u) ∩ N(v).
    # Pick any 3 from C, say c1, c2, c3.
    # If we can find another vertex w such that w is connected to u, v, and at least one of c1, c2, c3? No.
    
    # Let's use the definition:
    # K3,3 exists iff there are two disjoint sets A, B of size 3 such that all edges between A and B are present.
    
    # We will use a "conflict" counter.
    # For each pair of disjoint triples (A, B), count the number of missing edges between A and B.
    # If any count is 0, we have a K3,3.
    
    # To speed up, we can precompute the list of all (A, B) pairs.
    # There are C(20,3) * C(17,3) / 2? No, A and B are distinct sets.
    # Number of ways to choose A: 1140.
    # Number of ways to choose B from remaining 17: 680.
    # Total pairs: 1140 * 680 = 775,200.
    # This is too many to check every step in pure Python.
    
    # We need a faster necessary condition or a smarter check.
    
    # Observation:
    # If G is K3,3-free, then for any 3 vertices u, v, w, the number of common neighbours is not necessarily small.
    # But if we look at the complement?
    
    # Let's try a different approach:
    # Start with a dense random graph.
    # Use Simulated Annealing.
    # Move: Add or remove a random edge.
    # Cost: Number of K3,3 subgraphs.
    # To estimate Cost fast, we can sample.
    # Or, we can use the fact that n=20 is small.
    
    # Let's try to optimize the exact check in Python using bitmasks.
    # Represent adjacency as a list of 20-bit integers.
    # For a pair (u, v), common neighbours mask = adj[u] & adj[v].
    # If popcount(common) >= 3, we have candidates.
    # But we need to check if ANY 3 of them form a K3,3 with u, v and some other 3?
    # No, K3,3 partitions are A and B.
    # If u, v are in A, we need 3 common neighbours in B.
    # So if |N(u) ∩ N(v)| >= 3, it does NOT imply K3,3.
    # It implies that u, v and any 3 common neighbours form a K2,3.
    # To form K3,3, we need a third vertex w in A such that w is also connected to those 3 in B.
    # So, if |N(u) ∩ N(v) ∩ N(w)| >= 3 for some w, then we have a K3,3?
    # Yes! If u, v, w are in A, and they share 3 common neighbours b1, b2, b3 in B, then {u,v,w} x {b1,b2,b3} is a K3,3.
    
    # So the condition for K3,3 is:
    # There exist distinct u, v, w such that |N(u) ∩ N(v) ∩ N(w)| >= 3.
    
    # Let's verify.
    # If G contains K3,3 with partitions A={a1,a2,a3} and B={b1,b2,b3}.
    # Then N(a1) ∩ N(a2) ∩ N(a3) contains {b1, b2, b3}.
    # So |N(a1) ∩ N(a2) ∩ N(a3)| >= 3.
    
    # Conversely, if |N(u) ∩ N(v) ∩ N(w)| >= 3, let the common neighbours be x, y, z.
    # Then u, v, w are connected to x, y, z.
    # So {u, v, w} and {x, y, z} form a K3,3.
    # Note: We must ensure {u,v,w} and {x,y,z} are disjoint.
    # Since x, y, z are neighbours of u, v, w, and the graph is simple (no loops), x, y, z are distinct from u, v, w.
    # Are x, y, z distinct from each other? Yes, they are a set of 3.
    # Are u, v, w distinct? Yes, we pick distinct vertices.
    # So the condition is:
    # G is K3,3-free iff for all distinct u, v, w, |N(u) ∩ N(v) ∩ N(w)| <= 2.
    
    # This is MUCH faster to check!
    # Number of triples (u, v, w): C(20, 3) = 1140.
    # For each triple, compute intersection of 3 bitmasks.
    # 1140 operations per check. Very fast.
    
    # Let's implement Simulated Annealing with this check.
    
    def count_k33(adj_masks):
        """
        adj_masks: list of 20 integers, each is a 20-bit mask.
        Returns the number of triples (u, v, w) with |N(u) ∩ N(v) ∩ N(w)| >= 3.
        """
        violations = 0
        # Iterate over all triples u < v < w
        for u in range(n):
            for v in range(u + 1, n):
                inter_uv = adj_masks[u] & adj_masks[v]
                if inter_uv == 0:
                    continue
                for w in range(v + 1, n):
                    inter_uvw = inter_uv & adj_masks[w]
                    # Count bits in inter_uvw
                    # If count >= 3, it's a violation.
                    # We can just check if inter_uvw has at least 3 bits set.
                    # A fast way: (x & (x-1)) etc.
                    # Or precomputed popcount?
                    # Python int.bit_count() is available in 3.8+?
                    # Let's use bin(x).count('1') or x.bit_count()
                    if inter_uvw.bit_count() >= 3:
                        violations += 1
        return violations

    # Initial state: Random dense graph
    # Or start with K2,18?
    # K2,18 is K3,3-free.
    # Let's start with a random graph with p=0.5 and then anneal.
    # Or start with the best known seed?
    # The prompt says "Keep the function names and signatures of the seed program."
    # The seed program returns a list of edges.
    
    # Let's use Simulated Annealing to maximize edges subject to violations == 0.
    # Energy = -edges + lambda * violations.
    # We want to minimize Energy.
    # Or simply: 
    # If move reduces violations, accept if it increases edges or keeps same.
    # If move increases violations, accept with probability exp(-delta_E / T).
    
    # Better: 
    # Objective: Maximize edges.
    # Constraint: violations == 0.
    # Penalty method: Score = edges - M * violations.
    # Maximize Score.
    
    def get_edges_from_masks(adj_masks):
        edges = []
        for u in range(n):
            mask = adj_masks[u]
            for v in range(u + 1, n):
                if mask & (1 << v):
                    edges.append([u, v])
        return edges

    # Initial best
    best_adj = [0] * n
    # Start with K2,18
    for u in range(2):
        for v in range(2, n):
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
    
    best_edges = get_edges_from_masks(best_adj)
    best_score = len(best_edges)
    
    # Simulated Annealing
    # State: adj_masks
    # Move: Flip a random edge (u, v)
    
    current_adj = best_adj[:]
    current_violations = count_k33(current_adj)
    current_edges = len(get_edges_from_masks(current_adj))
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline:
        # Random move
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip edge (u, v)
        bit = 1 << v
        if current_adj[u] & bit:
            # Remove edge
            current_adj[u] ^= bit
            current_adj[v] ^= (1 << u)
            edge_change = -1
        else:
            # Add edge
            current_adj[u] |= bit
            current_adj[v] |= (1 << u)
            edge_change = 1
        
        new_edges = current_edges + edge_change
        new_violations = count_k33(current_adj)
        
        # Score: edges - 1000 * violations
        old_score = current_edges - 1000 * current_violations
        new_score = new_edges - 1000 * new_violations
        
        delta = new_score - old_score
        
        if delta > 0:
            accept = True
        else:
            # Boltzmann probability
            if T > 0:
                accept = random.random() < np.exp(delta / T)
            else:
                accept = False
        
        if accept:
            current_edges = new_edges
            current_violations = new_violations
            
            if current_violations == 0 and current_edges > best_score:
                best_score = current_edges
                best_adj = current_adj[:]
        else:
            # Revert move
            if edge_change == -1:
                # We removed, so we add back
                current_adj[u] |= bit
                current_adj[v] |= (1 << u)
            else:
                # We added, so we remove
                current_adj[u] ^= bit
                current_adj[v] ^= (1 << u)
        
        T *= alpha
    
    # Final check
    if current_violations == 0 and current_edges > best_score:
        best_score = current_edges
        best_adj = current_adj[:]
        
    return get_edges_from_masks(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
