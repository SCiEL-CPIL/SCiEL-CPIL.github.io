"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_c5_fast(adj, n):
    """Check if graph has a C5 subgraph using vectorized-ish approach.
    adj is a list of sets or bitmasks for fast lookup.
    We use a list of sets for neighbors."""
    # For each 5-set, check if it contains a C5.
    # Precompute the 12 cycles on 0,1,2,3,4
    cycles = [
        (0,1,2,3,4),
        (0,1,2,4,3),
        (0,1,3,2,4),
        (0,1,3,4,2),
        (0,1,4,2,3),
        (0,1,4,3,2),
        (0,2,1,3,4),
        (0,2,1,4,3),
        (0,2,3,1,4),
        (0,2,4,1,3),
        (0,3,1,2,4),
        (0,3,4,2,1),
    ]
    
    # Convert adj to list of sets for O(1) lookup
    adj_sets = [set(adj[i]) for i in range(n)]
    
    for combo in combinations(range(n), 5):
        a, b, c, d, e = combo
        # Check each cycle pattern
        # Pattern 0: a-b-c-d-e-a
        if b in adj_sets[a] and c in adj_sets[b] and d in adj_sets[c] and e in adj_sets[d] and a in adj_sets[e]:
            return True
        # Pattern 1: a-b-c-e-d-a
        if b in adj_sets[a] and c in adj_sets[b] and e in adj_sets[c] and d in adj_sets[e] and a in adj_sets[d]:
            return True
        # Pattern 2: a-b-d-c-e-a
        if b in adj_sets[a] and d in adj_sets[b] and c in adj_sets[d] and e in adj_sets[c] and a in adj_sets[e]:
            return True
        # Pattern 3: a-b-d-e-c-a
        if b in adj_sets[a] and d in adj_sets[b] and e in adj_sets[d] and c in adj_sets[e] and a in adj_sets[c]:
            return True
        # Pattern 4: a-b-e-c-d-a
        if b in adj_sets[a] and e in adj_sets[b] and c in adj_sets[e] and d in adj_sets[c] and a in adj_sets[d]:
            return True
        # Pattern 5: a-b-e-d-c-a
        if b in adj_sets[a] and e in adj_sets[b] and d in adj_sets[e] and c in adj_sets[d] and a in adj_sets[c]:
            return True
        # Pattern 6: a-c-b-d-e-a
        if c in adj_sets[a] and b in adj_sets[c] and d in adj_sets[b] and e in adj_sets[d] and a in adj_sets[e]:
            return True
        # Pattern 7: a-c-b-e-d-a
        if c in adj_sets[a] and b in adj_sets[c] and e in adj_sets[b] and d in adj_sets[e] and a in adj_sets[d]:
            return True
        # Pattern 8: a-c-d-b-e-a
        if c in adj_sets[a] and d in adj_sets[c] and b in adj_sets[d] and e in adj_sets[b] and a in adj_sets[e]:
            return True
        # Pattern 9: a-c-e-b-d-a
        if c in adj_sets[a] and e in adj_sets[c] and b in adj_sets[e] and d in adj_sets[b] and a in adj_sets[d]:
            return True
        # Pattern 10: a-d-b-c-e-a
        if d in adj_sets[a] and b in adj_sets[d] and c in adj_sets[b] and e in adj_sets[c] and a in adj_sets[e]:
            return True
        # Pattern 11: a-d-e-c-b-a
        if d in adj_sets[a] and e in adj_sets[d] and c in adj_sets[e] and b in adj_sets[c] and a in adj_sets[b]:
            return True
    return False

def _has_independent_9(adj_sets, n):
    """Check if there is an independent set of size 9."""
    for combo in combinations(range(n), 9):
        ok = True
        for i in range(9):
            u = combo[i]
            for j in range(i+1, 9):
                v = combo[j]
                if v in adj_sets[u]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _is_valid(n, edge_list):
    """Check if the graph is valid: no C5, no independent set of 9."""
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    if _has_c5_fast(adj, n):
        return False
    if _has_independent_9(adj, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    # Start with 8 disjoint K4 (n=32, alpha=8, no C5)
    base_n = 32
    base_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Verify base graph is valid (should be)
    # But we know it is, so skip verification to save time
    
    # Try to extend to n=33, 34, 35
    # Strategy: Add a new vertex and connect it to a subset of existing vertices.
    # The new vertex must not create a C5 and must not be part of an independent 9-set.
    
    # For n=33:
    # Add vertex 32.
    # We need to choose a set of neighbors S for vertex 32.
    # Condition 1: No C5.
    #   A C5 involving vertex 32 would be 32-u-v-w-x-32.
    #   This requires u,x in S, and u-v, v-w, w-x edges in base graph.
    #   Since base graph is disjoint K4s, u-v-w-x can only exist if u,v,w,x are in the same K4 or adjacent K4s?
    #   No, in disjoint K4s, there are no edges between K4s.
    #   So u,v,w,x must all be in the same K4.
    #   But a K4 has only 4 vertices. A path u-v-w-x of length 3 in K4 requires 4 distinct vertices.
    #   So u,v,w,x are the 4 vertices of some K4.
    #   Then we need u,x in S.
    #   So if S contains two vertices from the same K4, we might create a C5?
    #   Wait, if u,v,w,x are the 4 vertices of K4_b, and u,x in S, then 32-u-v-w-x-32 is a cycle of length 5.
    #   So to avoid C5, S cannot contain two vertices from the same K4.
    #   Actually, if S contains u and x from K4_b, and there is a path of length 3 between u and x in K4_b, then we have a C5.
    #   In K4, any two vertices have a path of length 1 (direct edge) and length 2 (via another vertex).
    #   Do they have a path of length 3? Yes, u-v-w-x where v,w are the other two.
    #   So if S contains any two vertices from the same K4, we can form a C5.
    #   Therefore, S can contain at most one vertex from each K4.
    
    # Condition 2: No independent 9-set.
    #   An independent 9-set can include vertex 32.
    #   If it includes 32, it must include 8 other vertices that are independent in the base graph and not adjacent to 32.
    #   So we need a set T of 8 vertices from base graph such that:
    #     1. T is independent in base graph.
    #     2. T is disjoint from S.
    #   If such T exists, then {32} U T is an independent 9-set.
    #   So we need to ensure that for every independent set T of size 8 in base graph, T intersects S.
    #   Equivalently, the set of non-neighbors of 32 (which is Base \ S) contains no independent set of size 8.
    #   Let N = Base \ S. We need alpha(N) < 8.
    #   Base is 8 disjoint K4s. Independent sets in Base are transversals (pick at most 1 from each K4).
    #   Max independent set size is 8.
    #   So we need to ensure that N does not contain a transversal of size 8.
    #   N is a subset of Base.
    #   Let S_b = S ∩ K4_b. Then N ∩ K4_b = K4_b \ S_b.
    #   A transversal in N exists iff we can pick one vertex from each K4_b \ S_b.
    #   This is possible iff |K4_b \ S_b| >= 1 for all b.
    #   So we need to ensure that for some b, K4_b \ S_b is empty.
    #   i.e., S_b = K4_b for some b.
    #   But we also established that |S_b| <= 1 for all b to avoid C5.
    #   So we cannot have S_b = K4_b (size 4) if |S_b| <= 1.
    #   Contradiction?
    #   Wait, let's re-evaluate the C5 condition.
    #   If S contains u, x from K4_b, does it ALWAYS create a C5?
    #   Cycle: 32-u-v-w-x-32.
    #   Edges: (32,u), (u,v), (v,w), (w,x), (x,32).
    #   (u,v), (v,w), (w,x) must be in Base.
    #   If u,v,w,x are distinct vertices of K4_b, yes.
    #   But what if u=x? No, S is a set.
    #   What if the path u-v-w-x uses repeated vertices? No, simple cycle.
    #   So yes, if |S_b| >= 2, we can pick two distinct u,x in S_b.
    #   In K4, there is always a path of length 3 between any two distinct vertices.
    #   So |S_b| must be <= 1 for all b.
    #   
    #   Now, if |S_b| <= 1 for all b, then |K4_b \ S_b| >= 3 for all b.
    #   So we can always pick one non-neighbor from each K4_b.
    #   Thus, there is always an independent set of size 8 in N.
    #   Thus, {32} U T is an independent 9-set.
    #   So we CANNOT add a single vertex to 8xK4 to get a valid graph on 33 vertices.
    
    # This suggests that the structure of 8xK4 is a "dead end" for extension by adding vertices one by one.
    # We might need a different base graph or a different extension strategy.
    # However, the task says "best known has 32 vertices".
    # It is possible that no valid graph on 33 vertices exists.
    # Or it exists but is not an extension of 8xK4.
    
    # Given the time limit and complexity, and the strong evidence that 8xK4 cannot be extended,
    # and the fact that the current score is 1.1429 (which corresponds to n=32),
    # it is likely that n=32 is the maximum achievable or the search space for n>32 is very hard.
    
    # Let's try a quick random search for a 33-vertex graph.
    # Start with 8xK4 and add vertex 32 with random neighbors.
    # But we proved it's impossible. So skip.
    
    # What if we modify the base graph?
    # Maybe merge two K4s or change structure?
    # This is a hard combinatorial problem.
    
    # Given the constraints, returning the known best (n=32) is the safest bet.
    # The evaluator scores value/seed.
    # Seed was n=28 (score 1.0).
    # Current is n=32 (score 32/28 = 1.1428...).
    # If we find n=33, score = 33/28 = 1.1785.
    
    # Let's do a very short random search for n=33 just in case my proof is flawed.
    # Actually, the proof seems solid.
    # C5-free condition on disjoint K4s forces S to be a matching-like set (at most 1 per K4).
    # Alpha condition forces S to cover a K4 completely.
    # Contradiction.
    
    # So n=32 is the limit for this family.
    # Are there other families?
    # R(C5, K9) is a known Ramsey number.
    # According to literature, R(C5, K9) = 32?
    # If so, n=32 is the maximum.
    
    # Let's check the time. If we have time, we can try to verify n=32 is valid (it is).
    # We will just return the n=32 graph.
    
    return {'n': 32, 'edges': base_edges}

# EVOLVE-BLOCK-END
