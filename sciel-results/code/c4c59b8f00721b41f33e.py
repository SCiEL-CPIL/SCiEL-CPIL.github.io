import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = 0
    best_graph = None
    
    # Helper to check if adding (u, v) creates K3,3
    # adj is a list of bitmasks
    def can_add(adj, u, v):
        # Get common neighbors of u and v
        # Note: u and v are not adjacent yet in the check logic usually, 
        # but we check potential conflict.
        common = adj[u] & adj[v]
        
        # If fewer than 3 common neighbors, no K3,3 can be formed involving u, v as part of the 3-set on one side?
        # Wait. The K3,3 partitions are A and B.
        # If we add (u, v), u and v are in different partitions?
        # No, u and v are adjacent, so they are in the same partition of any bipartite subgraph?
        # No, in a bipartite graph, edges connect A and B.
        # If we add edge (u, v), u is in A, v is in B (or vice versa).
        # A K3,3 requires 3 vertices in A and 3 in B.
        # If u in A, v in B, then we need 2 more in A and 2 more in B.
        # The common neighbors of u and v are candidates for the other side?
        # Let's re-evaluate the condition.
        
        # A K3,3 exists if there are sets A={a1,a2,a3} and B={b1,b2,b3} such that all ai-bj edges exist.
        # Suppose we add edge (u, v).
        # Case 1: u and v are in the same partition (e.g., A).
        #   Then we need 1 more vertex w in A, and 3 vertices in B connected to u, v, w.
        #   So we need w such that N(w) contains 3 vertices that are also in N(u) and N(v).
        #   i.e., |N(u) \cap N(v) \cap N(w)| >= 3.
        # Case 2: u and v are in different partitions (u in A, v in B).
        #   Then we need 2 more vertices in A (a2, a3) and 2 more in B (b2, b3).
        #   This is more complex.
        
        # However, the standard characterization for K_{3,3}-free is:
        # For any 3 vertices x, y, z, |N(x) \cap N(y) \cap N(z)| <= 2.
        # This is equivalent.
        # When adding (u, v), we might violate this if:
        # 1. There exists w such that |N(u) \cap N(v) \cap N(w)| >= 3.
        #    This covers the case where u, v, w are in the same partition.
        # 2. Are there other cases?
        #    If u, v are in different partitions, say u in A, v in B.
        #    We need a2, a3 in A and b2, b3 in B.
        #    The edges (u, b2), (u, b3), (v, a2), (v, a3) etc must exist.
        #    This doesn't immediately imply a triple intersection of neighborhoods.
        #    
        # Let's stick to the triple intersection check:
        # A graph is K_{3,3}-free iff for every triple of vertices {x,y,z}, 
        # the number of common neighbors is at most 2.
        
        # So, when adding (u, v), we must ensure that for any w,
        # the new common neighborhood of {u, v, w} does not become 3.
        # Before adding (u, v), let C = N(u) \cap N(v) \cap N(w).
        # After adding (u, v), u is in N(v) and v is in N(u).
        # Does this add any vertices to N(u) \cap N(v) \cap N(w)?
        # The common neighbors of u, v, w are vertices x such that x is connected to u, v, and w.
        # Adding (u, v) doesn't change N(x). It changes N(u) and N(v).
        # But N(u) \cap N(v) \cap N(w) depends on N(u), N(v), N(w).
        # If x was already connected to u, v, w, it's in the intersection.
        # Adding (u, v) does NOT add any new vertices to the intersection N(u) \cap N(v) \cap N(w).
        # Why? Because the intersection consists of neighbors. The neighbors of u, v, w are determined by edges from u, v, w to x.
        # The edge (u, v) is an edge between u and v. It does not affect whether x is a neighbor of u, v, or w.
        # 
        # Wait. This logic suggests that adding an edge NEVER creates a K_{3,3} via the triple intersection condition?
        # That can't be right.
        
        # Let's re-read the condition.
        # "No three vertices have three common neighbors".
        # Let S = {u, v, w}. Common neighbors are $N(S) = \{x : x \sim u, x \sim v, x \sim w\}$.
        # Adding edge (u, v) changes the graph.
        # Does it change $N(S)$?
        # $x$ is in $N(S)$ if $x$ is connected to $u, v, w$.
        # The connection $x \sim u$ is independent of $u \sim v$.
        # So $N(S)$ is invariant under adding edge (u, v) UNLESS $x$ is u or v?
        # No, $x$ must be a vertex.
        # So, if we add an edge, the set of common neighbors of any triple of OTHER vertices is unchanged.
        # What about triples involving u or v?
        # Let S = {u, v, w}.
        # $N(S) = N(u) \cap N(v) \cap N(w)$.
        # Before adding (u, v), $u \notin N(v)$ and $v \notin N(u)$.
        # After adding (u, v), $u \in N(v)$ and $v \in N(u)$.
        # Does this add u or v to the intersection?
        # $u \in N(u) \cap N(v) \cap N(w)$?
        # $u \in N(u)$ is false (simple graph). So u is never in its own neighborhood.
        # So the set of common neighbors $N(S)$ does NOT change when adding an edge between two vertices in S?
        # 
        # Let's check a counter-example.
        # Start with empty graph. Add edges to form K_{3,3}.
        # If the property is "for all triples, |N(S)| <= 2", and this property is invariant under edge addition (for triples involving the edge endpoints), then we could never violate it?
        # 
        # Ah, the property is not invariant for triples NOT involving the edge endpoints?
        # No, I argued it is invariant for ALL triples.
        # 
        # Let's re-evaluate.
        # If the property "no 3 vertices have 3 common neighbors" is maintained by ANY edge addition, then ANY graph is K_{3,3}-free?
        # That is false. K_{3,3} has 3 vertices in A with 3 common neighbors in B.
        # So the property MUST be violated by some edge addition.
        # Where is my logic flaw?
        # 
        # Flaw: "x is in N(S) if x is connected to u, v, w".
        # This is correct.
        # "Adding (u, v) does not change whether x is connected to u, v, w".
        # This is correct.
        # So N(S) is invariant.
        # 
        # Then how can we form a K_{3,3}?
        # If we build a K_{3,3}, at the end, the triple of A-vertices has 3 common neighbors.
        # So at some point, the count must have gone from 2 to 3.
        # But if N(S) is invariant, it can't change.
        # 
        # Wait. Is it possible that I am checking the wrong condition?
        # "No K_{3,3}" is equivalent to "No 3 vertices have 3 common neighbors"?
        # Yes, if there are 3 vertices with 3 common neighbors, they form a K_{3,3}.
        # Conversely, if there is a K_{3,3}, the 3 vertices in one part have the 3 vertices in the other part as common neighbors.
        # 
        # So if N(S) is invariant under edge addition, we can never create a K_{3,3}?
        # 
        # Let's trace adding an edge to a K_{3,3} minus one edge.
        # Let A={1,2,3}, B={4,5,6}.
        # Edges: all except (1,6).
        # N({1,2,3}) = {4,5}. Size 2.
        # Add edge (1,6).
        # Now N({1,2,3}) = {4,5,6}. Size 3.
        # Did N({1,2,3}) change?
        # Before: 4~1, 4~2, 4~3. 5~1, 5~2, 5~3. 6~2, 6~3 (but not 6~1).
        # So 6 was NOT in N({1,2,3}).
        # After adding (1,6): 6~1, 6~2, 6~3.
        # So 6 IS in N({1,2,3}).
        # 
        # My previous logic: "x is connected to u, v, w".
        # For x=6, S={1,2,3}.
        # Before: 6~1 was False. So 6 not in intersection.
        # After: 6~1 is True. So 6 in intersection.
        # 
        # My flaw was: "Adding (u, v) does not change whether x is connected to u, v, w".
        # If x=6, u=1, v=6? No, u=1, v=6 is the edge added.
        # If the edge added is (u, v), and x is one of u or v?
        # In the example, the edge added was (1,6).
        # The triple was {1,2,3}.
        # x=6 is NOT in the triple {1,2,3}.
        # But x=6 is one of the endpoints of the added edge (1,6).
        # 
        # So, adding edge (u, v) CAN change N(S) if one of the endpoints (u or v) is a common neighbor of S.
        # Specifically, if $u \in N(S)$ or $v \in N(S)$ before the addition?
        # No, $u$ becomes connected to $v$.
        # If $v \in S$, then $u$ becomes a neighbor of $v$ (which is in S).
        # If $u \in S$, then $v$ becomes a neighbor of $u$ (which is in S).
        # If neither $u$ nor $v$ is in S, does N(S) change?
        # N(S) = $\{x : x \sim s \forall s \in S\}$.
        # If $x$ is not $u$ or $v$, its connections to $s \in S$ are unchanged.
        # So only $u$ and $v$ can enter or leave N(S).
        # 
        # So, when adding (u, v):
        # 1. For any triple S not containing u or v, N(S) is unchanged.
        # 2. For any triple S containing u (but not v), N(S) might gain v if v is connected to all other elements in S.
        # 3. For any triple S containing v (but not u), N(S) might gain u if u is connected to all other elements in S.
        # 4. For any triple S containing both u and v, N(S) might gain... u? No, u not in N(u). v? No.
        #    So N(S) for S containing both u and v is unchanged?
        #    Wait. If S={u, v, w}, N(S) = N(u) \cap N(v) \cap N(w).
        #    Adding (u, v) makes u \in N(v) and v \in N(u).
        #    But u \notin N(u) and v \notin N(v).
        #    So u and v cannot be in N(S).
        #    Other x are unaffected.
        #    So N(S) is unchanged for S containing both u and v.
        
        # Therefore, the only risk is:
        # - Adding v to N(S) where S contains u (and 2 others).
        # - Adding u to N(S) where S contains v (and 2 others).
        
        # So the check is:
        # Let S be a triple containing u. Let S = {u, a, b}.
        # v enters N(S) if v is connected to a and b.
        # So we need to check if v is connected to a and b.
        # If v is connected to a and b, then after adding (u, v), v becomes a common neighbor of u, a, b.
        # We must ensure that BEFORE adding, |N({u, a, b})| <= 1?
        # No, we need |N({u, a, b})| after addition <= 2.
        # If v enters, the size increases by 1.
        # So we need |N({u, a, b})| before addition <= 1.
        # Similarly for u entering N({v, a, b}).
        
        # So the algorithm for can_add(u, v):
        # 1. For each pair (a, b) such that a, b != u, v:
        #    Check if v is connected to a and b.
        #    If yes, check current size of N({u, a, b}).
        #    If size >= 2, adding v makes it 3 -> Invalid.
        # 2. For each pair (a, b) such that a, b != u, v:
        #    Check if u is connected to a and b.
        #    If yes, check current size of N({v, a, b}).
        #    If size >= 2, adding u makes it 3 -> Invalid.
        
        # This is O(n^2) per edge check.
        # n=20, so 400 checks. Very fast.
        
        # Implementation:
        # adj is list of bitmasks.
        # to check if v connected to a and b: (adj[v] >> a) & 1 and (adj[v] >> b) & 1.
        # to get N({u, a, b}): adj[u] & adj[a] & adj[b].
        # popcount of that.
        
        pass

    # We will implement this check.
    
    def check_valid(adj, u, v):
        # u, v are the endpoints of the edge to add.
        # We need to ensure no triple gets a 3rd common neighbor.
        
        # Case 1: v becomes a common neighbor of some triple {u, a, b}
        # This happens if v is adjacent to a and b.
        # We need |N({u, a, b})| < 2 (i.e., 0 or 1) before adding.
        # If |N({u, a, b})| >= 2, then adding v makes it >= 3.
        
        # Iterate over all pairs a, b from V \ {u, v}
        for a in range(n):
            if a == u or a == v: continue
            for b in range(a + 1, n):
                if b == u or b == v: continue
                
                # Check if v is connected to a and b
                if (adj[v] >> a) & 1 and (adj[v] >> b) & 1:
                    # v is a candidate common neighbor for {u, a, b}
                    common = adj[u] & adj[a] & adj[b]
                    # Count bits
                    if common.bit_count() >= 2:
                        return False
        
        # Case 2: u becomes a common neighbor of some triple {v, a, b}
        for a in range(n):
            if a == u or a == v: continue
            for b in range(a + 1, n):
                if b == u or b == v: continue
                
                # Check if u is connected to a and b
                if (adj[u] >> a) & 1 and (adj[u] >> b) & 1:
                    common = adj[v] & adj[a] & adj[b]
                    if common.bit_count() >= 2:
                        return False
        
        return True

    # Greedy construction with multiple restarts
    max_restarts = 5000
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        adj = [0] * n
        edges_in_graph = 0
        
        for u, v in all_edges:
            if check_valid(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_in_graph += 1
                
        if edges_in_graph > best_edges:
            best_edges = edges_in_graph
            best_graph = [adj[:]] # store copy
            
        # Optional: Local search
        # If we have time, try to add more edges by removing some?
        # Given n=20, greedy is likely close to optimal.
        
    # Reconstruct edge list from best_graph
    if best_graph is None:
        return []
        
    final_adj = best_graph[0]
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (final_adj[i] >> j) & 1:
                res.append([i, j])
                
    return res
