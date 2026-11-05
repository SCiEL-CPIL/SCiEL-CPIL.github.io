import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    q = 5
    mod = q
    
    # Enumerate all points in PG(2, q)
    # Points are [x:y:z] with not all zero, modulo scaling by nonzero elements
    # Canonical form: first nonzero coordinate is 1
    points = []
    # Case 1: x != 0, canonical x = 1
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    # Case 2: x = 0, y != 0, canonical y = 1
    for z in range(q):
        points.append((0, 1, z))
    # Case 3: x = 0, y = 0, z != 0, canonical z = 1
    points.append((0, 0, 1))
    
    assert len(points) == q*q + q + 1, f"Expected {q*q+q+1}, got {len(points)}"
    
    # Build adjacency matrix for the polarity graph
    # (x1,y1,z1) ~ (x2,y2,z2) if x1*x2 + y1*y2 + z1*z2 == 0 mod q
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                A[i, j] = 1
                A[j, i] = 1
    
    # Verify edge count
    num_edges = np.sum(A) // 2
    # print(f"Polarity graph has {num_edges} edges")
    
    if num_edges != 90:
        # Fallback if construction failed unexpectedly
        # This should not happen for q=5
        pass

    def count_new_c4s(u, v, A_mat):
        """
        Count the number of 4-cycles created by adding edge (u, v) to the graph defined by A_mat.
        This is the number of paths of length 3 between u and v.
        Path: u - a - b - v.
        a is neighbor of u, b is neighbor of v, and (a, b) is an edge.
        a != b.
        """
        if A_mat[u, v] == 1:
            return 0 # Already an edge
            
        count = 0
        # Get neighbors of u and v
        # Using numpy for speed
        nu = np.where(A_mat[u] == 1)[0]
        nv = np.where(A_mat[v] == 1)[0]
        
        # Iterate over neighbors
        # n is small, so simple loops are fine
        for a in nu:
            for b in nv:
                if a == b:
                    continue
                if A_mat[a, b] == 1:
                    count += 1
        return count

    # 1. Find the best edge to add to the original 90-edge graph
    best_cost = float('inf')
    best_add_edge = None
    
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 0:
                cost = count_new_c4s(i, j, A)
                if cost < best_cost:
                    best_cost = cost
                    best_add_edge = (i, j)
    
    # Current best solution: Add best_add_edge to A
    # Score is best_cost
    current_best_score = best_cost
    current_best_A = A.copy()
    if best_add_edge:
        u, v = best_add_edge
        current_best_A[u, v] = 1
        current_best_A[v, u] = 1

    # 2. Try single-edge swaps to see if we can improve
    # Remove one edge from the polarity graph, add a different edge.
    # Note: The "different edge" can be the one we removed (which just reverts) 
    # or a non-edge of the original graph.
    
    # List of edges in the original polarity graph
    pol_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
    
    # List of non-edges in the original polarity graph
    non_pol_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
    
    # We iterate over each edge to remove
    for e_rem in pol_edges:
        if time.time() > deadline - 10:
            break
            
        u_rem, v_rem = e_rem
        
        # Create a modified adjacency matrix without e_rem
        A_mod = A.copy()
        A_mod[u_rem, v_rem] = 0
        A_mod[v_rem, u_rem] = 0
        
        # Now try adding every possible non-edge to A_mod
        # The non-edges of A_mod are:
        # 1. The edge we removed (e_rem)
        # 2. All original non-edges (non_pol_edges)
        
        # Candidate 1: Re-add e_rem (This is just the original graph, cost 0, but we need 91 edges, so this is invalid as a final state unless we add another edge? No, we remove 1 and add 1. If we add the same one back, we are back to 90 edges. We need 91. So we must add a DIFFERENT edge.)
        # Wait.
        # Original: 90 edges.
        # Swap: Remove e_rem, Add e_add.
        # Result: 90 edges.
        # BUT we need 91 edges.
        # So a single swap (remove 1, add 1) maintains the edge count at 90.
        # We started with 90. We need 91.
        # So we must ADD 1 edge to the 90-edge graph.
        # We CANNOT swap 1 for 1, because that keeps us at 90 edges.
        # We must ADD 1 edge.
        # So the "Swap" logic above is flawed for this specific constraint (90 -> 91).
        # 
        # Let's re-read carefully.
        # "construct() -> exactly 91 distinct edges"
        # "ex(31;C4) = 90"
        # So we start with a 90-edge C4-free graph.
        # We MUST add 1 edge.
        # We CANNOT remove an edge and add one, because that results in 90 edges.
        # We CANNOT remove an edge and add two, because that's 2 operations, and we are looking for the best 91-edge graph.
        # 
        # Is it possible that a 91-edge graph that is NOT a superset of the 90-edge polarity graph is better?
        # Yes.
        # Example: Remove 2 edges from polarity graph, add 3 edges. Total 91.
        # Or Remove 1 edge, add 2 edges. Total 91.
        # 
        # However, the polarity graph is the UNIQUE extremal graph for C4-free.
        # Any 91-edge graph must contain at least one C4.
        # The "cost" of adding an edge to the C4-free graph is the number of C4s it creates.
        # If we modify the core (remove edges), we might reduce the cost of adding new edges.
        # 
        # Let's consider the operation: Remove 1 edge e_rem, Add 2 edges e_add1, e_add2.
        # This results in 91 edges.
        # The number of C4s in the new graph is the number of C4s in (G \ {e_rem} + e_add1 + e_add2).
        # This is complex to evaluate directly.
        # 
        # But notice: The current best is just "Add 1 edge to G".
        # Score = Cost(e_add, G).
        # 
        # If we do "Remove 1, Add 2":
        # New Graph H = G \ {e_rem} \cup {e1, e2}.
        # C4s in H = C4s in (G \ {e_rem}) + C4s involving e1 or e2.
        # Since G \ {e_rem} is a subgraph of G (which is C4-free), G \ {e_rem} is also C4-free.
        # So all C4s in H must involve e1 or e2.
        # 
        # This seems likely to be worse or equal, because we are adding 2 edges instead of 1.
        # Unless the edges we add have very low cost because the removal of e_rem broke many paths.
        # 
        # However, the "Add 1 edge" strategy is the most direct.
        # Given the time limit and the fact that the polarity graph is highly symmetric, the minimum cost edge is likely the global optimum or very close.
        # 
        # Let's check if the "Remove 1, Add 2" strategy can beat "Add 1".
        # If Cost(e_add, G) = K.
        # Can we find e_rem, e1, e2 such that C4s in H < K?
        # 
        # Given the small size, we could search for "Remove 1, Add 2" if we had time.
        # But "Add 1" is O(n^3). "Remove 1, Add 2" is O(n^5) roughly.
        # 31^5 is ~ 28 million. Might be too slow in Python.
        # 
        # Let's stick to the "Add 1 edge" strategy, which is proven to be the bottleneck for ex(n, C4)+1 problems.
        # The inspirations also only do "Add 1 edge".
        
        pass # Skip the swap logic as it doesn't satisfy the edge count constraint directly without adding complexity that might not yield improvement.

    # Final check
    # We have current_best_A which is G + best_add_edge.
    # It has 91 edges.
    
    # Convert to list of edges
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if current_best_A[i, j] == 1:
                final_edges.append([i, j])
                
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    
    return final_edges
