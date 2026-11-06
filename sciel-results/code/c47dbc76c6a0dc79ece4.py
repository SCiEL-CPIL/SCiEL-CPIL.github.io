import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n = 29
    # Generate Paley coloring for n=29
    # Vertices 0..28
    # Edge (i,j) is 0 if (j-i) % 29 is a quadratic residue, else 1
    squares = set()
    for x in range(1, 29):
        squares.add((x * x) % 29)
        
    # colors is a list of length n*(n-1)//2
    # index for (i,j) with i<j:
    # sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # Or simply use a 2D array for internal representation and flatten at the end
    
    # 2D array A where A[i][j] is color of edge (i,j)
    # A[i][j] = A[j][i]
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % n
            if diff in squares:
                A[i][j] = 0
                A[j][i] = 0
            else:
                A[i][j] = 1
                A[j][i] = 1
    # Diagonal is 0, doesn't matter

    # Function to count monochromatic K2,2,2
    # A K2,2,2 in color c exists if there are 3 pairs P1, P2, P3 such that
    # all edges between P1, P2, P3 are color c.
    # This is equivalent to:
    # There exist 6 distinct vertices v1..v6 that can be partitioned into 3 pairs
    # such that the 12 cross-edges are all color c.
    
    # Efficient check:
    # Precompute for each pair of pairs (P1, P2) if all 4 cross edges are color c.
    # Then we need 3 such pairs P1, P2, P3 that are disjoint and mutually "good".
    
    # Let's define "good_pair_pair_c(P1, P2)" as True if all 4 edges between P1 and P2 are color c.
    # We need to find 3 disjoint pairs P1, P2, P3 such that
    # good(P1,P2,c) and good(P1,P3,c) and good(P2,P3,c).
    
    # Number of pairs in K29 is 29*28/2 = 406.
    # Number of pair-pairs is 406*405/2 ~ 82k.
    # For each color c, we can build a graph G_c on the 406 edges (vertices of G_c are edges of K_n).
    # Two vertices in G_c are connected if the corresponding edges in K_n are disjoint and all 4 cross-edges are color c.
    # Then we are looking for a triangle in G_c such that the 3 edges of the triangle in G_c correspond to 3 disjoint edges in K_n.
    # Wait, the vertices of G_c are edges of K_n. A triangle in G_c corresponds to 3 edges of K_n, say e1, e2, e3.
    # The condition "disjoint" must be checked. If e1, e2, e3 are disjoint, then their endpoints form 6 vertices.
    # And the conditions good(e1,e2,c), good(e1,e3,c), good(e2,e3,c) ensure the K2,2,2 structure.
    
    # So, algorithm to count violations:
    # 1. List all edges of K_n.
    # 2. For color 0:
    #    a. Create a graph H0 where nodes are edges of K_n.
    #    b. Connect u,v in H0 if edge u and edge v in K_n are disjoint AND all 4 cross edges are 0.
    #    c. Count triangles in H0 where the 3 nodes are pairwise disjoint edges in K_n.
    #       Actually, if u,v are connected in H0, they are disjoint by definition of the edge in H0.
    #       So any triangle in H0 corresponds to 3 pairwise disjoint edges in K_n.
    #       So we just need to count triangles in H0.
    # 3. Same for color 1.
    # Total violations = Triangles(H0) + Triangles(H1).
    
    # Counting triangles in a graph with ~400 nodes is fast.
    # We can use numpy for adjacency matrix multiplication or just simple loops.
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i,j))
    m = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    def count_violations(A):
        # A is n x n array
        # Build H0 and H1
        # H_c is m x m boolean matrix
        # H_c[u,v] = 1 if edges[u] and edges[v] are disjoint and cross-edges are c
        
        # Precompute disjointness?
        # edges[u] = (a,b), edges[v] = (c,d). Disjoint if len({a,b,c,d}) == 4.
        
        # This might be slow in Python if done naively for every flip.
        # But m=406. m^2 = 165k.
        # Checking disjointness and 4 edge colors for 165k pairs is ~1M ops.
        # This is fast enough for a single check.
        # But we do this in a local search loop.
        
        tri_count = 0
        for c in [0, 1]:
            # Build adjacency list or matrix for H_c
            # To save time, we can just iterate over triples of edges? No, that's m^3 ~ 66M. Too slow.
            # Triangle counting in H_c:
            # Use matrix multiplication?
            # H_c is sparse?
            # Let's build H_c as a numpy array.
            H_c = np.zeros((m, m), dtype=int)
            for u in range(m):
                e_u = edges[u]
                for v in range(u+1, m):
                    e_v = edges[v]
                    # Check disjoint
                    if len(set(e_u + e_v)) == 4:
                        # Check 4 cross edges
                        # e_u = (a,b), e_v = (c,d)
                        # Edges: (a,c), (a,d), (b,c), (b,d)
                        a, b = e_u
                        c1, d1 = e_v
                        if A[a][c1] == c and A[a][d1] == c and A[b][c1] == c and A[b][d1] == c:
                            H_c[u, v] = 1
                            H_c[v, u] = 1
            # Count triangles in H_c
            # Trace(H_c^3) / 6
            # m=406. Matrix mult is 406^3 ~ 66M ops.
            # numpy is fast.
            tri = np.trace(np.dot(np.dot(H_c, H_c), H_c)) // 6
            tri_count += tri
        return tri_count

    # Initial count
    # This might take a second or two.
    current_violations = count_violations(A)
    
    if current_violations == 0:
        # Flatten and return
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(int(A[i][j]))
        return {'n': n, 'colors': colors}
        
    # Local Search
    # We want to minimize current_violations.
    # Flipping one edge in K_n changes H_c significantly.
    # Instead of recalculating all triangles from scratch, we can update.
    # But given m is small, maybe full recalc is okay if we limit iterations?
    # 66M ops for mat mult. Python overhead for the setup loop is 165k * (disjoint check + 4 lookups).
    # 165k * 10 ops = 1.6M ops.
    # Total ~ 70M ops per evaluation.
    # In Python, this might take 0.5 - 1 second per evaluation.
    # We have 110 seconds. So we can do ~100-200 evaluations.
    # This is not enough for a robust local search if we start far from 0.
    
    # Optimization:
    # We can precompute the list of pairs of edges that are disjoint.
    # disjoint_pairs = [(u, v) for u in range(m) for v in range(u+1, m) if disjoint(edges[u], edges[v])]
    # There are ~ 406*405/2 * (prob disjoint) pairs.
    # Prob disjoint is high.
    # Let's precompute disjoint_pairs.
    
    disjoint_pairs = []
    for u in range(m):
        e_u = edges[u]
        for v in range(u+1, m):
            e_v = edges[v]
            if len(set(e_u + e_v)) == 4:
                disjoint_pairs.append((u, v))
                
    def count_violations_fast(A):
        tri_count = 0
        for c in [0, 1]:
            H_c = np.zeros((m, m), dtype=int)
            for u, v in disjoint_pairs:
                e_u = edges[u]
                e_v = edges[v]
                a, b = e_u
                c1, d1 = e_v
                if A[a][c1] == c and A[a][d1] == c and A[b][c1] == c and A[b][d1] == c:
                    H_c[u, v] = 1
                    H_c[v, u] = 1
            tri = np.trace(np.dot(np.dot(H_c, H_c), H_c)) // 6
            tri_count += tri
        return tri_count

    current_violations = count_violations_fast(A)
    
    if current_violations == 0:
        colors = [int(A[i][j]) for i in range(n) for j in range(i+1, n)]
        return {'n': n, 'colors': colors}

    # Local Search
    # Strategy: Flip random edges.
    # Since full recalc is expensive, maybe we can do a smarter update?
    # Or just accept the cost and run as many iterations as possible.
    
    # To speed up triangle counting, we can use the fact that H_c is symmetric.
    # Also, we can stop early if we find 0? No, we need to know if it's 0.
    
    # Let's try to improve the initial state if possible.
    # Maybe Paley 29 is not valid?
    # If it has violations, we need to fix them.
    
    # Heuristic:
    # If we can't get 0 in 100s, we return the best n=29 coloring we have?
    # No, the task requires a VALID coloring.
    # "Neither colour class may contain K2,2,2".
    # So if we don't reach 0 violations, the result is invalid and scores 0.
    # So we MUST reach 0 violations.
    
    # If Paley 29 has violations, we must fix them.
    # If it's impossible to fix, we might need to try a different initial state or n.
    # But n=29 is the record. It is highly likely there is a valid solution.
    
    # Let's add a timeout check.
    
    best_A = A.copy()
    best_violations = current_violations
    
    while time.time() < deadline:
        # Pick a random edge to flip
        u, v = random.choice(edges)
        
        # Flip
        A[u][v] = 1 - A[u][v]
        A[v][u] = A[u][v]
        
        new_violations = count_violations_fast(A)
        
        if new_violations < best_violations:
            best_A = A.copy()
            best_violations = new_violations
            if new_violations == 0:
                break
        else:
            # Revert
            A[u][v] = 1 - A[u][v]
            A[v][u] = A[u][v]
            
    if best_violations == 0:
        colors = [int(best_A[i][j]) for i in range(n) for j in range(i+1, n)]
        return {'n': n, 'colors': colors}
        
    # If we didn't find a solution for n=29, what then?
    # The task says "Maximize n".
    # If n=29 is invalid in our search, maybe n=17 (seed) is safer?
    # But the seed is just a starting point.
    # If we fail to find n=29, we should probably return the seed result to ensure a non-zero score?
    # The seed score is 1.0.
    # If we return an invalid n=29, score is 0.
    # So if best_violations > 0, we should fallback to the seed?
    # Or try n=30? No, n=30 is harder.
    # I will fallback to the seed if n=29 fails.
    
    # Seed: Paley 17
    n_seed = 17
    squares_seed = {x * x % 17 for x in range(1, 17)}
    colors_seed = [0 if (j - i) % 17 in squares_seed else 1 for i in range(17) for j in range(i + 1, 17)]
    return {'n': 17, 'colors': colors_seed}
