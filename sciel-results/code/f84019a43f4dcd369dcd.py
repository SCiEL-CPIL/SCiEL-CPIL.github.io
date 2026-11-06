"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 31
    target_edges = 91
    
    # Build the polarity graph of PG(2,5) - a C4-free graph with 31 vertices and 90 edges
    # PG(2,5) has 31 points. We use the standard construction.
    # Points are represented as equivalence classes of (x,y,z) over GF(5)
    # The polarity: two points are adjacent if they are "orthogonal" under a symmetric bilinear form
    
    # Generate all points of PG(2,5)
    def pg2_points(q):
        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero coordinate 1
                    idx = 0
                    while idx < 3 and (x, y, z)[idx] == 0:
                        idx += 1
                    factor = (x, y, z)[idx]
                    inv_factor = pow(factor, q - 2, q)  # modular inverse
                    norm = tuple((x, y, z)[i] * inv_factor % q for i in range(3))
                    if norm not in seen:
                        seen.add(norm)
                        points.append(norm)
        return points
    
    points = pg2_points(5)
    assert len(points) == 31
    
    # Define the polarity: use the symmetric bilinear form
    # B((x1,y1,z1), (x2,y2,z2)) = x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 + z1*x2 + z2*x1 (mod 5)
    # Actually, let's use a standard polarity for PG(2,5)
    # The standard one: B(p,q) = x_p*x_q + y_p*y_q + z_p*z_q + x_p*y_q + y_p*x_q (mod 5)
    # Let me use: B(p,q) = x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 + z1*x2 + z2*x1 (mod 5)
    # Simpler: use the Hermitian-like form B(p,q) = x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 (mod 5)
    
    # Actually, for a polarity graph to be C4-free with q=5, we need:
    # B(p,q) = 0 means adjacent
    # The standard construction uses: B((x1,y1,z1),(x2,y2,z2)) = x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 (mod 5)
    
    def bilinear(p, q_point, q_mod=5):
        x1, y1, z1 = p
        x2, y2, z2 = q_point
        return (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2) % q_mod
    
    # Build adjacency: p ~ q if B(p,q) = 0 and p != q
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if bilinear(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edge_count = int(adj.sum() // 2)
    
    # If we don't have exactly 90 edges, try different bilinear forms
    # Try various forms to find one giving 90 edges
    forms = [
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 + z1*x2 + z2*x1) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 + z1*y2 + z2*y1) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + 2*x1*y2 + 2*y1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 + x1*z2 + z1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + 3*x1*y2 + 3*y1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*z2 + z1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + y1*z2 + z1*y2) % 5,
    ]
    
    best_adj = None
    best_edge_count = -1
    
    for form in forms:
        adj_temp = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2, p3 = points[i]
                q1, q2, q3 = points[j]
                if form(p1, p2, p3, q1, q2, q3) == 0:
                    adj_temp[i, j] = 1
                    adj_temp[j, i] = 1
        ec = int(adj_temp.sum() // 2)
        if ec == 90:
            best_adj = adj_temp
            best_edge_count = 90
            break
        elif ec > best_edge_count and ec <= 91:
            best_adj = adj_temp
            best_edge_count = ec
    
    if best_adj is None:
        # Fallback: use a random graph with 91 edges
        best_adj = np.zeros((n, n), dtype=np.int8)
        random.seed(42)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for i, j in all_pairs[:91]:
            best_adj[i, j] = 1
            best_adj[j, i] = 1
        best_edge_count = 91
    else:
        adj = best_adj
    
    # Count C4s in a graph
    def count_c4(adj_matrix):
        # A 4-cycle (a,b,c,d) is counted by: for each pair (a,c), count common neighbors
        # Number of C4s = sum over all pairs (a,c) of C(common_neighbors(a,c), 2)
        # But this counts each C4 four times (once for each diagonal)
        # Actually, each C4 has 2 diagonals, and for each diagonal, the two endpoints have 2 common neighbors (the other two vertices)
        # So: C4 count = (1/4) * sum_{a<c} C(common_neighbors(a,c), 2)
        # Wait, let me think again.
        # For a C4: vertices a-b-c-d-a. Diagonals are (a,c) and (b,d).
        # common_neighbors(a,c) = {b, d}, so C(2,2) = 1
        # common_neighbors(b,d) = {a, c}, so C(2,2) = 1
        # Total sum = 2, and we divide by 4? No...
        # Actually each C4 contributes exactly 2 to the sum (one for each diagonal pair having 2 common neighbors).
        # So C4 count = sum_{a<c} C(common_neighbors(a,c), 2) / 2
        
        A = adj_matrix.astype(np.int32)
        # common neighbors of i and j = (A @ A)[i,j] - A[i,j] (subtract the direct edge if present)
        # Actually, (A@A)[i,j] counts walks of length 2 from i to j, which includes the direct edge path i->k->j where k is any neighbor
        # common_neighbors = (A @ A)[i, j] - A[i, j]  (subtract if i and j are directly connected, since that's not a common neighbor path)
        # Wait no. (A@A)[i,j] = sum_k A[i,k]*A[k,j]. If i and j are adjacent, then k can be i or j? No, A[i,i]=0.
        # (A@A)[i,j] counts the number of vertices k such that both (i,k) and (k,j) are edges.
        # If i and j are adjacent, k=i gives A[i,i]*A[i,j]=0, k=j gives A[i,j]*A[j,j]=0.
        # So (A@A)[i,j] is exactly the number of common neighbors, regardless of whether i,j are adjacent.
        
        A2 = A @ A
        # For each pair (i,j), i<j, compute C(common_neighbors, 2)
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                cn = A2[i, j]
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 2
    
    # Current C4 count
    current_c4 = count_c4(adj)
    
    # Now we need exactly 91 edges. If we have 90, add one edge that minimizes C4 increase.
    if best_edge_count < 91:
        # Find the non-edge whose addition minimizes C4 count
        best_add = None
        best_c4_after = float('inf')
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    # Adding edge (i,j): how many new C4s?
                    # A new C4 containing edge (i,j) must be i-j-k-l-i or i-j-l-k-i
                    # This is a C4 where (i,j) is an edge.
                    # New C4s = number of paths of length 3 from i to j that don't use edge (i,j)
                    # = common_neighbors of i and j... no.
                    # A C4 containing edge (i,j): i-j-a-b-i where a != i,b, b != i,j,a
                    # So we need a neighbor a of j (a != i) and a neighbor b of i (b != j) such that a-b is an edge.
                    # Count = sum over a in N(j)\{i} of |N(a) ∩ N(i)| \ {j}
                    # Let's just compute C4 count after adding the edge
                    adj_temp = adj.copy()
                    adj_temp[i, j] = 1
                    adj_temp[j, i] = 1
                    c4_temp = count_c4(adj_temp)
                    if c4_temp < best_c4_after:
                        best_c4_after = c4_temp
                        best_add = (i, j)
        
        if best_add is not None:
            i, j = best_add
            adj[i, j] = 1
            adj[j, i] = 1
            current_c4 = best_c4_after
        else:
            # Should not happen
            i, j = 0, 1
            adj[i, j] = 1
            adj[j, i] = 1
            current_c4 = count_c4(adj)
    
    elif best_edge_count > 91:
        # Need to remove edges to get to 91
        while int(adj.sum() // 2) > 91:
            best_remove = None
            best_c4_after = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 1:
                        adj_temp = adj.copy()
                        adj_temp[i, j] = 0
                        adj_temp[j, i] = 0
                        c4_temp = count_c4(adj_temp)
                        if c4_temp < best_c4_after:
                            best_c4_after = c4_temp
                            best_remove = (i, j)
            if best_remove:
                i, j = best_remove
                adj[i, j] = 0
                adj[j, i] = 0
                current_c4 = best_c4_after
            else:
                break
    
    # Local search: swap an existing edge for a non-edge to reduce C4 count
    # Keep track of best solution found
    best_adj = adj.copy()
    best_c4 = current_c4
    
    # Precompute degree and other useful info for speed
    def count_c4_fast(adj_matrix):
        A = adj_matrix.astype(np.int32)
        A2 = A @ A
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                cn = A2[i, j]
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 2
    
    # Local search with swaps
    # A swap: remove edge (i,j) and add edge (k,l) where (k,l) is a non-edge
    # This keeps the edge count constant at 91
    
    # To speed up, we can estimate the C4 change
    # Removing edge (i,j) removes all C4s that contain edge (i,j)
    # Adding edge (k,l) adds all C4s that contain edge (k,l)
    
    def c4s_containing_edge(adj_matrix, i, j):
        """Count C4s that contain the specific edge (i,j)."""
        # A C4 containing edge (i,j): i-j-a-b-i
        # a is a neighbor of j (a != i), b is a neighbor of i (b != j), and a-b is an edge
        count = 0
        for a in range(n):
            if a == i or a == j:
                continue
            if adj_matrix[j, a] == 1:
                for b in range(n):
                    if b == i or b == j or b == a:
                        continue
                    if adj_matrix[i, b] == 1 and adj_matrix[a, b] == 1:
                        count += 1
        # Each C4 containing edge (i,j) is counted once (a and b are determined)
        # Wait, i-j-a-b-i: a is next to j, b is next to i, and a-b is edge.
        # This uniquely identifies the C4. So count is correct.
        return count
    
    # Local search
    random.seed(12345)
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try a random swap
        # Pick a random existing edge to remove
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 1]
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
        
        if not edges or not non_edges:
            break
        
        # Try several random swaps
        improved = False
        for _ in range(50):
            i, j = random.choice(edges)
            k, l = random.choice(non_edges)
            
            # Check if this swap is valid (no duplicate edges)
            if (k == i or k == j or l == i or l == j) and not (set([k,l]) == set([i,j])):
                # This might create issues, let's be careful
                pass
            
            # Compute C4 change
            removed_c4 = c4s_containing_edge(adj, i, j)
            added_c4 = c4s_containing_edge(adj, k, l)  # approximate: before adding, count C4s that would form
            
            # But adding edge (k,l) might create C4s that also involve edge (i,j) being removed
            # For simplicity, use the full count
            adj_temp = adj.copy()
            adj_temp[i, j] = 0
            adj_temp[j, i] = 0
            adj_temp[k, l] = 1
            adj_temp[l, k] = 1
            
            # Verify edge count is still 91
            if int(adj_temp.sum() // 2) != 91:
                continue
            
            c4_temp = count_c4_fast(adj_temp)
            
            if c4_temp < best_c4:
                adj = adj_temp
                best_adj = adj.copy()
                best_c4 = c4_temp
                no_improve = 0
                improved = True
                # Recompute edges and non_edges
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 1]
                non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
                break
            else:
                no_improve += 1
        
        if improved:
            continue
        
        # If no improvement in this batch, try a few more structured moves
        if no_improve % 100 == 0 and time.time() < deadline:
            # Try targeted swaps: remove an edge that's in many C4s, add a non-edge that's in few C4s
            edge_c4_counts = {}
            for i, j in edges:
                if time.time() > deadline:
                    break
                edge_c4_counts[(i, j)] = c4s_containing_edge(adj, i, j)
            
            if edge_c4_counts:
                # Find the edge in the most C4s
                worst_edge = max(edge_c4_counts, key=edge_c4_counts.get)
                
                # Find non-edges with low C4 potential
                non_edge_c4_counts = {}
                for k, l in non_edges[:200]:  # limit for speed
                    if time.time() > deadline:
                        break
                    non_edge_c4_counts[(k, l)] = c4s_containing_edge(adj, k, l)
                
                if non_edge_c4_counts:
                    best_non_edge = min(non_edge_c4_counts, key=non_edge_c4_counts.get)
                    
                    i, j = worst_edge
                    k, l = best_non_edge
                    
                    adj_temp = adj.copy()
                    adj_temp[i, j] = 0
                    adj_temp[j, i] = 0
                    adj_temp[k, l] = 1
                    adj_temp[l, k] = 1
                    
                    if int(adj_temp.sum() // 2) == 91:
                        c4_temp = count_c4_fast(adj_temp)
                        if c4_temp < best_c4:
                            adj = adj_temp
                            best_adj = adj.copy()
                            best_c4 = c4_temp
                            no_improve = 0
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                result.append([i, j])
    
    # Ensure exactly 91 edges
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    
    return result
