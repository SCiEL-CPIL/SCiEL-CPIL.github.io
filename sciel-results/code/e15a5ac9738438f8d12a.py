import numpy as np

def construct():
    n = 31
    target_edges = 91
    
    # Field F_5
    # Squares in F_5: {0, 1, 4}
    # Non-squares: {2, 3}
    
    # Generate all points in PG(2, 5)
    # A point is an equivalence class of non-zero vectors in F_5^3 under scalar multiplication.
    # We represent each point by its canonical representative:
    # The first non-zero coordinate is 1.
    # If the first non-zero is at index i, then coord[i] = 1, and coords before i are 0.
    # If all coords are 0, skip (zero vector).
    
    points = []
    for z in range(5):
        for y in range(5):
            for x in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical form
                if x != 0:
                    # x is already 1..4, we want it to be 1.
                    # Actually, we can just iterate over all and normalize.
                    pass
                
    # Better: iterate over all (x,y,z) in F_5^3 \ {0}, normalize, and use a set to avoid duplicates.
    # Normalization: find first non-zero component, divide all by it.
    seen = set()
    pts = []
    for z in range(5):
        for y in range(5):
            for x in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Find first non-zero
                if x != 0:
                    # Scale so x becomes 1
                    inv_x = pow(x, -1, 5)
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                    rep = (nx, ny, nz)
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                    rep = (nx, ny, nz)
                else: # z != 0
                    inv_z = pow(z, -1, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                    rep = (nx, ny, nz)
                
                if rep not in seen:
                    seen.add(rep)
                    pts.append(rep)
    
    # There should be (5^3 - 1) / (5 - 1) = 124 / 4 = 31 points.
    assert len(pts) == 31, f"Expected 31 points, got {len(pts)}"
    
    # Assign indices 0..30
    # We can just use the order in `pts`.
    # To be deterministic, let's sort them or just keep the generation order.
    # The generation order is z, y, x loops.
    
    # Define a polarity.
    # For q odd, a standard polarity is given by a symmetric matrix M.
    # If M is symmetric, the polarity is a correlation.
    # We want a C4-free graph.
    # Polarity graphs of PG(2,q) are C4-free.
    # Let's use the standard "orthogonal" polarity or a similar one.
    # For q=5, let's try M = [[1, 0, 0], [0, 1, 0], [0, 0, 2]].
    # Check if this works.
    # Edge (u, v) if u^T M v = 0.
    # Note: If u^T M u = 0, u is absolute (isotropic). We ignore self-loops.
    
    M = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 2]], dtype=int)
    
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            u = np.array(pts[i])
            v = np.array(pts[j])
            val = (u @ M @ v) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    edges = int(np.sum(A) / 2)
    
    # Check edge count
    # For q=5, ex(31; C4) = 90.
    # Polarity graphs typically have q(q+1)/2 + q/2 + ... edges?
    # Number of edges in a polarity graph of PG(2,q) is roughly q(q+1)/2 + q/2.
    # For q=5: 5*6/2 + 5/2? No.
    # Let's just check the count.
    
    # If edges > 91, remove edges with most 4-cycles.
    # If edges < 91, add edges with fewest 4-cycles.
    # If edges == 91, done.
    
    def count_4_cycles(A):
        common = A @ A
        iu = np.triu_indices(n, k=1)
        vals = common[iu]
        return int(np.sum(vals * (vals - 1) / 2))
    
    def get_edge_4_cycle_counts(A):
        counts = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    N_u = np.where(A[u] == 1)[0]
                    N_v = np.where(A[v] == 1)[0]
                    N_u = N_u[N_u != v]
                    N_v = N_v[N_v != u]
                    if len(N_u) > 0 and len(N_v) > 0:
                        sub = A[np.ix_(N_u, N_v)]
                        cnt = int(np.sum(sub))
                        counts[u, v] = cnt
                        counts[v, u] = cnt
        return counts

    def remove_edge_most_4_cycles(A):
        counts = get_edge_4_cycle_counts(A)
        max_val = np.max(counts)
        if max_val == 0:
            return None
        idx = np.unravel_index(np.argmax(counts), counts.shape)
        u, v = idx
        A[u, v] = 0
        A[v, u] = 0
        return (u, v)

    def add_edge_min_4_cycles(A):
        best_u, best_v = -1, -1
        min_cyc = float('inf')
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    common = int(np.dot(A[u], A[v]))
                    new_cyc = common * (common - 1) // 2
                    if new_cyc < min_cyc:
                        min_cyc = new_cyc
                        best_u, best_v = u, v
        if best_u == -1:
            return None
        A[best_u, best_v] = 1
        A[best_v, best_u] = 1
        return (best_u, best_v)

    # Adjust edge count to 91
    # It is possible the polarity graph has exactly 90, 91, or more.
    # For q=5, let's see.
    # If it has 90, we add 1.
    # If it has 91, we are done.
    # If it has >91, we remove.
    
    # Let's run the adjustment loop.
    # Note: The polarity graph is C4-free, so count_4_cycles should be 0 initially.
    # Removing edges keeps it C4-free.
    # Adding edges might introduce 4-cycles.
    
    while edges > target_edges:
        res = remove_edge_most_4_cycles(A)
        if res is None:
            break
        edges -= 1
        
    while edges < target_edges:
        res = add_edge_min_4_cycles(A)
        if res is None:
            break
        edges += 1
        
    # Extract edges
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if A[u, v]:
                edges_list.append([u, v])
                
    return edges_list
