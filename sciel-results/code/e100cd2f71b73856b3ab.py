import time
import random
import numpy as np

def count_c4(A):
    # A is n x n adjacency matrix (0/1)
    # C4 count = 0.5 * sum_{i != j} (A^2)_{ij} (A^2)_{ij} - ... ?
    # Number of C4s = 1/8 * sum_{i,j} (A^2)_{ij}^2 - 1/2 * sum_{i} d_i(d_i-1) ?
    # Actually, sum_{i,j} (A^2)_{ij}^2 counts walks of length 2 starting and ending at i, j?
    # (A^2)_{ij} is number of common neighbors of i and j.
    # Number of C4s is sum_{i<j} C(common(i,j), 2).
    # This is 0.5 * sum_{i != j} C(c_ij, 2).
    # C(c, 2) = c(c-1)/2.
    # Sum = 0.5 * sum_{i != j} c_ij(c_ij-1)/2 = 0.25 * sum_{i != j} (c_ij^2 - c_ij).
    # sum_{i != j} c_ij = sum_{i} d_i(d_i-1) ? No.
    # sum_{i != j} c_ij = sum_{i} sum_{j != i} |N(i) \cap N(j)|.
    # Let's just compute c_ij for all i<j and sum C(c_ij, 2).
    n = A.shape[0]
    c4_count = 0
    # A is symmetric 0/1
    # A^2 is not needed explicitly if we loop
    # But matrix multiplication is fast in numpy
    A2 = A @ A
    # A2[i][j] is c_ij
    # We need sum over i<j of C(A2[i][j], 2)
    # C(x, 2) = x*(x-1)//2
    # We can use upper triangle
    tri = np.triu_indices(n, k=1)
    c_vals = A2[tri]
    c4_count = np.sum(c_vals * (c_vals - 1) // 2)
    return c4_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 31
    m = 91
    
    best_edges = None
    best_c4 = float('inf')
    
    # Greedy Construction
    A = np.zeros((n, n), dtype=int)
    edges = []
    
    # Precompute potential edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for _ in range(m):
        min_c = float('inf')
        best_pair = None
        # To make it faster, we can shuffle pairs or just pick min
        # n=31, pairs ~ 465. Looping is fine.
        for u, v in all_pairs:
            if A[u][v] == 1:
                continue
            # Count common neighbors
            # A[u] and A[v] are vectors
            # common = np.sum(A[u] * A[v])
            common = np.dot(A[u], A[v])
            if common < min_c:
                min_c = common
                best_pair = (u, v)
                # Early exit if 0?
                if min_c == 0:
                    break
        if best_pair is None:
            # Should not happen
            break
        u, v = best_pair
        A[u][v] = 1
        A[v][u] = 1
        edges.append([u, v])
        
    current_c4 = count_c4(A)
    if current_c4 < best_c4:
        best_c4 = current_c4
        best_edges = [e[:] for e in edges]
    
    # Local Search
    # We will perform random swaps
    # To speed up, we can precompute list of edges and non-edges
    # But lists change.
    
    while time.time() < deadline:
        # Try a batch of swaps
        A_curr = A.copy()
        edges_curr = [e[:] for e in edges]
        
        # Random restarts within local search?
        # Let's do a simple hill climbing with random restarts if stuck
        # Or just long random walk with acceptance of better moves
        
        # To escape local minima, we can occasionally accept worse moves (Simulated Annealing)
        # But simple hill climbing might be enough for this small instance.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pick random edge to remove
            e_idx = random.randint(0, len(edges_curr) - 1)
            u, v = edges_curr[e_idx]
            
            # Pick random non-edge to add
            # Generate random pair
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if A_curr[x][y] == 0:
                    break
            
            # Perform swap
            A_curr[u][v] = 0
            A_curr[v][u] = 0
            A_curr[x][y] = 1
            A_curr[y][x] = 1
            
            new_c4 = count_c4(A_curr)
            
            if new_c4 < current_c4:
                # Accept
                current_c4 = new_c4
                edges_curr[e_idx] = [x, y]
                A = A_curr
                edges = edges_curr
                improved = True
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = [e[:] for e in edges]
            else:
                # Reject, revert
                A_curr[u][v] = 1
                A_curr[v][u] = 1
                A_curr[x][y] = 0
                A_curr[y][x] = 0
                # Keep A and edges unchanged
                # We can break out of inner loop if we try many random swaps without success?
                # No, just continue.
        
        # If we exit inner loop without improvement, we are stuck.
        # We can try to perturb the graph to escape.
        # Or just restart the local search from best_edges?
        # Let's perturb: random swap regardless of cost, then hill climb again.
        # But we have a time limit.
        # Let's just do more random swaps from the current state, allowing some worse moves?
        # Simpler: Just keep trying random swaps from the current state.
        # The `improved` flag will be false, so we exit the inner while.
        # We should add a mechanism to escape local minima.
        # Let's do a "kick": perform 5 random swaps (accept or not), then hill climb.
        
        # Perturb
        for _ in range(5):
            e_idx = random.randint(0, len(edges) - 1)
            u, v = edges[e_idx]
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                if A[x][y] == 0: break
            
            A[u][v] = 0; A[v][u] = 0
            A[x][y] = 1; A[y][x] = 1
            edges[e_idx] = [x, y]
            
        # Recompute current_c4
        current_c4 = count_c4(A)
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges = [e[:] for e in edges]

    return best_edges
