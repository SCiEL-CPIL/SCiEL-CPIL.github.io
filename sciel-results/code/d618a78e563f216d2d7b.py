import time
import numpy as np
import random

def construct():
    n = 40
    deadline = 115.0  # Safety margin from 120s limit
    start_time = time.time()
    
    # Initialize adjacency matrix
    A = np.zeros((n, n), dtype=np.int64)
    
    # Helper to check if adding edge (u, v) creates a C6
    # A C6 is formed if there is a path of length 4 between u and v in the current graph.
    # This is equivalent to (A^4)[u, v] > 0.
    # Computing A^4 for every edge check is expensive (O(n^3) or O(n^6) naive).
    # Instead, we can check if there exists w1, w2, w3, w4 such that 
    # u-w1-w2-w3-w4-v is a path.
    # A faster check: 
    # Count paths of length 4 between u and v.
    # Let M = A. We want to know if (M^4)[u,v] > 0.
    # We can compute M^2 once? No, M changes.
    
    # Alternative efficient check for C6:
    # Adding (u,v) creates a C6 if there is a path u-x-y-z-v of length 4.
    # This is equivalent to: sum_{x,y,z} A[u,x] * A[x,y] * A[y,z] * A[z,v] > 0.
    # Let's precompute nothing, but use vectorization.
    # For a candidate edge (u,v):
    # 1. Get neighbors of u: N_u
    # 2. Get neighbors of v: N_v
    # 3. We need a path u-x-y-z-v.
    #    x in N_u, z in N_v.
    #    We need y such that x-y and y-z are edges.
    #    So we need to check if there exists x in N_u, z in N_v, such that 
    #    there is a path x-y-z of length 2.
    #    This is equivalent to: (A^2)[x, z] > 0 for some x in N_u, z in N_v.
    #    So: Does N_u intersect with (A^2) * N_v? 
    #    Specifically, compute B = A @ A. Then check if any B[x, z] > 0 where x in N_u and z in N_v.
    #    This is: max(B[N_u, :][:, N_v]) > 0.
    #    Complexity: O(deg(u) * deg(v)) for the check, plus O(n^3) for A^2 update?
    #    Updating A^2 is hard.
    
    # Let's stick to a simpler, robust local search approach that worked in p5 (score 1.2).
    # Strategy: Start with K5s. Try to add random edges. If C6 created, remove a random edge from the cycle or reject.
    # Actually, p5 used "compute_power_5". Let's look at the diff. It used matrix powers.
    # A^5[u,v] > 0 means there is a path of length 5. Adding (u,v) creates a C6 if there is a path of length 5 between u and v.
    # So check (A^5)[u,v] > 0.
    
    # To make this fast, we can maintain A2 = A@A, A3 = A2@A, etc?
    # Or just compute A^5 on the fly for candidate edges? 
    # n=40 is small. A^5 is 40x40. 
    # Computing A^5 takes ~5 matrix multiplications of 40x40. Very fast.
    # But we do it for every candidate edge.
    
    # Better: Precompute A5 = np.linalg.matrix_power(A, 5) initially.
    # When we add an edge, A changes. We must recompute A5.
    # Recomputing A5 for n=40 is negligible (< 1ms).
    # So: 
    # 1. Start with K5s.
    # 2. Compute A5.
    # 3. Loop:
    #    a. Pick random non-edge (u,v).
    #    b. Check if A5[u,v] > 0.
    #    c. If A5[u,v] == 0: Add edge. Update A. Recompute A5.
    #    d. If A5[u,v] > 0: Skip.
    #    e. If time limit, break.
    #    f. Periodically, try to remove an edge and add two others? (2-opt style)
    #       Or just try random edge swaps: remove (u,v), try to add (x,y).
    
    # Let's implement the greedy add + random restart/swap.
    
    def compute_A5(mat):
        # Efficient power for small matrix
        # Use standard multiplication
        return np.linalg.matrix_power(mat, 5)
    
    # Initialize
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                A[u, v] = 1
                A[v, u] = 1
                
    A5 = compute_A5(A)
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edge_count = np.sum(A) // 2
    best_A = A.copy()
    
    while time.time() < deadline:
        # Shuffle remaining non-edges or just pick random pairs
        # To be efficient, maintain a list of non-edges? 
        # n=40, total edges 780. 
        # Just pick random u, v.
        
        improved = False
        # Try a batch of random additions
        for _ in range(50):
            if time.time() > deadline:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or A[u, v] == 1:
                continue
            
            # Check if adding (u,v) creates C6
            # Path of length 5 between u and v?
            if A5[u, v] == 0:
                # Safe to add
                A[u, v] = 1
                A[v, u] = 1
                A5 = compute_A5(A)
                improved = True
                best_edge_count = np.sum(A) // 2
                best_A = A.copy()
            
            # If A5[u,v] > 0, we can't add.
        
        if not improved:
            # Try edge swaps: remove one, add one?
            # Or random restart?
            # Let's try a simple swap: pick an edge (u,v), remove it, then try to add random edges until we regain count or fail.
            # This is complex. 
            # Alternative: Randomize the order of "try to add" to escape local minima?
            # Actually, since we only add, we are stuck in a local maximum of the greedy addition.
            # We need to remove edges to allow better additions.
            
            # Strategy: Pick a random edge, remove it. Then try to add random edges.
            # If we end up with more edges, keep it. Else revert.
            
            # Get a random edge
            edges_in_graph = np.argwhere(A > 0)
            if len(edges_in_graph) == 0:
                break
            idx = random.randint(0, len(edges_in_graph) - 1)
            u, v = edges_in_graph[idx]
            if u > v:
                u, v = v, u
            
            # Save state
            A_old = A.copy()
            A5_old = A5.copy()
            
            # Remove edge
            A[u, v] = 0
            A[v, u] = 0
            A5 = compute_A5(A)
            
            # Try to add edges aggressively
            # We want to add at least 1 edge to be net positive, or 2 to be net +1.
            # Let's try to add 2 edges.
            added = 0
            attempts = 0
            max_attempts = 100
            while added < 2 and attempts < max_attempts and time.time() < deadline:
                attempts += 1
                nu = random.randint(0, n - 1)
                nv = random.randint(0, n - 1)
                if nu == nv or A[nu, nv] == 1:
                    continue
                if A5[nu, nv] == 0:
                    A[nu, nv] = 1
                    A[nv, nu] = 1
                    A5 = compute_A5(A)
                    added += 1
            
            new_count = np.sum(A) // 2
            if new_count > best_edge_count:
                best_edge_count = new_count
                best_A = A.copy()
            else:
                # Revert if not better
                A = A_old
                A5 = A5_old

    # Return best found
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
    return edges

# Ensure function name matches
# The task says: Keep the function names and signatures of the seed program.
# Seed: def construct():
