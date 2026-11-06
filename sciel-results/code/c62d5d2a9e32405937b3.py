import numpy as np
import time

def partition(adjacency):
    adj = np.array(adjacency, dtype=np.float64)
    n = len(adj)
    
    # Deadline guard: 150s limit, use 145s to be safe
    deadline = time.time() + 145.0
    
    # Precompute Laplacian
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    # Compute eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        eigvecs = np.random.randn(n, 10)
        
    # Use first K non-trivial eigenvectors
    K = 10
    if n <= K:
        K = max(1, n - 1)
        
    V = eigvecs[:, 1:K+1]
    
    E = np.sum(adj) / 2.0
    
    best_cut = -1
    best_labels = None
    
    # Number of random hyperplanes to try
    # We can estimate how many fit in time.
    # Each restart is very fast.
    # Let's just run until deadline.
    
    while time.time() < deadline - 1.0:
        # Generate random hyperplane
        r = np.random.randn(K)
        r /= np.linalg.norm(r)
        
        # Project
        projections = V @ r
        x = np.sign(projections)
        x[x == 0] = 1
        
        # Local Search
        # Minimize Q = x^T A x
        # Flip v if x_v * (A x)_v > 0
        
        # Initial Q
        Ax = adj @ x
        Q = x @ Ax
        
        improved = True
        while improved:
            improved = False
            S = Ax
            gains = x * S
            flip_mask = gains > 0
            
            if np.any(flip_mask):
                # Update x
                x[flip_mask] *= -1
                improved = True
                
                # Update Ax efficiently?
                # Ax_new = Ax_old - 2 * (sum of flipped columns?)
                # Actually, Ax = A @ x.
                # If we flip a set F of vertices, x_new = x - 2 * 1_F.
                # Ax_new = A @ (x - 2 * 1_F) = Ax - 2 * A @ 1_F.
                # A @ 1_F is the sum of columns of A indexed by F.
                # This is faster than full mat-vec if |F| is small.
                # But for N=100, full mat-vec is fine.
                Ax = adj @ x
                
                # Update Q
                Q = x @ Ax
        
        cut_val = E - 0.25 * Q
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = (x + 1) // 2
            
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        
    return best_labels
