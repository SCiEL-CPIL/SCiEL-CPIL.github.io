"""Program for matrix multiplication tensor decomposition task."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Strategy: Split into two independent 3x3x3 multiplications.
    Attempt to find a rank-23 decomposition for 3x3x3.
    If found, total rank is 46.
    If not found, fallback to rank 54 (schoolbook).
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 140  # Safety margin
    
    def is_valid_decomposition_full(U, V, W, rank):
        """Check if U,V,W of given rank exactly decompose the multiplication tensor (including zeros)."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check all valid indices (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check invalid indices (should be 0)
        # Invalid if the indices don't form a valid path i->j->k
        # In this specific tensor, ALL i,j,k are valid paths for the multiplication structure?
        # No, the tensor M_{n,m,p} has 1s at (i,j,k) and 0s elsewhere in the basis.
        # The basis is indexed by (i,j), (j',k'), (k'',i'').
        # The entry is 1 if j=j', k=k'', i''=i.
        # So we must check all combinations of (i,j), (j',k'), (k'',i'') where they don't match.
        
        # There are (n*m) * (m*p) * (p*n) total entries.
        # Valid entries are n*m*p.
        # Invalid entries are (n*m)*(m*p)*(p*n) - n*m*p.
        # For 3,3,6: 9 * 18 * 18 = 2916 total. 54 valid. 2862 invalid.
        
        # We can vectorize this check for speed.
        # Construct the full tensor T from U, V, W
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # a = i*m + j
        # b = j'*p + k'
        # c = k''*n + i''
        
        # We need T[a, b, c] == 1 if a,b,c form a valid path, else 0.
        
        # Let's create the target tensor
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    target[a, b, c] = 1.0
        
        # Compute actual tensor
        # T = U @ diag? No.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a 3-mode tensor product.
        
        # Efficient check:
        # For each r, we add outer(U[:,r], V[:,r], W[:,r])
        # But we can just loop over r and accumulate.
        
        actual = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        if not np.allclose(actual, target, atol=1e-9):
            return False
            
        return True

    def find_3x3x3_decomposition(rank=23, seed=42, max_time=60):
        """Find a rank-rank decomposition for 3x3x3 matrix multiplication."""
        start_time = time.time()
        rng = np.random.RandomState(seed)
        n, m, p = 3, 3, 3
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    target[a, b, c] = 1.0
        
        def compute_error(U, V, W):
            # U: (9, rank), V: (9, rank), W: (9, rank)
            actual = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            return np.sum((actual - target)**2)

        # Try random initializations
        for attempt in range(20):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with small random values
            scale = 0.1
            U = rng.randn(n*m, rank) * scale
            V = rng.randn(m*p, rank) * scale
            W = rng.randn(p*n, rank) * scale
            
            # Simple gradient descent
            lr = 0.01
            for iteration in range(2000):
                if time.time() - start_time > max_time:
                    break
                
                # Compute actual tensor
                actual = np.zeros((n*m, m*p, p*n))
                for r in range(rank):
                    actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                error = np.sum((actual - target)**2)
                if error < 1e-12:
                    break
                
                # Compute gradients
                # dE/dU[a, r] = 2 * sum_{b,c} (actual[a,b,c] - target[a,b,c]) * V[b,r] * W[c,r]
                # This is expensive.
                
                # Approximate gradient using finite differences? Too slow.
                # Use analytic gradient.
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                diff = actual - target
                
                for r in range(rank):
                    # grad_U[:, r] = sum_{b,c} diff[:, b, c] * V[b, r] * W[c, r]
                    # This is a contraction.
                    # temp1[b, c] = diff[:, b, c]
                    # grad_U[:, r] = einsum('abc, b, c -> a', diff, V[:, r], W[:, r])
                    grad_U[:, r] = np.einsum('abc,b,c->a', diff, V[:, r], W[:, r])
                    
                    grad_V[:, r] = np.einsum('abc,a,c->b', diff, U[:, r], W[:, r])
                    
                    grad_W[:, r] = np.einsum('abc,a,b->c', diff, U[:, r], V[:, r])
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize to prevent divergence?
                # Optional: project to unit norm?
                
            if compute_error(U, V, W) < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    # Try to find rank-23 decomposition for 3x3x3
    result_3x3 = find_3x3x3_decomposition(rank=23, seed=42, max_time=120)
    
    if result_3x3 is not None:
        U1, V1, W1 = result_3x3
        rank_total = 46
        U_full = [[0] * rank_total for _ in range(9)]
        V_full = [[0] * rank_total for _ in range(18)]
        W_full = [[0] * rank_total for _ in range(18)]
        
        # First 23 columns: for k=0,1,2
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r] = U1[i*3+j][r]
                for j in range(3):
                    for k in range(3):
                        V_full[j*3+k][r] = V1[j*3+k][r]
                    for i in range(3):
                        W_full[k*3+i][r] = W1[k*3+i][r]
        
        # Next 23 columns: for k=3,4,5
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r+23] = U1[i*3+j][r]
                for j in range(3):
                    for k in range(3):
                        V_full[j*3+(k+3)][r+23] = V1[j*3+k][r]
                    for i in range(3):
                        W_full[(k+3)*3+i][r+23] = W1[k*3+i][r]
        
        # Verify full decomposition
        if is_valid_decomposition_full(U_full, V_full, W_full, rank_total):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
