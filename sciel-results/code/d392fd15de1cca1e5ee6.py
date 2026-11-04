"""Matrix multiplication tensor decomposition 4x4 x 4x9."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def check_validity(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        # The tensor T[i,j,k,l] = delta_{j,l} where i in 0..n-1, j in 0..m-1, k in 0..m-1 (input2 row), l in 0..p-1
        # Wait, the standard matrix multiplication tensor for A (n x m) * B (m x p) -> C (n x p)
        # T_{i,k,j} = delta_{...} no.
        # T_{i,k,j} = 1 if the term A_{i,j} * B_{j,k} contributes to C_{i,k}.
        # So T[i, k, j] = 1.
        # In the factorization C_{i,k} = sum_r U_{i, j, r} * V_{j, k, r} * W_{r} ... no.
        # Standard tensor decomposition:
        # T_{i,k,j} = sum_r U_{i,r} V_{k,r} W_{j,r}
        # Here indices are:
        # i: row of A (0..n-1)
        # j: col of A / row of B (0..m-1)
        # k: col of B (0..p-1)
        # T_{i,j,k} = 1 if j is the shared index.
        # So for a specific (i, j, k), the value is 1.
        # The factorization is:
        # sum_r U_{i * m + j, r} * V_{j * p + k, r} * W_{k * n + i, r} = 1 for all i, j, k?
        # No, that's not right.
        # The tensor T has dimensions (n*m, m*p, p*n)? No.
        # The tensor T is of size n x m x m x p?
        # Let's stick to the interface:
        # U shape (n*m, rank)
        # V shape (m*p, rank)
        # W shape (p*n, rank)
        # The decomposition is:
        # For each output entry C_{i,k} (where i in 0..n-1, k in 0..p-1),
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # In the bilinear form:
        # C_{i,k} = sum_r U_{i,m+j? no} ...
        
        # Let's look at the indices in the current check_validity:
        # for i in range(n):
        #   for k in range(p):
        #     for j1 in range(m):
        #       for j2 in range(m):
        #         s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
        #         expected = 1.0 if j1 == j2 else 0.0
        
        # This implies:
        # sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1, j2}
        
        # This is exactly the condition for the tensor to represent matrix multiplication.
        # If this holds, then for any matrices A and B:
        # C_{i,k} = sum_{j1, j2} A_{i,j1} B_{j2,k} (sum_c W_{k,i}^{(c)} U_{i,j1}^{(c)} V_{j2,k}^{(c)})
        # = sum_{j1, j2} A_{i,j1} B_{j2,k} delta_{j1,j2}
        # = sum_j A_{i,j} B_{j,k}
        # which is correct.
        
        # So the check is correct.
        
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(U.shape[1]):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def build_A_for_w(w, U, V):
        """Build the linear system matrix A for a given w index."""
        r = U.shape[1]
        # A has shape (n*m*m*p, r) ? No.
        # We want to solve for W[w, :] such that:
        # sum_c W[w, c] * U[:, c] * V[:, c] (element-wise product, flattened) = b_w
        
        # Let u_c = U[:, c] (size n*m)
        # Let v_c = V[:, c] (size m*p)
        # The product term for a specific w is:
        # sum_c W[w, c] * (u_c * v_c) (where * is outer product flattened? No, element-wise in the tensor sense)
        
        # The condition is:
        # sum_c W[w, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1,j2}
        
        # For a fixed w (which encodes i and k), we fix i and k.
        # i = w % n
        # k = w // n
        
        # The equation is over j1, j2:
        # sum_c W[w, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1,j2}
        
        # This is a linear system in W[w, :].
        # The unknowns are W[w, 0], ..., W[w, r-1].
        # The equations are indexed by (j1, j2).
        # For each (j1, j2), the coefficient of W[w, c] is U[i*m+j1, c] * V[j2*p+k, c].
        
        # So A[j1*m+j2, c] = U[i*m+j1, c] * V[j2*p+k, c]
        # b[j1*m+j2] = 1 if j1==j2 else 0
        
        # Wait, the current code builds A of shape (n*m*m*p, r). That's much larger than m*m.
        # Let's check the current code's build_A_for_w:
        # A = np.zeros((n * m * m * p, r))
        # for c in range(r):
        #     col_u = U[:, c]
        #     col_v = V[:, c]
        #     A[:, c] = np.outer(col_u, col_v).flatten()
        
        # And build_b_for_w:
        # b = np.zeros(n * m * m * p)
        # i_w = w % n
        # k_w = w // n
        # for j in range(m):
        #     u = i_w * m + j
        #     v = j * p + k_w
        #     b[u * (m * p) + v] = 1.0
        
        # This sets b[i_w*m+j, j*p+k_w] = 1.
        # And A[:, c] is the full outer product of U[:,c] and V[:,c].
        # The inner product A @ W[w,:] = b means:
        # sum_c (outer(U[:,c], V[:,c]))[flat_index] * W[w,c] = b[flat_index]
        
        # The flat index for (u, v) is u*(m*p) + v.
        # So sum_c U[u, c] * V[v, c] * W[w, c] = b[u*(m*p)+v]
        
        # In our case, u = i*m+j1, v = j2*p+k.
        # So sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[w, c] = 1 if j1==j2 and u/v match the specific j, else 0.
        
        # The current code sets b[u*(m*p)+v] = 1 only when u = i*m+j and v = j*p+k for the SAME j.
        # So b[(i*m+j)*(m*p) + (j*p+k)] = 1 for j=0..m-1.
        # All other entries are 0.
        
        # This correctly enforces the condition for the specific i, k.
        # The dimension of A is (n*m * m*p, r) = (n*m*m*p, r).
        # This is correct, though sparse.
        
        i_w = w % n
        k_w = w // n
        
        A = np.zeros((n * m * m * p, r))
        # We can be more efficient, but let's stick to the structure for correctness.
        # Actually, we can just build the dense matrix.
        for c in range(r):
            col_u = U[:, c]
            col_v = V[:, c]
            A[:, c] = np.outer(col_u, col_v).flatten()
        return A

    def build_b_for_w(w):
        """Build the RHS vector b for a given w index."""
        b = np.zeros(n * m * m * p)
        i_w = w % n
        k_w = w // n
        for j in range(m):
            u = i_w * m + j
            v = j * p + k_w
            b[u * (m * p) + v] = 1.0
        return b

    def try_rank_with_linear_solve(r, num_restarts=50, seed=42):
        """Try to find a valid decomposition with rank r using linear system for W."""
        rng = np.random.RandomState(seed)
        
        b_vectors = [build_b_for_w(w) for w in range(p * n)]
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                A = build_A_for_w(w, U, V)
                b = b_vectors[w]
                
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None

    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower-rank solutions
    # The current range is 36 to 144.
    # We know that for 4x4x4, the minimum rank is 16 (schoolbook) or 16? No, 4x4x4 min rank is 16?
    # Actually, the min rank for 4x4x4 is 16? No, it's 16? 
    # Wait, the rank of matrix multiplication tensor for n x n x n is known.
    # For n=2, rank=7. For n=3, rank=23. For n=4, rank=49? Or is it 49?
    # Actually, the best known for 4x4x4 is 49? No, that's too high.
    # Schoolbook for 4x4x4 is 64.
    # Laderman's algorithm for 4x4x4 has rank 22? No, that's for 4x4x4?
    # Laderman's algorithm is for 4x4x4? No, it's for 4x4x4?
    # Let's check. Laderman (1976) gave a rank 22 algorithm for 4x4x4? 
    # Actually, the rank of 4x4x4 multiplication is 22? No, I think it's 22?
    # Wait, the rank of 4x4x4 is 22? 
    # Actually, the rank of 4x4x4 is 22? 
    # Let's just say we want to find something better than 144.
    
    # The current code tries r from 36 to 144.
    # We should probably try a wider range or smarter range.
    # But the linear solve is expensive.
    
    # Let's try a few specific ranks.
    for r in range(100, 144):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_with_linear_solve(r, num_restarts=20, seed=42 + r)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
