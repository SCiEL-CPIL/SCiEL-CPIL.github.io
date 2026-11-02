import numpy as np
import time
from itertools import product

def construct():
    t0 = time.time()
    deadline = 150.0 - 10.0  # safety margin
    
    n, m, p = 2, 2, 3
    
    # Build the matmul tensor T[i,j,k] = delta[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # T is (n, m, p) tensor where T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to C[i,k]. Actually T[i,j,k] = 1 for all i,j,k since each pair (i,j,k)
    # corresponds to A[i,j]*B[j,k] contributing to C[i,k].
    # Wait, the tensor for matrix multiplication is:
    # T_{i,j,k} such that C[i,k] = sum_j A[i,j] * B[j,k]
    # In the tensor decomposition: C[i,k] = sum_r U[i*m+j... 
    # Actually the standard form is:
    # C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U has shape (n,m,rank), V has shape (m,p,rank), W has shape (n,p,rank)
    # But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # With W[k*n+i, r] convention.
    
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because each term A[i,j]*B[j,k] contributes to C[i,k].
    
    # We want: for all i,k: sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j,r]*V[j*m+k,r]) * W[k*n+i,r]
    # Actually: C[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*m+k, r]
    # This must equal sum_j A[i,j] * B[j,k]
    
    # So we need: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*m+k, r] summed over j 
    # Wait no. Let me re-read.
    
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*m+k, r]) * W[k*n+i, r]
    # This must equal sum_j A[i,j] * B[j,k]
    
    # So for each r, define M_r[i,j] = U[i*m+j, r] and N_r[j,k] = V[j*m+k, r]
    # Then C[i,k] = sum_r W[k*n+i, r] * (M_r * N_r)[i,k]
    # where (M_r * N_r)[i,k] = sum_j M_r[i,j] * N_r[j,k]
    
    # We need: sum_r W[k*n+i, r] * sum_j U[i*m+j,r] * V[j*m+k,r] = sum_j A[i,j]*B[j,k]
    
    # This is a tensor decomposition of the matmul tensor.
    
    # The matmul tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # We need: T[i,j,k] = sum_r U[i*m+j,r] * V[j*m+k,r] * W[k*n+i,r]
    # Wait, that's not right either. Let me think more carefully.
    
    # The bilinear algorithm for matrix multiplication:
    # C = A * B, where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-r decomposition means:
    # C[i,k] = sum_{r=0}^{r-1} W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    
    # Wait no, that's not standard. Let me look at this differently.
    
    # Standard tensor decomposition for matmul:
    # The matmul tensor is T in R^{n x m x p} with T[i,j,k] = 1 if... 
    # Actually T[i,j,k] = 1 for all i,j,k because every combination contributes.
    
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*m+k, r] * W[k*n+i, r]
    
    # Let me verify: if we compute sum_r U[i*m+j,r] * V[j*m+k,r] * W[k*n+i,r]
    # and this equals 1 for all i,j,k, then:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j (sum_r U[i*m+j,r]*V[j*m+k,r]*W[k*n+i,r]) * A[i,j] * B[j,k]
    # Hmm, that doesn't quite work. Let me reconsider.
    
    # Actually the correct formulation is:
    # We want to compute C = A*B using r scalar multiplications.
    # Define intermediate products: P_r = (U_r * A) * (V_r * B) where U_r, V_r are coefficient vectors
    # Then C[i,k] = sum_r W[i,k,r] * P_r[i,k]
    
    # More precisely:
    # P_r[i,k] = (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*m+k, r] * B[j,k])
    # C[i,k] = sum_r W[k*n+i, r] * P_r[i,k]
    
    # Expanding: C[i,k] = sum_r W[k*n+i,r] * sum_j U[i*m+j,r]*A[i,j] * sum_k' V[j'*m+k,r]*B[j',k]
    # = sum_j sum_k' (sum_r U[i*m+j,r] * V[j'*m+k,r] * W[k*n+i,r]) * A[i,j] * B[j',k]
    
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i*m+j,r] * V[j'*m+k,r] * W[k*n+i,r] = delta[j,j'] * delta... 
    # Wait, we need the coefficient of A[i,j]*B[j',k] to be 1 if j=j' and 0 otherwise.
    # But also we need it to be 1 only when the indices match properly.
    
    # Actually: sum_r U[i*m+j,r] * V[j'*m+k,r] * W[k*n+i,r] should equal 
    # 1 if j = j' (the middle index matches) and 0 otherwise.
    # Wait no, that's not right either. Let me be more careful.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The coefficient of A[i,j] * B[j',k] in the expression should be 1 if j=j' and 0 otherwise.
    
    # So we need: sum_r U[i*m+j,r] * V[j'*m+k,r] * W[k*n+i,r] = delta_{j,j'}
    
    # This is the condition! For all i, j, j', k:
    # sum_r U[i*m+j,r] * V[j'*m+k,r] * W[k*n+i,r] = 1 if j=j', 0 otherwise.
    
    # This is a system of equations. The tensor T[i,j,j',k] = delta_{j,j'} needs to be decomposed.
    
    # Actually, this is equivalent to saying the matmul tensor has a certain structure.
    
    # Let me just try to find a rank-10 or rank-11 decomposition by solving the system.
    
    # For rank r, we have:
    # U: (n*m) x r = 4 x r
    # V: (m*p) x r = 6 x r  
    # W: (p*n) x r = 6 x r
    # Total unknowns: 4r + 6r + 6r = 16r
    
    # Constraints: for each (i, j, j', k) with j != j', the sum must be 0.
    # For each (i, j, k) with j=j', the sum must be 1.
    # Number of constraints: n*m*m*p = 2*2*2*3 = 24 (but many are redundant)
    
    # Actually the constraint is: for all i in [0,n), j in [0,m), j' in [0,m), k in [0,p):
    # sum_r U[i*m+j,r] * V[j'*m+k,r] * W[k*n+i,r] = delta_{j,j'}
    
    # That's n*m*m*p = 2*2*2*3 = 24 equations, but they're not all independent.
    
    # Let me try rank 10 first.
    
    def try_rank(r, max_time=10.0):
        """Try to find a rank-r decomposition."""
        t_start = time.time()
        
        # Variables: U (4 x r), V (6 x r), W (6 x r)
        # We'll use a randomized approach with local search.
        
        # Initialize randomly
        np.random.seed(42)
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        n_iter = 1000
        lr = 0.1
        
        for it in range(n_iter):
            if time.time() - t_start > max_time:
                break
            
            if it == 0:
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)
            else:
                U = best_U + np.random.randn(n*m, r) * lr
                V = best_V + np.random.randn(m*p, r) * lr
                W = best_W + np.random.randn(p*n, r) * lr
            
            # Compute error
            # For each (i, j, j', k): sum_r U[i*m+j,r] * V[j'*m+k,r] * W[k*n+i,r] - delta_{j,j'}
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                s += U[i*m+j, rr] * V[jp*m+k, rr] * W[k*n+i, rr]
                            target = 1.0 if j == jp else 0.0
                            err += (s - target) ** 2
            
            if err < best_error:
                best_error = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if best_error < 1e-10:
                    # Found exact solution! Round to nearest integers/half-integers
                    U_r = np.round(best_U * 2) / 2
                    V_r = np.round(best_V * 2) / 2
                    W_r = np.round(best_W * 2) / 2
                    
                    # Verify
                    err2 = 0.0
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    s = 0.0
                                    for rr in range(r):
                                        s += U_r[i*m+j, rr] * V_r[jp*m+k, rr] * W_r[k*n+i, rr]
                                    target = 1.0 if j == jp else 0.0
                                    err2 += (s - target) ** 2
                    if err2 < 1e-10:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        return None
    
    # Try rank 10
    result = try_rank(10, max_time=30.0)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try rank 11 with known Strassen + schoolbook decomposition
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2x2 (columns 0,1 of B)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # Schoolbook for column 2
    U[0][7] = 1
    V[2][7] = 1
    
    U[1][8] = 1
    V[5][8] = 1
    
    U[2][9] = 1
    V[2][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1
    
    # W reconstruction
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    W[2][2] = 1; W[2][4] = 1
    W[4][7] = 1; W[4][8] = 1
    W[5][9] = 1; W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify():
    n, m, p = 2, 2, 3
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    r = U.shape[1]
    
    np.random.seed(123)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for rr in range(r):
                s = 0.0
                for j in range(m):
                    s += U[i*m+j, rr] * V[j*m+k, rr]
                val += W[k*n+i, rr] * s
            C_comp[i, k] = val
    
    if not np.allclose(C_true, C_comp, atol=1e-10):
        raise ValueError("Verification failed")

if __name__ == "__main__":
    _verify()
