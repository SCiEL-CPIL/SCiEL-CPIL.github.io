"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] ... 
    # Actually, the convention is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T_{i,j,k} where we need: sum_c U[i,j,c] V[j,k,c] = delta_{i,k} ... 
    # No, let me re-read the convention.
    
    # W[k*n+i, r] - so W is indexed by (k, i) for the output C[i,k]
    # The decomposition: for each r, we compute a product term
    # sum_r U[i*m+j, r] * V[j*p+k, r] should give something related to W
    
    # Actually, the standard bilinear decomposition of matrix mult:
    # C = A*B, where C[i,k] = sum_j A[i,j] B[j,k]
    # We write: sum_r (sum_j U_{i,j}^{(r)} A[i,j]) * (sum_k V_{j,k}^{(r)} B[j,k]) = sum_k W_{i,k}^{(r)} C[i,k]
    # So: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] is NOT quite right.
    
    # Let me re-derive. The tensor for matrix multiplication n×m, m×p is:
    # T[i, j, k] = 1 (it's a 3-tensor with indices i in [0,n), j in [0,m), k in [0,p))
    # The bilinear form: <T, A ⊗ B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # A rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # And the condition is: sum_r u_r[i*m+j] * v_r[j*p+k] = w_r[k*n+i] for the "delta" structure
    
    # More precisely, for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (in the appropriate sense)
    # Rank decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[i,k]
    # And we need: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for the correct structure
    
    # Wait, that's not right either. Let me think again.
    
    # The matrix multiplication tensor T^{n,m,p} is defined such that:
    # <T, A, B> = tr(C^T) where C = AB, i.e., sum_{i,k} C[i,k]
    # But actually, each output entry is separate. The full tensor for computing ALL outputs is:
    # T[i, j, k] where i indexes rows of A/C, j indexes the contraction, k indexes cols of B/C
    # T[i,j,k] = 1 means A[i,j] * B[j,k] contributes to C[i,k]
    
    # So the tensor has a 4-index structure if we separate input and output, but for bilinear rank:
    # We want: sum_r (sum_j U_r[i,j] A[i,j]) (sum_k V_r[j,k] B[j,k]) ... no.
    
    # Standard form: The bilinear complexity is the minimum r such that:
    # C[i,k] = sum_{r=1}^r sum_j (sum_j u_{r,i,j} A[i,j]) (v_{r,k} * something)
    
    # Let me use the concrete verification:
    # Given U (n*m × r), V (m*p × r), W (p*n × r)
    # The decomposition works if for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = 1 if this corresponds to a valid term, but actually:
    
    # The correct condition: The tensor T[i,j,k] = 1 for all (i,j,k) in the multiplication
    # T = sum_r u_r ⊗ v_r ⊗ w_r where:
    # u_r has indices (i,j), v_r has indices (j,k), w_r has indices (i,k) [or (k,i) per convention]
    # And T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    
    # But wait, T[i,j,k] should be 1 for the matrix multiplication tensor? No.
    # The matrix multiplication tensor is: T[i,j,k] = delta (it's always 1, meaning A[i,j]*B[j,k] contributes to C[i,k])
    # So we need: 1 = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] for all i,j,k? That can't be right.
    
    # I think the correct interpretation is:
    # The bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # Can be written as: C = sum_r (sum_j U_r[:,j] * A[:,j]) outer (sum_k V_r[k,:] * B[k,:]) * W_r
    # Hmm, this is getting confusing. Let me just verify numerically.
    
    # Test: for random A (n×m) and B (m×p), compute C = A@B
    # Then check if the decomposition reproduces C.
    
    # The decomposition says: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k2 V[j*p+k2, r] * B[j,k2]) ... 
    # No, that's not dimensionally correct.
    
    # Let me look at the seed code:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # So for each (i,j,k), we have a rank-1 term with U, V, W all being 1 at those positions.
    
    # The reconstruction should be:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j2 V[j2*p+k, r] * B[j2,k])
    # No, that's a trilinear form, not bilinear.
    
    # I think the correct formula is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k'])
    # But that has j free on the right side, which is wrong.
    
    # Let me just test the seed numerically to understand the convention.
    n, m, p = 2, 3, 8
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Test with random matrices
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Try: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j2 V[j2*p+k, r] * B[j2,k])
    # That's trilinear, skip.
    
    # Try: The intermediate product for term r is: P_r = (U_r^T A) * (V_r^T B) where U_r is n*m vector, V_r is m*p vector
    # But that gives a scalar, and we need C[i,k] for each i,k.
    
    # I think: for each r, define:
    # a_r = U_r reshaped to (n,m), so a_r[i,j] = U[i*m+j, r]
    # b_r = V_r reshaped to (m,p), so b_r[j,k] = V[j*p+k, r]  
    # w_r = W_r reshaped to (p,n), so w_r[k,i] = W[k*n+i, r]
    # Then: C[i,k] = sum_r w_r[k,i] * (a_r @ ... ) 
    # Actually: sum_r w_r[k,i] * (sum_j a_r[i,j] * b_r[j,k])? No, a_r[i,j]*b_r[j,k] summed over j gives a scalar per (r,i,k).
    
    # Let me try: C[i,k] = sum_r w_r[k,i] * sum_j a_r[i,j] * b_r[j,k]
    C_test = np.zeros((n, p))
    for r in range(rank):
        a_r = U[:, r].reshape(n, m)
        b_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                C_test[i, k] += w_r[k, i] * np.sum(a_r[i, :] * b_r[:, k])
    
    print("Max error:", np.max(np.abs(C_test - C_true)))
    # This should be ~0 if my understanding is correct.
    
    return np.max(np.abs(C_test - C_true)) < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with schoolbook rank 48
    best_rank = n * m * p
    best_result = None
    
    # Build schoolbook decomposition
    rank = best_rank
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # For 2×3 × 3×8, try ranks from 47 down to 24
    # Use a simple approach: try random initializations and verify
    
    # Actually, let me try a structured approach.
    # For 2×m × m×p, we can sometimes use rank < 2mp.
    # Known: 2×2×2 = 7, 2×3×3 = ? (possibly 11 or 12)
    
    # Let me try to find a decomposition at rank 40, 36, etc.
    # I'll use a simple verification-based search.
    
    for target_rank in range(47, 23, -1):
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random integer/half-integer entries in {-1, 0, 1} or {-2, -1, 0, 1, 2}
            U_rand = np.random.choice([-2, -1, 0, 1, 2], size=(n*m, target_rank))
            V_rand = np.random.choice([-2, -1, 0, 1, 2], size=(m*p, target_rank))
            W_rand = np.random.choice([-2, -1, 0, 1, 2], size=(p*n, target_rank))
            
            # Verify
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            C_test = np.zeros((n, p))
            for r in range(target_rank):
                a_r = U_rand[:, r].reshape(n, m)
                b_r = V_rand[:, r].reshape(m, p)
                w_r = W_rand[:, r].reshape(p, n)
                for i in range(n):
                    for k in range(p):
                        C_test[i, k] += w_r[k, i] * np.sum(a_r[i, :] * b_r[:, k])
            
            if np.max(np.abs(C_test - C_true)) < 1e-10:
                best_rank = target_rank
                best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                break
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
