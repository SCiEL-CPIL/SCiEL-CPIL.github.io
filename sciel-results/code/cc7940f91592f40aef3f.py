"""Matrix multiplication 2x3x6 with rank-minimal decomposition via Kronecker structure."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit
    
    n, m, p = 2, 3, 6
    
    # Try to find a lower-rank decomposition
    # The tensor T[i,j,k] has T[i,j,k] = 1 if the (i,k) entry of A@B is determined by A[i,j]*B[j,k]
    # We want to find U (6 x r), V (18 x r), W (12 x r) such that
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k) where A[i,j]*B[j,k] appears
    # Actually the tensor is: T_{i,j,k} such that the bilinear form is sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try the Kronecker product approach:
    # M_{2,3,6} can be seen as M_{2,1,2} ⊗ M_{1,3,3} but that's not quite right.
    # Better: M_{2,3,6} = M_{2,3,3} ⊕ M_{2,3,3} (direct sum along the p dimension)
    # This gives rank = 2 * rank(M_{2,3,3})
    
    # What is rank(M_{2,3,3})?
    # Lower bound: max(2*3, 3*3, 3*2) = max(6, 9, 6) = 9
    # Upper bound: schoolbook = 2*3*3 = 18
    # Known results suggest rank(M_{2,3,3}) might be 11 or 12.
    
    # Let me try to construct a decomposition of M_{2,3,3} with low rank.
    # For M_{2,3,3}, we need U (6 x r), V (9 x r), W (6 x r)
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # Strategy: Use the fact that M_{2,3,3} can be decomposed using the 2x2x2 tensor (rank 7)
    # M_{2,3,3} = M_{2,2,2} ⊗ M_{1,2,1} ⊕ something... this gets complex.
    
    # Let me try a different approach: use numerical search or known constructions.
    
    # Actually, let me try to use the "border rank" or known decompositions.
    # For now, let me try to construct a decomposition manually or via search.
    
    # Key insight: M_{2,3,6} can be partitioned as three M_{2,1,2} blocks along the j dimension.
    # M_{2,1,2} has rank 3 (since max(2,2,2)=2, and it's known to be 3).
    # So total rank would be 3 * 3 = 9? No wait, M_{2,1,2} rank is actually 3.
    # But we can't just add them independently because they share structure.
    
    # Let me reconsider. The multiplication tensor for A (2x3) * B (3x6) = C (2x6):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has entries T[i,j,k] = 1 for all (i,j,k) in [0,1]x[0,1,2]x[0,1,2,3,4,5]
    
    # The rank of this tensor is what we want to minimize.
    
    # Let me try a construction based on the Kronecker product of smaller tensors.
    # M_{2,3,6} = M_{2,1,2} ⊗ M_{1,3,3} doesn't directly work.
    
    # Alternative: Think of it as 2 independent M_{1,3,6} tensors (one for each row of A)
    # M_{1,3,6} has rank 6 (it's just the identity-like tensor, rank = max(3,6)=6... actually rank is 6)
    # Wait, M_{1,m,p} has rank m*p? No. M_{1,m,p} is just the product of a vector and matrix.
    # The tensor for M_{1,3,6} has T[j,k] = 1 for all j,k. This is just the all-ones tensor, rank 1?
    # No wait. For M_{1,m,p}, C[0,k] = sum_j A[0,j]*B[j,k].
    # The tensor is T[0,j,k] = 1 for all j,k. So T is a 3x6 all-ones matrix flattened, which has rank 1.
    # So M_{1,3,6} has rank 1!
    
    # Therefore M_{2,3,6} = M_{1,3,6} ⊕ M_{1,3,6} (two copies, one for each row of A)
    # Rank = 1 + 1 = 2? That can't be right because of the lower bound max(6,18,12)=18.
    
    # I'm confusing myself. Let me be more careful.
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) = 6 x 18 x 12
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), 0 otherwise.
    
    # For M_{1,3,6}: n=1, m=3, p=6
    # T[j, j*p+k, k*1+0] = T[j, 6j+k, k] = 1
    # This is a specific pattern, not all ones.
    
    # Let me just try to numerically find a low-rank decomposition using SVD or similar.
    # Flatten the tensor into a matrix and find its rank.
    
    # Reshape T (6x18x12) into a matrix (6*18) x 12 or other modes.
    T = np.zeros((6, 18, 12))
    for i in range(2):
        for j in range(3):
            for k in range(6):
                T[i*3+j, j*6+k, k*2+i] = 1.0
    
    # Compute the rank of different matricizations
    T_mode0 = T.reshape(6, 18*12)
    T_mode1 = T.reshape(18, 6*12)
    T_mode2 = T.reshape(12, 6*18)
    
    rank0 = np.linalg.matrix_rank(T_mode0)
    rank1 = np.linalg.matrix_rank(T_mode1)
    rank2 = np.linalg.matrix_rank(T_mode2)
    
    # The tensor rank is at least max of these
    lb = max(rank0, rank1, rank2)
    
    # Let's try to find a decomposition using CP format
    # We want U (6 x r), V (18 x r), W (12 x r) such that T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # Try rank = lb first
    r = lb
    U = np.random.randn(6, r)
    V = np.random.randn(18, r)
    W = np.random.randn(12, r)
    
    # Use ALS (Alternating Least Squares) to fit the decomposition
    def tensor_to_vec(T, U, V, W, mode):
        """Reshape tensor with one mode fixed"""
        pass
    
    # Simpler: use the matricization approach
    # T_mode0 = U @ (V ⊙ W)^T where ⊙ is Khatri-Rao product
    # So U = T_mode0 @ (V ⊙ W) @ pinv((V ⊙ W)^T @ (V ⊙ W))
    
    def khatri_rao(A, B):
        """Khatri-Rao product (column-wise Kronecker)"""
        return np.kron(A.T, B.T).T
    
    best_rank = None
    best_U, best_V, best_W = None, None, None
    
    for trial_rank in range(max(9, lb), 25):  # try ranks from lower bound up to 24
        if time.time() > deadline - 20:
            break
            
        U = np.random.randn(6, trial_rank)
        V = np.random.randn(18, trial_rank)
        W = np.random.randn(12, trial_rank)
        
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U
            VW = khatri_rao(V, W)  # (18*12) x trial_rank
            U = T_mode0 @ VW @ np.linalg.pinv(VW.T @ VW)
            
            # Update V
            UW = khatri_rao(U, W)  # (6*12) x trial_rank
            V = T_mode1 @ UW @ np.linalg.pinv(UW.T @ UW)
            
            # Update W
            UV = khatri_rao(U, V)  # (6*18) x trial_rank
            W = T_mode2 @ UV @ np.linalg.pinv(UV.T @ UV)
        
        # Check if this decomposition is exact (within tolerance)
        # Reconstruct T and check
        T_recon = np.zeros((6, 18, 12))
        for c in range(trial_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
        error = np.max(np.abs(T_recon - T))
        
        if error < 1e-10:
            # Found an exact decomposition
            best_rank = trial_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # If we found a low-rank decomposition, use it
    if best_rank is not None and best_rank < 36:
        # Convert to integer/half-integer if possible
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        
        # Try to round to half-integers
        for mat in [U_out, V_out, W_out]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    mat[i][j] = round(mat[i][j] * 2) / 2
        
        # Verify after rounding
        T_check = np.zeros((6, 18, 12))
        for c in range(best_rank):
            for i in range(6):
                for j in range(18):
                    for k in range(12):
                        T_check[i,j,k] += U_out[i][c] * V_out[j][c] * W_out[k][c]
        
        if np.max(np.abs(T_check - T)) < 1e-10:
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition
    n, m, p = 2, 3, 6
    rank = n * m * p
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
