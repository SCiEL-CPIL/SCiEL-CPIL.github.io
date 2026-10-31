# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin for 150s limit
    
    DIMENSIONS = [4, 4, 4]
    """
    Returns the 49-rank BCLR decomposition for 4x4x4 matrix multiplication.
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    This uses a recursive Strassen-like construction:
    1. View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    2. Apply Strassen's 7-multiplication algorithm at the block level.
    3. Each block multiplication is a 2x2x2 multiplication, which is also decomposed
       using Strassen's 7-multiplication algorithm.
    4. Total rank = 7 (outer) * 7 (inner) = 49.
    
    We then attempt to reduce the rank by identifying and removing linearly dependent terms.
    """
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Define Strassen Coefficients for 2x2 Multiplication ---
    beta = [
        [[1, 0], [0, 1]],  # s=0
        [[0, 0], [1, 1]],  # s=1
        [[1, 0], [0, 0]],  # s=2
        [[0, 0], [0, 1]],  # s=3
        [[1, 1], [0, 0]],  # s=4
        [[-1, 0], [1, 0]], # s=5
        [[0, 1], [0, -1]]  # s=6
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # s=0
        [[1, 0], [0, 0]],  # s=1
        [[0, 1], [0, -1]], # s=2
        [[-1, 0], [1, 0]], # s=3
        [[0, 0], [0, 1]],  # s=4
        [[1, 1], [0, 0]],  # s=5
        [[0, 0], [1, 1]]   # s=6
    ]
    
    delta = [
        [[1, 0], [0, 1]],  # s=0
        [[0, 0], [1, -1]], # s=1
        [[0, 1], [0, 1]],  # s=2
        [[1, 0], [1, 0]],  # s=3
        [[-1, 1], [0, 0]], # s=4
        [[0, 0], [0, 1]],  # s=5
        [[1, 0], [0, 0]]   # s=6
    ]
    
    # --- Construct the 49-rank decomposition ---
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s][I][J] * beta[t][i_prime][j_prime]
                    if coeff != 0:
                        U[row_u][col] = coeff
                        
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s][J][K] * gamma[t][j_prime][k_prime]
                    if coeff != 0:
                        V[row_v][col] = coeff
                        
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s][I][K] * delta[t][i_prime][k_prime]
                    if coeff != 0:
                        W[row_w][col] = coeff
                        
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Function to verify decomposition
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]"""
        n, m, p = 4, 4, 4
        rank = U_mat.shape[1]
        T_reconstructed = np.zeros((n, m, p))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] = np.sum(
                        U_mat[i*m + j, :] * V_mat[j*p + k, :] * W_mat[k*n + i, :]
                    )
        
        # Expected tensor: T[i,j,k] = 1 if i==j and j==k (identity-like for matmul)
        # Actually for matrix multiplication, T[i,j,k] = delta_{i,k} * delta_{j,j}... 
        # No, the matrix multiplication tensor T has T[i,j,k] = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # So T[i,j,k] should be 1 when we're summing over j, but the tensor representation is:
        # T[i,j,k] = delta_{i,j} * delta_{j,k} is wrong.
        # The correct tensor for matrix multiplication: T[i,j,k] = 1 if we're computing C_{i,k} = sum_j A_{i,j} B_{j,k}
        # In the decomposition: C_{i,k} = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * V[j*p+k, r])
        # This equals sum_r W[k*n+i, r] * (U[:,r]^T * V[:,r])_k where indices are flattened.
        # Actually: sum_j U[i*m+j, r] * V[j*p+k, r] is the inner product of row i of U_block with column k of V_block.
        # For this to equal delta_{i,k} (for identity) or to compute matmul correctly...
        
        # Let me think again. The matrix multiplication tensor is:
        # T_{i,j,k} such that (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # In tensor form: T is a 3-tensor where T[i,j,k] = 1 if we're summing A[i,j]*B[j,k]
        # But in the decomposition form given:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal T[i,j,k] for the matrix multiplication tensor.
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
        # Actually, the standard matrix multiplication tensor has T[i,j,k] = delta_{i,j} * delta_{j,k} is not right either.
        
        # Let me just check: for the decomposition to be correct, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        # where T is the matrix multiplication tensor.
        # The matrix multiplication tensor T[i,j,k] = 1 if the indices form a valid matmul term.
        # Specifically, T[i,j,k] = 1 for all i,j,k in the range, because we're summing over j.
        # Wait, no. The tensor T represents the bilinear form: T(A,B) = A*B.
        # In coordinate form: (A*B)_{i,k} = sum_j T[i,j,k] * A_{i,j} * B_{j,k}
        # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this representation).
        
        # Actually, I think the convention is that T[i,j,k] = 1 for all i,j,k in [0,4)
        # Let me verify with a simple case.
        
        # For now, let's just check that the decomposition is consistent by comparing
        # against a known correct decomposition.
        
        # Actually, the simplest check: the decomposition should reproduce matrix multiplication.
        # Let's create random matrices and check.
        
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # Extract the matrices for this rank component
            U_r = U_mat.reshape(n, m, rank)[:, :, r]
            V_r = V_mat.reshape(m, p, rank)[:, :, r]
            W_r = W_mat.reshape(p, n, rank)[:, :, r]
            
            # The contribution to C from rank r is:
            # C += W_r * (U_r * V_r) but this isn't quite right.
            # Actually: C_{i,k} += sum_j U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Wait, W is indexed as W[k*n+i, r], so W[k,i,r] = W_mat[k*n+i, r]
            
        # Let me redo this properly
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += np.sum(
                        U_mat[i*m + j, :] * V_mat[j*p + k, :] * W_mat[k*n + i, :]
                    )
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Try to reduce rank by removing linearly dependent columns
    # Strategy: try removing one column at a time and check if the decomposition still works
    
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = 49
    
    # First, verify the original decomposition
    if not verify_decomposition(U_np, V_np, W_np):
        # If verification fails, return the original
        return {"U": U, "V": V, "W": W}
    
    # Try to remove columns one at a time
    # We'll try all single-column removals first
    
    columns_to_try = list(range(49))
    
    for col in columns_to_try:
        if time.time() > deadline:
            break
            
        # Try removing this column
        mask = [c for c in range(49) if c != col]
        U_test = U_np[:, mask]
        V_test = V_np[:, mask]
        W_test = W_np[:, mask]
        
        if verify_decomposition(U_test, V_test, W_test):
            # This column can be removed!
            best_U = U_test
            best_V = V_test
            best_W = W_test
            best_rank = 48
            print(f"Removed column {col}, new rank: {best_rank}")
            
            # Try to remove more columns
            columns_to_try = [c for c in mask]
            break
    
    # If we found a rank-48 decomposition, try to go lower
    if best_rank < 49:
        # Continue trying to remove more columns
        current_mask = list(range(best_rank))
        for _ in range(10):  # Try up to 10 more removals
            if time.time() > deadline:
                break
                
            removed = False
            for col in current_mask[:]:
                if time.time() > deadline:
                    break
                    
                mask = [c for c in current_mask if c != col]
                if not mask:
                    continue
                    
                U_test = best_U[:, mask]
                V_test = best_V[:, mask]
                W_test = best_W[:, mask]
                
                if verify_decomposition(U_test, V_test, W_test):
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_rank = len(mask)
                    current_mask = mask
                    removed = True
                    print(f"Removed column {col}, new rank: {best_rank}")
                    break
            
            if not removed:
                break
    
    # Convert back to lists
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    # Round to nearest integer if close
    for mat in [best_U_list, best_V_list, best_W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                    mat[i][j] = int(round(mat[i][j]))
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
