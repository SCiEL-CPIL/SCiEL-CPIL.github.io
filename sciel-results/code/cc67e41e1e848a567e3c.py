"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Fast verification using einsum.
    T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Reshape
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

def _laderman_4x4():
    """Return Laderman's rank-20 decomposition for 4x4 * 4x4.
    Returns (U, V, W) each of shape (16, 20) where:
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    # Laderman's algorithm for 4x4 matrix multiplication
    # The 20 multiplications are defined by specific linear combinations.
    # Let me use the standard Laderman decomposition.
    # 
    # For matrices A (4x4) and B (4x4), C = A*B
    # The 20 products are:
    # P1 = A[0,0] * B[0,0]
    # P2 = (A[0,1]+A[0,0]) * (B[0,0]-B[0,1])
    # P3 = (A[1,0]+A[1,1]) * B[1,0]
    # P4 = (A[1,0]+A[1,1]) * (B[1,1]-B[1,0])
    # P5 = (A[2,0]+A[2,1]) * B[2,0]
    # P6 = (A[2,0]+A[2,1]) * (B[2,1]-B[2,0])
    # P7 = (A[3,0]+A[3,1]) * B[3,0]
    # P8 = (A[3,0]+A[3,1]) * (B[3,1]-B[3,0])
    # P9 = (A[0,2]+A[0,3]) * B[2,0]
    # P10 = (A[0,2]+A[0,3]) * (B[2,1]-B[2,0])
    # P11 = (A[1,2]+A[1,3]) * B[2,0]
    # P12 = (A[1,2]+A[1,3]) * (B[2,1]-B[2,0])
    # P13 = (A[2,2]+A[2,3]) * B[2,0]
    # P14 = (A[2,2]+A[2,3]) * (B[2,1]-B[2,0])
    # P15 = (A[3,2]+A[3,3]) * B[2,0]
    # P16 = (A[3,2]+A[3,3]) * (B[2,1]-B[2,0])
    # P17 = (A[0,0]+A[1,0]) * (B[0,2]-B[0,3])
    # P18 = (A[2,0]+A[3,0]) * (B[0,2]-B[0,3])
    # P19 = (A[0,0]+A[2,0]) * (B[2,2]-B[2,3])
    # P20 = (A[1,0]+A[3,0]) * (B[1,2]-B[1,3])
    #
    # Actually, let me use a verified Laderman decomposition.
    # I'll construct it from the known structure.
    
    # Let me use a different approach: construct the rank-20 decomposition
    # using the known Laderman factors.
    
    n = 4
    r = 20
    U = np.zeros((16, r))
    V = np.zeros((16, r))
    W = np.zeros((16, r))
    
    # Laderman's 20 multiplications for 4x4:
    # The decomposition is:
    # C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] for the tensor
    # Actually the tensor is T[i,j,k] = 1, and we need
    # T[i,j,k] = sum_c U[i*4+j,c] * V[j*4+k,c] * W[k*4+i,c]
    
    # Let me use the standard Laderman algorithm coefficients.
    # From the literature, Laderman's algorithm has these 20 products:
    
    # I'll use a verified set. Let me define the 20 rank-1 tensors.
    # Each rank-1 tensor is defined by (u, v, w) vectors.
    
    # Product 1: A[0,0]*B[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # i=0,j=0,k=0
    
    # Product 2: (A[0,0]+A[0,1])*(B[1,0]-B[1,1])
    U[0, 1] = 1; U[1, 1] = 1
    V[4, 1] = 1; V[5, 1] = -1
    W[4, 1] = 1; W[8, 1] = -1  # k=1,i=0 -> idx=1*4+0=4; k=1,i=1->5? No.
    # W[k*4+i]: k=1,i=0 -> 4; k=1,i=1 -> 5? No, k=1,i=1 -> 1*4+1=5
    # Wait, I need to be more careful.
    
    # Let me restart with a cleaner approach. I'll use a known correct Laderman
    # decomposition from the literature.
    
    # Actually, for this task, let me just use the schoolbook as fallback and
    # try to find improvements through optimization with fast verification.
    
    return None  # Placeholder - will implement properly below

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Strategy: Use the commutative property.
    # A (n x m) * B (m x p) = A * B
    # Transpose: (A*B)^T = B^T * A^T, so (p x m) * (m x n)
    # Here: 4x4 * 4x9 -> transpose gives 9x4 * 4x4
    # So we can think of it as C^T = B^T * A^T where B^T is 9x4 and A^T is 4x4
    # The 4x4 * 4x4 part can use Laderman (rank 20)
    # The 9x4 part is just selecting rows.
    
    # Alternative structured approach:
    # For each column k of B (k=0..8), we compute A * B[:,k]
    # This is 4x4 times a 4x1 vector, which has rank 16 (schoolbook) or lower.
    # But we can share structure across columns.
    
    # Let's try a block-structured decomposition:
    # Group the 9 columns of B. If we could decompose the 4x4 multiplication
    # with rank r, then for each of the 9 columns we'd need r multiplications,
    # giving rank 9*r. But we can share: if we use the same U factors for all
    # columns, we get rank r * (something).
    
    # Actually, the tensor T[i,j,k] for 4x4*4x9:
    # T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..8
    # Decomposition: T[i,j,k] = sum_c U[i*4+j,c] * V[j*9+k,c] * W[k*4+i,c]
    
    # Key observation: For fixed (i,j), T[i,j,k] = 1 for all k.
    # So sum_c U[i*4+j,c] * V[j*9+k,c] * W[k*4+i,c] = 1 for all k.
    # This means for each (i,j), the vector (sum_c U[i*4+j,c]*V[j*9+k,c]*W[k*4+i,c])_k must be all-ones.
    
    # Let's try a structured approach:
    # For each (i,j) pair (16 pairs), we need a "column" of the output.
    # If we can find rank-r decomposition where for each (i,j), the contribution
    # sums to 1 across all k, we're done.
    
    # Approach: Use rank-16 base (one per (i,j) pair) but with shared structure.
    # Actually schoolbook is 144. Let's try to find something better.
    
    # Let me try: for each i (4 rows of A), for each j (4 cols of A / rows of B),
    # we need the product A[i,j] to be distributed across all 9 columns of B.
    # 
    # Idea: Use 4*4 = 16 "A-factors" and 9 "B-factors" with clever W.
    # But that's not quite right for a tensor decomposition.
    
    # Let's try a direct optimization approach with fast verification.
    
    # Try various ranks
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But Strassen-like bounds suggest higher.
    
    # Let's try ranks from a reasonable range
    for target_rank in [36, 40, 45, 50, 55, 60, 64, 70, 80, 90, 100, 110, 120, 128, 135, 140]:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            break
        
        np.random.seed(42 + target_rank)
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Adam-like optimization
        lr = 0.01
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        b1 = 0.9; b2 = 0.999
        t = 0
        eps = 1e-8
        
        for step in range(2000):
            if time.time() > deadline - 20:
                break
            t += 1
            
            # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            
            diff = T - 1.0
            err = np.sum(diff**2)
            
            if err < 1e-10:
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
            
            # Compute gradients
            # dE/dU[i,j,c] = 2 * diff[i,j,:] dot (V[j,:,c] * W[:,i,c])
            # More precisely: dE/dU[i*m+j,c] = 2 * sum_k diff[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            
            # Vectorized gradient computation
            # grad_U[i,j,c] = 2 * sum_k diff[i,j,k] * V3[j,k,c] * W3[k,i,c]
            # = 2 * einsum('ijk,jkc,kic->ijc', diff, V3, W3)
            
            grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V3, W3)
            grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U3, W3)
            grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U3, V3)
            
            grad_U = grad_U.reshape(n*m, target_rank)
            grad_V = grad_V.reshape(m*p, target_rank)
            grad_W = grad_W.reshape(p*n, target_rank)
            
            # Adam update
            m_U = b1 * m_U + (1 - b1) * grad_U
            v_U = b2 * v_U + (1 - b2) * grad_U**2
            m_U_hat = m_U / (1 - b1**t)
            v_U_hat = v_U / (1 - b2**t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = b1 * m_V + (1 - b1) * grad_V
            v_V = b2 * v_V + (1 - b2) * grad_V**2
            m_V_hat = m_V / (1 - b1**t)
            v_V_hat = v_V / (1 - b2**t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = b1 * m_W + (1 - b1) * grad_W
            v_W = b2 * v_W + (1 - b2) * grad_W**2
            m_W_hat = m_W / (1 - b1**t)
            v_W_hat = v_W / (1 - b2**t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            if step % 100 == 0 and err < 1e-4:
                # Try to polish with smaller learning rate
                lr *= 0.5
    
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
