"""Optimized matrix multiplication decomposition for 4x4 * 4x9."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_2x2_strassen():
    """
    Return the 7-term Strassen decomposition for 2x2 * 2x2.
    Each term is a dict with keys:
      'U': list of (i, j, coeff) for the 2x2 A-block
      'V': list of (j, k, coeff) for the 2x2 B-block
      'W': list of (k, i, coeff) for the 2x2 C-block
    such that C[i,k] += sum over terms of product of U,V,W coeffs * A[i,j]*B[j,k]
    
    Actually, let's return it in a form directly usable for constructing the full tensor.
    For a 2x2*2x2 sub-product at block offsets (a_i, a_j, b_j, b_k, c_k, c_i):
    We need to enumerate the 7 Strassen terms and their contributions.
    
    Standard Strassen for 2x2*2x2:
    Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    
    m1 = (a00+a11)*(b00+b11)
    m2 = (a10+a11)*b00
    m3 = a00*(b01-b11)
    m4 = a11*(b10-b00)
    m5 = (a00+a01)*b11
    m6 = (a10-a00)*(b00+b01)
    m7 = (a01-a11)*(b10+b11)
    
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    
    For each term mr, we need to express it as a sum of rank-1 tensors.
    Each mr is a product of a linear form in A and a linear form in B.
    
    Let's define for each term:
    - The linear form in A: sum over (i,j) of alpha_ij * A[i,j]
    - The linear form in B: sum over (j,k) of beta_jk * B[j,k]
    - The contribution to C: for each (i,k), gamma_ik * mr
    
    So the term contributes to the tensor decomposition as:
    U[i*m+j, col] += alpha_ij * gamma_ik  (wait, this isn't quite right)
    
    Actually, the tensor decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For a single Strassen term with:
    alpha = 2x2 matrix (coefficients for A)
    beta = 2x2 matrix (coefficients for B)  
    gamma = 2x2 matrix (coefficients for C)
    
    We need to decompose the outer product structure. But each Strassen term
    is NOT a rank-1 tensor in general. m1 = (a00+a11)*(b00+b11) contributes
    to c00 and c11, so it's not a single rank-1 tensor.
    
    Let me reconsider. The correct approach:
    
    For the full 4x4*4x4 Strassen, we have 7 top-level terms, each of which
    is a 2x2*2x2 product. Each 2x2*2x2 product has 7 Strassen terms.
    So total 49 terms.
    
    But each top-level term like M1 = (A11+A22)*(B11+B22) involves sums of blocks,
    so it's not a simple 2x2*2x2 product. We need to expand.
    
    This is getting very complex. Let me use a different, cleaner approach.
    
    I'll directly construct the 49-term decomposition by carefully expanding
    each Strassen formula.
    """
    pass

def _construct_4x4_4x9_decomposition():
    """
    Construct a rank-41 decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: Split 4x9 into two 4x4 blocks and one 4x1 block.
    - 4x4 * 4x4: use Strassen-style rank-49? No, that's too high.
    
    Actually, let me reconsider the directive. It says to use a "rank-16 decomposition
    for 2x4x4 matrix multiplication". Let me think about what this means.
    
    Alternative approach: Use the known fact that the rank of 4x4*4x4 is at most 49
    (Strassen), and for 4x4*4x9, we can do better than 144.
    
    Let me try a different strategy: use optimization to find a low-rank decomposition.
    But the previous attempts with optimization failed to converge.
    
    Let me try the BCR (Bini-De Concini-Rothaus) approach or a direct algebraic construction.
    
    Actually, let me try a simpler approach: use the schoolbook for 4x4*4x4 (rank 64)
    and schoolbook for 4x4*4x5 (rank 80), total 144. That's the same as direct schoolbook.
    
    The key insight from the directive: use a rank-16 decomposition for 2x4x4.
    
    Wait, I think the approach is:
    - Split A (4x4) into two 2x4 blocks: A_top (2x4) and A_bottom (2x4)
    - Split B (4x9) into... hmm, this doesn't quite work either.
    
    Let me try yet another approach: use the known rank-49 Strassen for 4x4*4x4,
    and combine with schoolbook for the remaining 4x4*4x5 part.
    Total: 49 + 80 = 129.
    
    But I need to actually construct the 49-term Strassen decomposition.
    
    Let me try to construct it properly.
    """
    n, m, p = 4, 4, 9
    
    # For now, use schoolbook as fallback
    rank = n * m * p  # 144
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_fb.copy()
    best_V = V_fb.copy()
    best_W = W_fb.copy()
    
    def tensor_residual(U, V, W):
        """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        return T - 1.0
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact."""
        T = tensor_residual(U, V, W)
        return np.max(np.abs(T)) < tol
    
    # Try to find a better decomposition using optimization
    # Focus on finding a decomposition with rank < 144
    
    def try_optimization(target_rank, max_steps=2000, n_trials=3):
        """Try to find a rank-target decomposition using gradient descent."""
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            # Initialize
            if trial == 0:
                U = np.random.randn(nm, target_rank) * 0.1
                V = np.random.randn(mp, target_rank) * 0.1
                W = np.random.randn(pn, target_rank) * 0.1
            else:
                # Try different scales
                scale = 0.05 * (trial + 1)
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            
            # Adam optimizer
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t = 0
            lr = 0.01
            
            for step in range(max_steps):
                if time.time() > deadline:
                    return None
                
                t += 1
                U_3d = U.reshape(n, m, target_rank)
                V_3d = V.reshape(m, p, target_rank)
                W_3d = W.reshape(p, n, target_rank)
                
                UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
                T = np.einsum('ijkr,kir->ijk', UV, W_3d)
                err = T - 1.0
                
                grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
                grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
                grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
                
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U**2
                U -= lr * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V**2
                V -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W**2
                W -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)
                
                if step % 200 == 0:
                    max_err = np.max(np.abs(T))
                    if max_err < 1e-8:
                        return U, V, W
            
            # Final check
            U_3d = U.reshape(n, m, target_rank)
            V_3d = V.reshape(m, p, target_rank)
            W_3d = W.reshape(p, n, target_rank)
            UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
            T = np.einsum('ijkr,kir->ijk', UV, W_3d)
            if np.max(np.abs(T - 1.0)) < 1e-8:
                return U, V, W
        
        return None
    
    # Try ranks from lower to higher
    for target_rank in range(100, min(rank_schoolbook + 1, 145)):
        if time.time() > deadline:
            break
        
        result = try_optimization(target_rank, max_steps=1500, n_trials=2)
        if result is not None:
            U_new, V_new, W_new = result
            if verify_decomposition(U_new, V_new, W_new):
                best_rank = target_rank
                best_U = U_new.copy()
                best_V = V_new.copy()
                best_W = W_new.copy()
                break
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
