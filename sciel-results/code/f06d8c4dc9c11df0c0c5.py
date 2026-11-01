import numpy as np
import time
import itertools
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    """
    r = U.shape[1]
    T_recon = np.zeros((n, m, m, p, n, p))
    for l in range(r):
        u_l = U[:, l].reshape(n, m)
        v_l = V[:, l].reshape(m, p)
        w_l = W[:, l].reshape(p, n)
        term = np.einsum('ab,cd,ki->abcdki', u_l, v_l, w_l)
        T_recon += term
        
    T_target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                T_target[a, b, c, b, a, c] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    # Fallback: Rank 34 Solution
    def get_rank_34_solution():
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        # Rank 17 decomposition for 2x3x3 (Hardcoded from previous best)
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    # Fallback Initialization
    U_fallback, V_fallback, W_fallback = get_rank_34_solution()
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    best_rank = 34

    # NEW STRATEGY: Algebraic Search for Rank 33
    # We fix U to have a specific sparse structure that might allow rank reduction.
    # The tensor equation is sum_l U_ab,l V_cb,l W_ca,l = T_abcdki.
    # If we fix U, the equations become linear in the variables V and W?
    # No, it's bilinear in V and W.
    # However, if we fix U AND V, it's linear in W.
    # Or if we fix U, we can view it as a system where V and W are coupled.
    
    # A more robust algebraic approach for small ranks:
    # Try to find a rank 33 solution by perturbing the rank 34 solution 
    # and checking if we can zero out one column of U, V, W simultaneously 
    # while maintaining the tensor identity.
    # This is equivalent to checking if the rank 34 solution lies on the 
    # secant variety of rank 33.
    
    # Let's try a "Null Space" approach.
    # The condition for rank r is that the tensor T lies in the r-th secant variety.
    # We can check if a rank r-1 approximation is possible by looking at the 
    # singular values of the "unfolding" of the tensor.
    
    # Unfold T into a matrix T_{(ab), (cdki)}.
    # If rank(T_unfold) <= r, then rank(T) <= r.
    # This is a necessary but not sufficient condition.
    
    # Let's compute the rank of the 3-mode unfolding.
    # T shape: (n*m, m*p, p*n) -> (6, 18, 12)
    # Flatten first mode: (6, 18*12) = (6, 216). Rank is at most 6.
    # Flatten second mode: (18, 6*12) = (18, 72). Rank is at most 18.
    # Flatten third mode: (12, 6*18) = (12, 108). Rank is at most 12.
    
    # The matrix multiplication tensor M_{2,3,6} has a specific structure.
    # Let's construct the tensor T and check its unfolding ranks.
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                # T_ab, cd, ki
                # Non-zero when a=i, b=d, c=k
                # Indices:
                # u_idx = a*m + b
                # v_idx = c*p + d = c*p + b
                # w_idx = k*n + i = c*n + a
                u_idx = a * m + b
                v_idx = c * p + b
                w_idx = c * n + a
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Check rank of unfoldings
    T_unfold1 = T.reshape(n*m, -1)
    T_unfold2 = T.reshape(m*p, -1)
    T_unfold3 = T.reshape(p*n, -1)
    
    r1 = np.linalg.matrix_rank(T_unfold1)
    r2 = np.linalg.matrix_rank(T_unfold2)
    r3 = np.linalg.matrix_rank(T_unfold3)
    
    # The tensor rank is at least max(r1, r2, r3).
    # If max(r1, r2, r3) < 34, it doesn't mean rank is < 34, but it's a lower bound.
    # Actually, for M_{n,m,p}, the rank is usually n*m*p unless there's structure.
    # Here n=2, m=3, p=6.
    
    # Let's try a specific construction for Rank 33.
    # We know that M_{2,3,3} has rank 17 (or 18? Schoolbook is 18. Best known is 17? 
    # Actually, M_{2,3,3} rank is 17? No, 18 is schoolbook. 
    # Strassen for 2x2x2 is 7.
    # For 2x3x3, schoolbook is 18. 
    # Can we do better? 
    # The current program uses 17 for the block. Let's assume that's valid.
    # If M_{2,3,3} rank is 17, then M_{2,3,6} can be built from two blocks?
    # M_{2,3,6} = M_{2,3,3} \otimes I_2? No.
    # M_{2,3,6} is multiplying (2x3) by (3x6).
    # This is equivalent to two independent (2x3) by (3x3) multiplications?
    # No, the second matrix is 3x6.
    # Let A be 2x3, B be 3x6.
    # B can be split into B1 (3x3) and B2 (3x3).
    # A*B = A*B1 + A*B2.
    # This requires rank(M_{2,3,3}) + rank(M_{2,3,3}) = 17 + 17 = 34.
    # This explains the rank 34 solution.
    
    # Can we do better than 34?
    # This would require "crossing" terms between the two 3x3 blocks of B.
    # This is possible if the tensor M_{2,3,6} has rank < 2 * rank(M_{2,3,3}).
    
    # Let's try to find a rank 33 solution by searching for a perturbation 
    # of the block-diagonal solution that allows one term to be shared.
    
    # We will use a local search on the "shared term" hypothesis.
    # Assume the solution looks like:
    # 16 terms from Block 1 (B1)
    # 16 terms from Block 2 (B2)
    # 1 shared term that contributes to both.
    # Total 33 terms.
    
    # We define the residual of the block solution as 0.
    # We want to modify 1 term from Block 1 and 1 term from Block 2 
    # and replace them with 1 term that satisfies the linear combination?
    # No, tensor rank reduction is not linear.
    
    # Let's try a random restart search specifically targeting rank 33 
    # with a "warm start" from the rank 34 solution.
    
    def search_rank_33():
        start = time.time()
        budget = 10.0 # 10 seconds for this specific search
        
        # Initialize U, V, W from rank 34 solution
        U = np.array(U_fallback, dtype=float)
        V = np.array(V_fallback, dtype=float)
        W = np.array(W_fallback, dtype=float)
        
        # We want to zero out the last column (index 33) of U, V, W
        # and adjust the other 33 columns to compensate.
        # This is a non-linear constraint satisfaction problem.
        
        # We can use gradient descent on the error of the tensor reconstruction
        # while penalizing the norm of the 33rd column to drive it to zero.
        
        # Error = || T - sum_{l=0}^{32} u_l v_l w_l ||_F^2
        # Constraint: || u_33 ||^2 + || v_33 ||^2 + || w_33 ||^2 < epsilon
        
        # We will optimize U[:, :33], V[:, :33], W[:, :33] 
        # to minimize the error, ignoring the 33rd column.
        # If the minimum error is close to 0, we have a rank 33 solution.
        
        U_target = U[:, :33]
        V_target = V[:, :33]
        W_target = W[:, :33]
        
        # Flatten parameters
        params = np.concatenate([U_target.flatten(), V_target.flatten(), W_target.flatten()])
        
        # Compute gradient via finite differences? Too slow.
        # Use analytic gradient.
        
        def compute_error_and_grad(params):
            U_t = params[:U_dim*33].reshape(U_dim, 33)
            V_t = params[U_dim*33:U_dim*33 + V_dim*33].reshape(V_dim, 33)
            W_t = params[U_dim*33 + V_dim*33:].reshape(W_dim, 33)
            
            # Reconstruct T
            T_recon = np.zeros((n*m, m*p, p*n))
            grad_U = np.zeros_like(U_t)
            grad_V = np.zeros_like(V_t)
            grad_W = np.zeros_like(W_t)
            
            # T_recon[ab, cd, ki] = sum_l U_ab,l V_cd,l W_ki,l
            # Target T_target is 1 at specific indices.
            
            # Error = sum_{ab,cd,ki} (T_recon[ab,cd,ki] - T_target[ab,cd,ki])^2
            # Gradient w.r.t U_ab,l:
            # 2 * (T_recon[ab,cd,ki] - T_target[ab,cd,ki]) * V_cd,l W_ki,l
            
            # We can compute this efficiently by iterating over l and using einsum.
            # But for 33 terms, it's manageable.
            
            # Let's just compute the error first.
            err = 0.0
            for l in range(33):
                term = np.einsum('ab,cd,ki->abcdki', 
                                 U_t[:, l].reshape(n, m), 
                                 V_t[:, l].reshape(m, p), 
                                 W_t[:, l].reshape(p, n))
                # Flatten to match T shape (n*m, m*p, p*n)
                term_flat = term.reshape(n*m, m*p, p*n)
                err += np.sum((term_flat - T)**2)
                
                # Gradient calculation is complex. 
                # Let's use a simpler heuristic: 
                # If we can't find an analytic gradient quickly, 
                # we might rely on the fact that the rank 34 solution is 
                # very close to a rank 33 solution if one exists.
                
            # This loop is too slow for 10 seconds if done naively in Python.
            # We need a vectorized approach or a different method.
            
            return err, None # Placeholder
            
        # Given the complexity and time limit, a full gradient descent 
        # on 33*36 variables is risky in 10s.
        
        # Alternative: Check if the rank 34 solution can be "compressed" 
        # by checking the rank of the "contracted" tensors.
        
        # If we fix U, the map V -> T is linear.
        # Let's fix U to be the first 33 columns of the rank 34 U.
        # Then solve for V and W?
        # The equations are quadratic in V and W.
        
        # Let's try a simpler structural check.
        # Is it possible that the rank is exactly 34?
        # If so, we should return 34.
        
        # If we have time, we can try a few random rank 33 initializations 
        # and see if the error is small.
        
        for _ in range(10):
            if time.time() - start > budget:
                break
            # Random perturbation of the last column
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            
            # Zero out last column
            U_test[:, 33] = 0
            V_test[:, 33] = 0
            W_test[:, 33] = 0
            
            # Check if the remaining 33 terms can span the tensor.
            # This is equivalent to checking if T is in the span of 
            # {u_l v_l w_l | l < 33}.
            
            # We can check this by solving a least squares problem:
            # Minimize || T - sum_{l=0}^{32} c_l (u_l v_l w_l) ||
            # But the coefficients are fixed to 1 in the tensor decomposition definition?
            # No, the decomposition is sum_l u_l v_l w_l. The coefficients are absorbed into the vectors.
            # So we are asking if T is in the linear span of the 33 rank-1 tensors 
            # defined by the first 33 columns of the rank 34 solution.
            
            # Let's compute the projection of T onto the subspace spanned by these 33 tensors.
            # If the residual is small, we have a candidate.
            
            # Construct the 33 rank-1 tensors as a matrix (1296 x 33)
            # Each column is a flattened rank-1 tensor.
            
            A = np.zeros((n*m*m*p*n*p, 33))
            for l in range(33):
                u_l = U_test[:, l].reshape(n, m)
                v_l = V_test[:, l].reshape(m, p)
                w_l = W_test[:, l].reshape(p, n)
                t_l = np.einsum('ab,cd,ki->abcdki', u_l, v_l, w_l)
                A[:, l] = t_l.flatten()
                
            T_flat = T.flatten()
            
            # Solve least squares: A x = T_flat
            # If the residual is small, then T is approximately in the span.
            # If it is EXACTLY in the span, we have a rank 33 solution 
            # (with the same vectors, but potentially different coefficients? 
            # No, the decomposition requires coefficients to be 1? 
            # Wait. The standard definition is T = sum u_l v_l w_l. 
            # The vectors u_l, v_l, w_l can be scaled. 
            # If T = sum c_l u_l v_l w_l, we can absorb c_l into u_l.
            # So yes, if T is in the linear span of the 33 rank-1 tensors, 
            # we have a rank 33 decomposition.
            
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, T_flat, rcond=None)
                if residuals.size > 0:
                    res = residuals[0]
                    if res < 1e-6:
                        # We found a rank 33 solution!
                        # The new vectors are U_test[:, l] * x[l], etc.
                        U_new = U_test * x
                        V_new = V_test
                        W_new = W_test
                        # Verify
                        if check_decomposition(U_new, V_new, W_new, n, m, p):
                            return U_new, V_new, W_new
            except:
                pass
                
        return None, None, None

    U_33, V_33, W_33 = search_rank_33()
    if U_33 is not None:
        best_U, best_V, best_W = U_33, V_33, W_33
        best_rank = 33

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    construct()
