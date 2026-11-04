"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _round_to_rational(mat, max_denom=8):
    """Attempt to round float matrix entries to simple rationals (integers or half-integers)."""
    rounded = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if abs(val) < 1e-10:
                rounded[i, j] = 0
            else:
                # Try to find a fraction with small denominator
                found = False
                for d in range(1, max_denom + 1):
                    num = round(val * d)
                    if abs(val - num / d) < 1e-8:
                        rounded[i, j] = num / d
                        found = True
                        break
                if not found:
                    rounded[i, j] = val
    return rounded

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if the decomposition is exact (or close enough to be considered exact after rounding)."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # The known rank-17 solution
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Index mapping:
    # a: A[i,j] -> i*3+j
    # b: B[j,k] -> j*3+k
    # c: C[i,k] -> k*2+i
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    T = _build_tensor(n, m, p)
    
    # Verify the rank-17 solution
    if _verify_decomposition(U, V, W, T):
        # Now attempt to find a rank-16 solution using a structural approach:
        # We formulate the problem as finding 16 rank-1 tensors that sum to T.
        # We use a perturbative approach: start from the rank-17 solution and try to
        # merge two terms or find a linear dependency that allows reduction.
        
        # Strategy: Try to find a rank-16 decomposition by solving a system of
        # polynomial equations. We parameterize the factors and use numerical solving.
        
        # Alternative structural approach: Use the fact that the tensor is a matrix
        # multiplication tensor. Try different block decompositions.
        
        # Let's try a different block structure:
        # Instead of Strassen on top-left 2x2, try other decompositions.
        
        # Attempt 1: Try rank-16 by removing one term and adjusting others
        # This is a nonlinear problem, so we use a simple iterative approach.
        
        best_rank = 17
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try to find a rank-16 solution by dropping each term and solving
        # for adjustments using least squares with constraints.
        
        for drop_idx in range(17):
            if time.time() > deadline - 20:
                break
            
            # Keep all terms except drop_idx
            keep = [r for r in range(17) if r != drop_idx]
            U0 = U[:, keep].copy()
            V0 = V[:, keep].copy()
            W0 = W[:, keep].copy()
            
            # Compute the residual: T - sum of kept terms
            residual = T.copy()
            for r in range(16):
                residual -= np.einsum('a,b,c->abc', U0[:,r], V0[:,r], W0[:,r])
            
            # Check if residual is zero (unlikely but possible)
            if np.allclose(residual, 0, atol=1e-9):
                # We found a rank-16 solution!
                # Round to rationals
                U_r = _round_to_rational(U0, max_denom=4)
                V_r = _round_to_rational(V0, max_denom=4)
                W_r = _round_to_rational(W0, max_denom=4)
                if _check_exact(U_r, V_r, W_r, T):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    # Use unrounded but verify
                    if _check_exact(U0, V0, W0, T):
                        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
            
            # Try to absorb the residual into one of the kept terms
            # This is a more complex optimization; skip for now due to time constraints
        
        # Attempt 2: Try a completely different decomposition structure
        # Use block decomposition with different block sizes
        
        # For 2x3 * 3x3, try:
        # Split B into B_top (2x3) and B_bottom (1x3)
        # C = A_top * B_top + A_bottom * B_bottom
        
        # A_top is 2x2, A_bottom is 2x1
        # B_top is 2x3, B_bottom is 1x3
        
        # C_top = A_top * B_top (2x2 * 2x3 -> 2x3)
        # C_bottom = A_bottom * B_bottom (2x1 * 1x3 -> 2x3)
        
        # For C_top: use Strassen-like for 2x2 * 2x2 part (left 2 columns)
        # plus direct for right column
        # For C_bottom: direct
        
        # This is essentially what we already have. Let's try other structures.
        
        # Try: Split A into A_left (2x2) and A_right (2x1)
        # C = A_left * B_left + A_right * B_right
        # B_left is 2x3, B_right is 1x3
        # A_left * B_left: 2x2 * 2x3
        # A_right * B_right: 2x1 * 1x3
        
        # This is the same as before.
        
        # Let's try a more aggressive approach: use numerical optimization
        # to find a rank-16 decomposition.
        
        # Parameterize 16 rank-1 terms and minimize ||sum - T||^2
        # This is a non-convex problem, so use random restarts.
        
        rng = np.random.default_rng(42)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize random rank-16 factors
            U_init = rng.standard_normal((n_a, 16)) * 0.5
            V_init = rng.standard_normal((n_b, 16)) * 0.5
            W_init = rng.standard_normal((n_c, 16)) * 0.5
            
            # Use gradient descent with simple updates
            # This is computationally expensive; limit iterations
            lr = 0.01
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U_init)
                grad_V = np.zeros_like(V_init)
                grad_W = np.zeros_like(W_init)
                
                for r in range(16):
                    # Compute contribution of term r
                    term_r = np.einsum('a,b,c->abc', U_init[:,r], V_init[:,r], W_init[:,r])
                    diff = term_r - T / 16  # Rough gradient direction
                    
                    # Gradient for U
                    grad_U[:,r] += np.einsum('abc,bc->a', diff, V_init[:,r] * W_init[:,r])
                    grad_V[:,r] += np.einsum('abc,ac->b', diff, U_init[:,r] * W_init[:,r])
                    grad_W[:,r] += np.einsum('abc,ab->c', diff, U_init[:,r] * V_init[:,r])
                
                # Update
                U_init -= lr * grad_U
                V_init -= lr * grad_V
                W_init -= lr * grad_W
            
            # Check result
            U_r = _round_to_rational(U_init, max_denom=8)
            V_r = _round_to_rational(V_init, max_denom=8)
            W_r = _round_to_rational(W_init, max_denom=8)
            
            if _check_exact(U_r, V_r, W_r, T):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            if _check_exact(U_init, V_init, W_init, T):
                return {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_init.tolist()}
        
        # If no rank-16 found, return the rank-17 solution
        U_final = _round_to_rational(U, max_denom=4)
        V_final = _round_to_rational(V, max_denom=4)
        W_final = _round_to_rational(W, max_denom=4)
        
        if _check_exact(U_final, V_final, W_final, T):
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if the rank-17 solution fails
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
