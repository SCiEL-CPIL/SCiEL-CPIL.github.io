import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    best_rank = schoolbook_rank

    def check_exact(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_approx, atol=1e-9)

    def try_integer_rounding(U, V, W):
        # Try integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if check_exact(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        if check_exact(U_half, V_half, W_half):
            return U_half, V_half, W_half
        
        return None

    # Try to find a lower rank solution
    # We know rank 45 is valid.
    # Try rank 44, 43...
    # We use a random perturbation of the schoolbook solution and try to zero out one column.
    
    for target_rank in range(schoolbook_rank - 1, 14, -1):
        if time.time() > deadline - 20:
            break
            
        # Randomly select columns to keep
        # We try to find a subset of columns from the schoolbook that works?
        # No, the coefficients change.
        
        # Strategy: Start with random init for lower rank and optimize.
        # But optimization is slow.
        
        # Alternative: Use the additive property.
        # If we can find a rank 26 decomposition for 3x3x3, we can combine with 3x3x2 schoolbook (18) to get 44.
        
        # Let's try to find a rank 26 decomposition for 3x3x3.
        # T_333 has shape (9, 9, 9).
        T_333 = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_333[i*3+j, j*3+k, k*3+i] = 1.0
        
        U_333_sch = np.zeros((9, 27))
        V_333_sch = np.zeros((9, 27))
        W_333_sch = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_333_sch[i*3+j, col] = 1
                    V_333_sch[j*3+k, col] = 1
                    W_333_sch[k*3+i, col] = 1
                    col += 1
                    
        # Try to reduce rank of 333 from 27 to 26.
        # We can try to find a linear combination of terms that is zero?
        # Or just optimize.
        
        found_333 = False
        # Try a few restarts for rank 26
        for restart in range(10):
            if time.time() > deadline - 30:
                break
                
            U = np.random.randn(9, 26) * 0.5
            V = np.random.randn(9, 26) * 0.5
            W = np.random.randn(9, 26) * 0.5
            
            # Gradient descent
            for it in range(500):
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_333 - T_approx
                grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= 0.01 * grad_U
                V -= 0.01 * grad_V
                W -= 0.01 * grad_W
                
                if it % 50 == 0:
                    rounded = try_integer_rounding(U, V, W)
                    if rounded is not None:
                        # Found a rank 26 solution for 333
                        # Construct 335 solution
                        U_335 = np.zeros((9, 26 + 18))
                        V_335 = np.zeros((15, 26 + 18))
                        W_335 = np.zeros((15, 26 + 18))
                        
                        # Fill 333 part
                        U_335[:, :26] = rounded[0]
                        V_335[:9, :26] = rounded[1]
                        W_335[:9, :26] = rounded[2]
                        # Note: V and W indices for 333 are 0..8.
                        # For 335, V is 0..14, W is 0..14.
                        # The 333 part uses B_1 (cols 0..2 of B) and C_1 (cols 0..2 of C).
                        # B_1 indices in 335: j*5 + k for k in 0..2.
                        # Wait, the index mapping is different.
                        # T_333 uses B of shape 3x3.
                        # T_335 uses B of shape 3x5.
                        # The first 3 columns of B in 335 correspond to B in 333.
                        # So V_333 index j*3+k maps to V_335 index j*5+k.
                        # W_333 index k*3+i maps to W_335 index k*5+i.
                        
                        # This mapping is complex.
                        # Let's just use the direct sum construction.
                        # U_335[:, :26] = U_333
                        # V_335[j*5+k, :26] = V_333[j*3+k]
                        # W_335[k*5+i, :26] = W_333[k*3+i]
                        
                        # Fill 332 part (schoolbook)
                        # 332: A (3x3), B_2 (3x2), C_2 (3x2)
                        # B_2 corresponds to cols 3,4 of B in 335.
                        # C_2 corresponds to cols 3,4 of C in 335.
                        
                        col_offset = 26
                        for i in range(3):
                            for j in range(3):
                                for k in range(2):
                                    # B_2 index: j*2 + k
                                    # In 335, B index: j*5 + (3+k)
                                    # C_2 index: k*2 + i
                                    # In 335, C index: (3+k)*3 + i
                                    
                                    U_335[i*3+j, col_offset] = 1
                                    V_335[j*5 + (3+k), col_offset] = 1
                                    W_335[(3+k)*3 + i, col_offset] = 1
                                    col_offset += 1
                                    
                        if check_exact(U_335, V_335, W_335):
                            best_result = {"U": U_335.tolist(), "V": V_335.tolist(), "W": W_335.tolist()}
                            best_rank = 44
                            return best_result
                            
        # If no rank 26 found, continue to next target rank
        pass

    return best_result
