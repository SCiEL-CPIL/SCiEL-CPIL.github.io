"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: Use dual number encoding to reduce the problem.
    Encode the 3x6 right operand as a 3x3 matrix over F[ε]/(ε²) where:
    - Column k of B maps to entry (k//2, k%2) in the dual-number matrix
    - B[j,k] = real_part[j, k//2] + ε * imag_part[j, k%2]
    
    Then apply a rank-23 decomposition over the dual numbers (from 3x3x3 multiplication),
    which gives a rank-46 decomposition over the reals for 3x3x6.
    
    The key insight: if we have a rank-r decomposition for 3x3x3 matrix multiplication
    over a field, we can lift it to a rank-2r decomposition for 3x3x6 by treating
    the 3x6 matrix as 3x3 over dual numbers.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    deadline = time.time() + 110
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the schoolbook rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_dual_number_approach():
        """
        Try to find a rank-23 decomposition for 3x3x3 matrix multiplication,
        then lift it to a rank-46 decomposition for 3x3x6 via dual numbers.
        
        The 3x3x3 multiplication tensor has known decompositions.
        We'll use a structured approach to find one, then convert.
        """
        # For 3x3x3 matrix multiplication, we need to find rank-r decomposition
        # where U, V, W are 9x r matrices such that:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for valid (i,j,k)
        
        # Known result: 3x3x3 matrix multiplication can be done with rank 23
        # (Strassen's algorithm for 2x2x2 gives rank 7, and there are results for 3x3x3)
        
        # Let's try to find a rank-23 decomposition for 3x3x3 using optimization
        # Then convert to rank-46 for 3x3x6
        
        n3, m3, p3 = 3, 3, 3
        
        def validate_3x3x3(U, V, W):
            """Validate a 3x3x3 decomposition."""
            rank = len(U[0]) if U else 0
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        s = 0
                        for c in range(rank):
                            s += U[i * m3 + j][c] * V[j * p3 + k][c] * W[k * n3 + i][c]
                        if s != 1:
                            return False
            return True
        
        def find_3x3x3_decomposition(r, max_iter=1000, timeout=30):
            """Find a rank-r decomposition for 3x3x3 matrix multiplication."""
            start_time = time.time()
            
            triples = [(i, j, k) for i in range(n3) for j in range(m3) for k in range(p3)]
            idx_U = np.array([i * m3 + j for i, j, k in triples])
            idx_V = np.array([j * p3 + k for i, j, k in triples])
            idx_W = np.array([k * n3 + i for i, j, k in triples])
            
            for restart in range(5):
                if time.time() - start_time > timeout:
                    break
                
                # Initialize with random values
                U = np.random.uniform(-1, 1, (n3 * m3, r))
                V = np.random.uniform(-1, 1, (m3 * p3, r))
                W = np.random.uniform(-1, 1, (p3 * n3, r))
                
                lr = 0.05
                for it in range(max_iter):
                    if time.time() - start_time > timeout:
                        break
                    
                    U_sel = U[idx_U, :]
                    V_sel = V[idx_V, :]
                    W_sel = W[idx_W, :]
                    
                    S = np.sum(U_sel * V_sel * W_sel, axis=1)
                    diff = S - 1.0
                    error = np.sum(diff * diff)
                    
                    if error < 1e-12:
                        # Try rounding to integers or half-integers
                        for scale in [1, 2]:
                            U_r = np.round(U * scale) / scale
                            V_r = np.round(V * scale) / scale
                            W_r = np.round(W * scale) / scale
                            
                            if validate_3x3x3(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                                return U_r, V_r, W_r
                        
                        # If rounding fails but error is tiny, accept
                        if error < 1e-10:
                            return U, V, W
                        continue
                    
                    term_U = 2.0 * diff[:, None] * V_sel * W_sel
                    term_V = 2.0 * diff[:, None] * U_sel * W_sel
                    term_W = 2.0 * diff[:, None] * U_sel * V_sel
                    
                    grad_U = np.zeros((n3 * m3, r))
                    grad_V = np.zeros((m3 * p3, r))
                    grad_W = np.zeros((p3 * n3, r))
                    
                    np.add.at(grad_U, idx_U, term_U)
                    np.add.at(grad_V, idx_V, term_V)
                    np.add.at(grad_W, idx_W, term_W)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    lr *= 0.999
                    
                    # Periodically try to round
                    if it % 100 == 0 and error < 1e-6:
                        for scale in [1, 2]:
                            U_r = np.round(U * scale) / scale
                            V_r = np.round(V * scale) / scale
                            W_r = np.round(W * scale) / scale
                            
                            if validate_3x3x3(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                                return U_r, V_r, W_r
            
            return None
        
        # Try to find a rank-23 decomposition for 3x3x3
        # If we can't find it quickly, fall back to higher ranks
        
        best_3x3x3 = None
        for r in range(19, 28):  # Try ranks 19-27
            if time.time() > deadline - 10:
                break
            result = find_3x3x3_decomposition(r, max_iter=500, timeout=15)
            if result is not None:
                best_3x3x3 = (r, result)
                break
        
        if best_3x3x3 is None:
            return None
        
        r3, (U3, V3, W3) = best_3x3x3
        
        # Now convert the 3x3x3 decomposition to a 3x3x6 decomposition
        # The mapping: B[j,k] where k in [0,6) maps to dual number entry
        # k = 2*col + row_in_dual, so col = k//2, row_in_dual = k%2
        # B[j,k] = real[j, col] + ε * imag[j, row_in_dual]
        
        # For the decomposition to work over dual numbers:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In dual number form: C[i, 2*col+row] = sum_j A[i,j] * (real[j,col] + ε*imag[j,row])
        # = sum_j A[i,j]*real[j,col] + ε * sum_j A[i,j]*imag[j,row]
        
        # So we need:
        # C_real[i,col] = sum_j A[i,j] * real[j,col]  (standard 3x3 multiplication)
        # C_imag[i,row,col] = sum_j A[i,j] * imag[j,row,col]  (but imag is 3x3 per col)
        
        # Actually, let me reconsider the encoding.
        # B is 3x6. We want to view it as 3x3 over dual numbers.
        # Let B_dual[j, col] = B[j, 2*col] + ε * B[j, 2*col+1]
        
        # Then C[i, 2*col] = sum_j A[i,j] * B[j, 2*col]
        # C[i, 2*col+1] = sum_j A[i,j] * B[j, 2*col+1]
        
        # Using the dual number decomposition with rank r3:
        # For each term c in the 3x3x3 decomposition:
        # U3[i*m+j, c] * V3[j*p+k, c] * W3[k*n+i, c] = 1 for valid (i,j,k) in 3x3x3
        
        # To get the 3x3x6 decomposition, we create two terms per original term:
        # One for the real part (k even) and one for the imaginary part (k odd)
        
        # For term c in 3x3x3 decomposition with indices (i,j,k) where k in [0,3):
        # This corresponds to B[j, 2*k] and B[j, 2*k+1] in the 3x3x6 problem
        
        # Create new U, V, W matrices for 3x3x6 with rank 2*r3
        rank6 = 2 * r3
        U6 = np.zeros((n * m, rank6))
        V6 = np.zeros((m * p, rank6))
        W6 = np.zeros((p * n, rank6))
        
        for c in range(r3):
            # Real part term (index 2*c)
            # U6[i*m+j, 2*c] = U3[i*m+j, c]
            U6[:, 2*c] = U3[:, c]
            
            # V6[j*p+k, 2*c] = V3[j*p+k, c] where k in [0,3) maps to k in [0,6)
            # For the real part, we use k directly
            for j in range(m):
                for k in range(3):
                    V6[j * p + k, 2*c] = V3[j * p3 + k, c]
            
            # W6[k*n+i, 2*c] = W3[k*n+i, c]
            for k in range(3):
                for i in range(n):
                    W6[k * n + i, 2*c] = W3[k * n3 + i, c]
            
            # Imaginary part term (index 2*c+1)
            U6[:, 2*c+1] = U3[:, c]
            
            # For the imaginary part, we use k+3 (i.e., k in [3,6))
            for j in range(m):
                for k in range(3):
                    V6[j * p + (k + 3), 2*c+1] = V3[j * p3 + k, c]
            
            for k in range(3):
                for i in range(n):
                    W6[(k + 3) * n + i, 2*c+1] = W3[k * n3 + i, c]
        
        # Validate the 3x3x6 decomposition
        U6_list = U6.tolist()
        V6_list = V6.tolist()
        W6_list = W6.tolist()
        
        if validate(U6_list, V6_list, W6_list):
            return {"U": U6_list, "V": V6_list, "W": W6_list}
        
        # If validation fails, the 3x3x3 decomposition might not be exact
        # Try with the raw floats if they're close enough
        if np.max(np.abs(U6 - np.round(U6 * 2) / 2)) < 1e-8:
            U6_r = np.round(U6 * 2) / 2
            V6_r = np.round(V6 * 2) / 2
            W6_r = np.round(W6 * 2) / 2
            if validate(U6_r.tolist(), V6_r.tolist(), W6_r.tolist()):
                return {"U": U6_r.tolist(), "V": V6_r.tolist(), "W": W6_r.tolist()}
        
        return None
    
    # First try the dual number approach
    result = try_dual_number_approach()
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
