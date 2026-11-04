"""
Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication.
Strategy: Numerical search via ALS and random restarts to beat the constructive rank-50 bound.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    # Construct the target tensor T
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-8):
        """Verify if U, V, W exactly decompose T."""
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Using einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_approx, atol=tol)

    def get_constructive_solution():
        """
        Returns the known rank-50 Strassen-based decomposition.
        This serves as the baseline and fallback.
        """
        def get_strassen_2x2():
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # M1 = (A00 + A11) * (B00 + B11)
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # M2 = (A10 + A11) * B00
            U[1, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1; W[3, 1] = -1
            
            # M3 = A00 * (B01 - B11)
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1; W[3, 2] = 1
            
            # M4 = A11 * (B10 - B00)
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[2, 3] = 1
            
            # M5 = (A00 + A01) * B11
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1; W[1, 4] = 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            U[1, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1; W[3, 6] = 1
            
            return U, V, W

        U_list = []
        V_list = []
        W_list = []
        
        # Strassen blocks for k in {0,1}, {2,3}, {4,5} and j in {0,1}, {2,3}
        for k_start in [0, 2, 4]:
            for j_start in [0, 2]:
                U_s, V_s, W_s = get_strassen_2x2()
                
                for r in range(7):
                    u_val = U_s[:, r]
                    v_val = V_s[:, r]
                    w_val = W_s[:, r]
                    
                    U_col = np.zeros(n*m)
                    V_col = np.zeros(m*p)
                    W_col = np.zeros(p*n)
                    
                    for li in range(4):
                        if u_val[li] != 0:
                            i_local = li // 2
                            j_local = li % 2
                            U_col[i_local * m + (j_start + j_local)] = u_val[li]
                            
                    for li in range(4):
                        if v_val[li] != 0:
                            j_local = li // 2
                            k_local = li % 2
                            V_col[(j_start + j_local) * p + (k_start + k_local)] = v_val[li]
                            
                    for li in range(4):
                        if w_val[li] != 0:
                            i_local = li // 2
                            k_local = li % 2
                            W_col[(k_start + k_local) * n + i_local] = w_val[li]
                            
                    U_list.append(U_col)
                    V_list.append(V_col)
                    W_list.append(W_col)
                    
        # Direct terms for k=6
        for i in range(n):
            for j in range(m):
                U_col = np.zeros(n*m)
                V_col = np.zeros(m*p)
                W_col = np.zeros(p*n)
                
                U_col[i * m + j] = 1
                V_col[j * p + 6] = 1
                W_col[6 * n + i] = 1
                
                U_list.append(U_col)
                V_list.append(V_col)
                W_list.append(W_col)
                
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        
        return U, V, W

    # Get the baseline solution
    U_base, V_base, W_base = get_constructive_solution()
    best_rank = U_base.shape[1]
    best_U, best_V, best_W = U_base, V_base, W_base
    
    # Search for better rank
    # We try ranks from best_rank - 1 down to a theoretical lower bound.
    # Lower bound for 2x4x4x7 is at least max(2*4, 4*7, 7*2) = 28?
    # Actually, rank is at least the dimension of the output space if inputs are full rank?
    # Let's try ranks 49, 48, 47, 46...
    
    # ALS function to refine a decomposition
    def als_refine(U, V, W, T, iterations=50, tol=1e-10):
        """
        Alternating Least Squares to minimize ||T - U V W||_F.
        """
        # Reshape T to matrix forms for ALS
        # T_1: (n*m, m*p*p*n) ? No, T is 3-way.
        # We solve for U given V, W:
        # T[a, :, :] = U[:, r] * (V * W)_{r, :, :}
        # Let M_VW be a matrix of size (m*p*p*n, rank) where M_VW[b*c, r] = V[b, r] W[c, r]
        # Then T_1[a, b*c] = sum_r U[a, r] M_VW[b*c, r]
        # So T_1 = U M_VW^T
        
        # Let's construct the matricizations
        # T_1: shape (n*m, m*p * p*n)
        # T_2: shape (m*p, n*m * p*n)
        # T_3: shape (p*n, n*m * m*p)
        
        T_1 = T.reshape(n*m, m*p * p*n)
        T_2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
        T_3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
        
        for _ in range(iterations):
            # Update U
            # M_VW: (m*p * p*n, rank)
            # M_VW[b*c, r] = V[b, r] * W[c, r]
            # We can compute this efficiently
            M_VW = np.einsum('br,cr->bc r', V, W).reshape(m*p * p*n, -1)
            # Solve U M_VW^T = T_1
            # U = T_1 M_VW (M_VW^T M_VW)^-1
            try:
                U = np.linalg.lstsq(M_VW.T, T_1.T, rcond=None)[0].T
            except:
                pass
                
            # Update V
            # M_UW: (n*m * p*n, rank)
            # M_UW[a*c, r] = U[a, r] * W[c, r]
            M_UW = np.einsum('ar,cr->ac r', U, W).reshape(n*m * p*n, -1)
            try:
                V = np.linalg.lstsq(M_UW.T, T_2.T, rcond=None)[0].T
            except:
                pass
                
            # Update W
            # M_UV: (n*m * m*p, rank)
            # M_UV[a*b, r] = U[a, r] * V[b, r]
            M_UV = np.einsum('ar,br->ab r', U, V).reshape(n*m * m*p, -1)
            try:
                W = np.linalg.lstsq(M_UV.T, T_3.T, rcond=None)[0].T
            except:
                pass
                
            # Check convergence
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - T_approx)
            if err < tol:
                break
                
        return U, V, W, err

    def try_rank(rank):
        """
        Attempt to find a decomposition of a specific rank.
        """
        if rank <= 0:
            return None
            
        # Initialize with random noise
        # We can also try initializing from the baseline by projecting/truncating
        # But random restarts are more robust for finding different local minima
        
        for restart in range(5):
            if time.time() > deadline:
                return None
                
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Refine
            U, V, W, err = als_refine(U, V, W, T, iterations=100)
            
            if err < 1e-8:
                # Verify
                if verify(U, V, W):
                    # Try to clean up to integers/half-integers if possible
                    # For now, return as is
                    return U, V, W
                    
        return None

    # Search loop
    # Start from best_rank - 1 down to 40 (arbitrary lower bound for search)
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline:
            break
            
        result = try_rank(rank)
        if result:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            # If we found a lower rank, we might want to continue searching lower
            # But given time constraints, we return the first improvement found?
            # No, we should continue to find the minimum.
            # However, ALS is stochastic. Finding one low rank solution doesn't guarantee
            # that a lower one exists or is easy to find.
            # Let's do a few more iterations for the new best rank to see if we can go lower?
            # The loop continues automatically.

    # Final verification and formatting
    if verify(best_U, best_V, best_W):
        # Attempt to convert to integers/half-integers if close
        # This is a heuristic cleanup
        U_out = np.round(best_U * 2) / 2
        V_out = np.round(best_V * 2) / 2
        W_out = np.round(best_W * 2) / 2
        
        if verify(U_out, V_out, W_out):
            return {
                "U": U_out.tolist(),
                "V": V_out.tolist(),
                "W": W_out.tolist()
            }
        else:
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
    else:
        # Fallback to baseline if something went wrong (shouldn't happen)
        return {
            "U": U_base.tolist(),
            "V": V_base.tolist(),
            "W": W_base.tolist()
        }
