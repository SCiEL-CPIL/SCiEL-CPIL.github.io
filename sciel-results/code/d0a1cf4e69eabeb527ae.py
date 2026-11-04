import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    n, m, p = 2, 4, 4
    
    # Fallback: The known Strassen-like decomposition of rank 28
    def make_strassen_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            # M1
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1

            # M2
            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1

            # M3
            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1

            # M4
            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1

            # M5
            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1

            # M6
            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1

            # M7
            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1

        # Block 1: A_L B_TL -> C_LL
        X1 = [0, 1, 4, 5]
        Y1 = [0, 1, 4, 5]
        Z1 = [0, 2, 1, 3]
        add_block(0, X1, Y1, Z1)

        # Block 2: A_R B_BL -> C_LR
        X2 = [2, 3, 6, 7]
        Y2 = [8, 9, 12, 13]
        Z2 = [0, 2, 1, 3]
        add_block(7, X2, Y2, Z2)

        # Block 3: A_L B_TR -> C_RL
        X3 = [0, 1, 4, 5]
        Y3 = [2, 3, 6, 7]
        Z3 = [4, 6, 5, 7]
        add_block(14, X3, Y3, Z3)

        # Block 4: A_R B_BR -> C_RR
        X4 = [2, 3, 6, 7]
        Y4 = [10, 11, 14, 15]
        Z4 = [4, 6, 5, 7]
        add_block(21, X4, Y4, Z4)

        return {"U": U, "V": V, "W": W}

    # Greedy Search for lower rank
    def try_greedy_search(max_steps=27, n_restarts=5):
        # Build Tensor T (8, 16, 8)
        T = np.zeros((8, 16, 8), dtype=float)
        for j in range(4):
            for i in range(2):
                for k in range(4):
                    a = 4 * i + j
                    b = 4 * j + k
                    c = 2 * k + i
                    T[a, b, c] = 1.0
        
        rng = np.random.RandomState(42)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                return None
            
            R = T.copy()
            components = []
            
            # For restarts > 0, add small random perturbation to escape local minima
            if restart > 0:
                R += rng.randn(*R.shape) * 0.01
            
            for step in range(max_steps):
                if time.time() > deadline - 10:
                    return None
                
                # 1. Find w (mode 3)
                R3 = R.reshape(8 * 16, 8)
                try:
                    _, _, Vh = np.linalg.svd(R3)
                    w = Vh[0] # Leading right singular vector
                except:
                    return None
                
                # 2. Contract to get R' (8, 16)
                R_prime = np.einsum('abc,c->ab', R, w)
                
                # 3. Find v (mode 2)
                try:
                    _, _, Vh2 = np.linalg.svd(R_prime)
                    v = Vh2[0] # Leading right singular vector
                except:
                    return None
                
                # 4. Contract to get R'' (8)
                R_double_prime = np.einsum('ab,b->a', R_prime, v)
                
                # 5. Find u (mode 1)
                u_norm = np.linalg.norm(R_double_prime)
                if u_norm < 1e-10:
                    break
                u = R_double_prime / u_norm
                
                # 6. Calculate alpha
                # <R, u v w>
                inner = np.einsum('abc,a,b,c->', R, u, v, w)
                norm_sq = (u_norm**2) * (np.linalg.norm(v)**2) * (np.linalg.norm(w)**2)
                if norm_sq < 1e-10:
                    break
                alpha = inner / norm_sq
                
                # 7. Update R
                R -= alpha * np.einsum('a,b,c->abc', u, v, w)
                
                # Store component
                components.append((u, v, w, alpha))
                
                # Check residual
                if np.linalg.norm(R) < 1e-8:
                    # Found a decomposition of rank step+1
                    # Attempt rational reconstruction
                    result = reconstruct_integer_solution(components)
                    if result is not None:
                        return result
                    
        return None

    def reconstruct_integer_solution(components):
        # This is a simplified reconstruction. 
        # We assume the coefficients are integers or half-integers.
        # We will try to scale each component to integers.
        
        rank = len(components)
        if rank == 0:
            return None
        
        U = np.zeros((8, rank))
        V = np.zeros((16, rank))
        W = np.zeros((8, rank))
        
        for idx, (u, v, w, alpha) in enumerate(components):
            # Try to find integer/half-integer representation
            # We need U[:,idx] * V[:,idx] * W[:,idx] to approximate alpha * u * v * w
            
            # Scale factor: try to make all entries integers or half-integers
            # The product of three vectors should give the tensor component
            
            # Strategy: distribute alpha among the three vectors
            # Try different distributions
            best_scale = None
            best_error = float('inf')
            
            for scale_u in [1, 2, 4, 8]:
                for scale_v in [1, 2, 4, 8]:
                    for scale_w in [1, 2, 4, 8]:
                        # u_scaled = u / scale_u
                        # v_scaled = v / scale_v  
                        # w_scaled = alpha * w / scale_w
                        # Check if these are close to integers or half-integers
                        
                        u_scaled = u / scale_u
                        v_scaled = v / scale_v
                        w_scaled = alpha * w / scale_w
                        
                        # Check if entries are close to multiples of 0.5
                        u_close = np.allclose(u_scaled * 2, np.round(u_scaled * 2), atol=0.01)
                        v_close = np.allclose(v_scaled * 2, np.round(v_scaled * 2), atol=0.01)
                        w_close = np.allclose(w_scaled * 2, np.round(w_scaled * 2), atol=0.01)
                        
                        if u_close and v_close and w_close:
                            # Calculate error
                            error = np.linalg.norm(u_scaled - u/scale_u) + \
                                   np.linalg.norm(v_scaled - v/scale_v) + \
                                   np.linalg.norm(w_scaled - alpha*w/scale_w)
                            if error < best_error:
                                best_error = error
                                best_scale = (scale_u, scale_v, scale_w)
            
            if best_scale is None:
                # Fallback: use default scaling
                best_scale = (1, 1, 1)
            
            scale_u, scale_v, scale_w = best_scale
            U[:, idx] = np.round(u / scale_u * 2) / 2
            V[:, idx] = np.round(v / scale_v * 2) / 2
            W[:, idx] = np.round(alpha * w / scale_w * 2) / 2
        
        # Verify the decomposition
        T_check = np.zeros((8, 16, 8))
        for r in range(rank):
            T_check += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        if not np.allclose(T_check, T, atol=0.1):
            return None
            
        # Convert to lists
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()
        
        return {"U": U_out, "V": V_out, "W": W_out}

    # Try greedy search
    result = try_greedy_search()
    
    if result is not None:
        return result
    else:
        return make_strassen_solution()
