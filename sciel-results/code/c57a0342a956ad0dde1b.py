"""Program for matrix multiplication 2x3x7 using HOSVD peeling."""
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 21
    dim_c = p * n  # 14
    
    # Build the matrix multiplication tensor T of shape (dim_a, dim_b, dim_c)
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid i, j, k
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Greedy rank-1 peeling using HOSVD
    def hosvd_peeling(T, max_rank):
        """Peel off rank-1 components from tensor T using HOSVD."""
        components = []
        T_rem = T.copy()
        for _ in range(max_rank):
            # Compute HOSVD: matrix unfoldings
            # Mode-1 unfolding: (dim_a, dim_b * dim_c)
            T1 = T_rem.reshape(dim_a, dim_b * dim_c)
            T2 = T_rem.reshape(dim_b, dim_c * dim_a)
            T3 = T_rem.reshape(dim_c, dim_a * dim_b)
            
            # SVD of each unfolding to get top singular vectors
            _, _, Vt1 = np.linalg.svd(T1, full_matrices=False)
            u = Vt1[0] if Vt1.shape[0] > 0 else np.ones(dim_a) / np.sqrt(dim_a)
            
            _, _, Vt2 = np.linalg.svd(T2, full_matrices=False)
            v = Vt2[0] if Vt2.shape[0] > 0 else np.ones(dim_b) / np.sqrt(dim_b)
            
            _, _, Vt3 = np.linalg.svd(T3, full_matrices=False)
            w = Vt3[0] if Vt3.shape[0] > 0 else np.ones(dim_c) / np.sqrt(dim_c)
            
            # Normalize
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            norm_w = np.linalg.norm(w)
            if norm_u > 1e-12: u = u / norm_u
            if norm_v > 1e-12: v = v / norm_v
            if norm_w > 1e-12: w = w / norm_w
            
            # Compute the scalar coefficient
            coeff = np.einsum('a,b,c,abc->', u, v, w, T_rem)
            
            if abs(coeff) < 1e-10:
                break
            
            components.append((coeff, u.copy(), v.copy(), w.copy()))
            
            # Subtract the rank-1 component
            T_rem -= coeff * np.einsum('a,b,c->abc', u, v, w)
            
            # Check if remaining tensor is negligible
            if np.linalg.norm(T_rem) < 1e-10:
                break
        
        return components
    
    # Try to find a decomposition with rank less than 42
    # Start with a reasonable max rank and try to project to half-integers
    best_result = None
    
    for max_rank in [41, 40, 39, 38, 37, 36, 35, 30, 25, 20]:
        components = hosvd_peeling(T, max_rank)
        actual_rank = len(components)
        
        if actual_rank == 0:
            continue
        
        # Build U, V, W from components
        U = np.zeros((dim_a, actual_rank))
        V = np.zeros((dim_b, actual_rank))
        W = np.zeros((dim_c, actual_rank))
        
        for r, (coeff, u, v, w) in enumerate(components):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
            # Distribute coeff: multiply into one factor (put it all in W for simplicity)
            W[:, r] *= coeff
        
        # Project to nearest half-integer grid
        U_half = np.round(2.0 * U) / 2.0
        V_half = np.round(2.0 * V) / 2.0
        W_half = np.round(2.0 * W) / 2.0
        
        # Verify exactness using the projected factors
        # Check: sum_r U_half[a,r] * V_half[b,r] * W_half[c,r] == T[a,b,c] for all a,b,c
        reconstructed = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        diff = np.abs(reconstructed - T)
        
        if np.max(diff) < 1e-9:
            # Convert to lists of integers/half-integers
            U_list = [[float(x) for x in U_half[i]] for i in range(dim_a)]
            V_list = [[float(x) for x in V_half[i]] for i in range(dim_b)]
            W_list = [[float(x) for x in W_half[i]] for i in range(dim_c)]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    # Fallback to schoolbook decomposition if no lower-rank exact decomposition found
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_a)]
        V = [[0] * rank for _ in range(dim_b)]
        W = [[0] * rank for _ in range(dim_c)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
