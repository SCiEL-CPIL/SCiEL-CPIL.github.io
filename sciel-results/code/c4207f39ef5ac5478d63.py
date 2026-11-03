import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Compute T_computed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - 1.0)) < tol
    
    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
        U shape: (4, 7) for A indices (i,j) with i,j in {0,1}
        V shape: (4, 7) for B indices (j,k) with j,k in {0,1}  
        W shape: (4, 7) for C indices (k,i) with k,i in {0,1}
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)*e
        # M3 = a*(f-h)
        # M4 = d*(g-e)
        # M5 = (a+b)*h
        # M6 = (c-a)*(e+f)
        # M7 = (b-d)*(g+h)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U coefficients for A = [[a,b],[c,d]]
        # Index: i*2+j
        U_coeffs = [
            {0: 1, 3: 1},    # M1: a, d
            {2: 1, 3: 1},    # M2: c, d
            {0: 1},          # M3: a
            {3: 1},          # M4: d
            {0: 1, 1: 1},    # M5: a, b
            {2: 1, 0: -1},   # M6: c, -a
            {1: 1, 3: -1}    # M7: b, -d
        ]
        
        # V coefficients for B = [[e,f],[g,h]]
        # Index: j*2+k
        V_coeffs = [
            {0: 1, 3: 1},    # M1: e, h
            {0: 1},          # M2: e
            {1: 1, 3: -1},   # M3: f, -h
            {2: 1, 0: -1},   # M4: g, -e
            {3: 1},          # M5: h
            {0: 1, 1: 1},    # M6: e, f
            {2: 1, 3: 1}     # M7: g, h
        ]
        
        # W coefficients for C = [[C00,C01],[C10,C11]]
        # Index: k*2+i
        W_coeffs = [
            {0: 1, 3: 1},    # M1: C00, C11
            {2: 1, 3: -1},   # M2: C10, -C11
            {1: 1, 3: 1},    # M3: C01, C11
            {0: 1, 2: 1},    # M4: C00, C10
            {0: -1, 1: 1},   # M5: -C00, C01
            {3: 1},          # M6: C11
            {0: 1}           # M7: C00
        ]
        
        for l in range(rank):
            for idx, coeff in U_coeffs[l].items():
                U[idx, l] += coeff
            for idx, coeff in V_coeffs[l].items():
                V[idx, l] += coeff
            for idx, coeff in W_coeffs[l].items():
                W[idx, l] += coeff
        
        return U, V, W
    
    def get_3x3x2_block():
        """
        Construct a 3x3x2 decomposition with rank 17.
        Uses Strassen for the 2x2x2 sub-block (i in {0,1}, j in {0,1}) 
        and schoolbook for the remaining parts.
        
        U shape: (9, rank) for A indices (i,j), i,j in {0,1,2}
        V shape: (6, rank) for B indices (j,k), j in {0,1,2}, k in {0,1}
        W shape: (6, rank) for C indices (k,i), k in {0,1}, i in {0,1,2}
        """
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        U_s, V_s, W_s = get_strassen_2x2x2()
        
        # First 7 terms: Strassen for i in {0,1}, j in {0,1}
        for l in range(7):
            # U: map local (i,j) in {0,1}^2 to global (i,j) in {0,1,2}^2
            for idx in range(4):
                if abs(U_s[idx, l]) > 1e-15:
                    i_local = idx // 2
                    j_local = idx % 2
                    U[i_local*3 + j_local, l] = U_s[idx, l]
            
            # V: map local (j,k) in {0,1}^2 to global (j,k), j in {0,1}, k in {0,1}
            for idx in range(4):
                if abs(V_s[idx, l]) > 1e-15:
                    j_local = idx // 2
                    k_local = idx % 2
                    V[j_local*2 + k_local, l] = V_s[idx, l]
            
            # W: map local (k,i) in {0,1}^2 to global (k,i), k in {0,1}, i in {0,1}
            for idx in range(4):
                if abs(W_s[idx, l]) > 1e-15:
                    k_local = idx // 2
                    i_local = idx % 2
                    W[k_local*3 + i_local, l] = W_s[idx, l]
        
        # Remaining 10 terms: schoolbook for the "cross" parts
        col = 7
        
        # i in {0,1}, j=2, k in {0,1}: 2*2 = 4 terms
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*2 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        
        # i=2, j in {0,1}, k in {0,1}: 2*2 = 4 terms
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*2 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        
        # i=2, j=2, k in {0,1}: 2 terms
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
        
        return U, V, W
    
    def try_reduce_to_rank_16(U17, V17, W17):
        """
        Try to reduce a rank-17 decomposition to rank 16 by finding
        a linear dependency among the 17 terms.
        """
        # Build the 17 tensor terms
        terms = []
        for c in range(17):
            u = U17[:, c].reshape(3, 3)
            v = V17[:, c].reshape(3, 2)
            w = W17[:, c].reshape(2, 3)
            t = np.zeros((3, 3, 2))
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        t[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            terms.append(t.flatten())
        
        terms = np.array(terms)  # shape (17, 18)
        
        # Try removing each term and see if it's in the span of the others
        for c_remove in range(17):
            if time_left() < 10:
                break
            
            others = [c for c in range(17) if c != c_remove]
            A = terms[others].T  # (18, 16)
            b = terms[c_remove]   # (18,)
            
            try:
                # Solve A @ x = b
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is exact
                if np.max(np.abs(A @ x - b)) < 1e-10:
                    # Try rounding to half-integers
                    x_rounded = np.round(x * 2) / 2
                    if np.max(np.abs(A @ x_rounded - b)) < 1e-10:
                        # Build the rank-16 decomposition
                        U_new = np.zeros((9, 16))
                        V_new = np.zeros((6, 16))
                        W_new = np.zeros((6, 16))
                        
                        col = 0
                        for c2 in others:
                            scale = x_rounded[col]
                            if abs(scale) > 1e-15:
                                U_new[:, col] = U17[:, c2] * scale
                                V_new[:, col] = V17[:, c2] * scale
                                W_new[:, col] = W17[:, c2] * scale
                            col += 1
                        
                        # Validate
                        T_check = np.zeros((3, 3, 2))
                        for c2 in range(16):
                            u = U_new[:, c2].reshape(3, 3)
                            v = V_new[:, c2].reshape(3, 2)
                            w = W_new[:, c2].reshape(2, 3)
                            for i in range(3):
                                for j in range(3):
                                    for k in range(2):
                                        T_check[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                        
                        if np.max(np.abs(T_check - 1.0)) < 1e-10:
                            return U_new, V_new, W_new
            except Exception:
                pass
        
        return None
    
    # Main logic:
    # 1. Get a 3x3x2 block (rank 17)
    # 2. Try to reduce to rank 16
    # 3. Replicate 4 times for 3x3x8
    # 4. Validate and return, or fall back
    
    U_block, V_block, W_block = get_3x3x2_block()
    rank_block = 17
    
    # Try to reduce to rank 16
    if time_left() > 20:
        reduced = try_reduce_to_rank_16(U_block, V_block, W_block)
        if reduced is not None:
            U_block, V_block, W_block = reduced
            rank_block = 16
    
    # Build the full 3x3x8 decomposition
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    for g in range(4):
        k0 = 2 * g
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U is the same for all groups
        U[:, col_start:col_end] = U_block
        
        # V: map local (j, k_local) to global (j, k0 + k_local)
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
        
        # W: map local (k_local, i) to global (k0 + k_local, i)
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Validate
    if validate(U, V, W):
        return result
    else:
        # Fallback to schoolbook
        return schoolbook()
