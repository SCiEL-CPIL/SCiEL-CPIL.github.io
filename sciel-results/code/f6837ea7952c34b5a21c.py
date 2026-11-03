"""Tensor decomposition for 4x4 * 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Use recursive Strassen decomposition with special handling for 2x2*2x3
    # 4x4 * 4x6: split into 2x2 blocks of 2x2 and 2x3
    # Each 2x2 * 2x3 sub-problem uses Strassen (7 multiplications)
    # Total rank = 7 * 7 = 49
    
    def build_decomp(n, m, p):
        """Build tensor decomposition for n x m * m x p matrix multiplication.
        Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) as numpy arrays.
        """
        # Base case: 1x1 * 1x1
        if n == 1 and m == 1 and p == 1:
            return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
        
        # Special case: 2x2 * 2x3 (use Strassen-like decomposition)
        if n == 2 and m == 2 and p == 3:
            # Split B into B_left (2x2) and B_right (2x1)
            # Use Strassen for A * B_left (2x2 * 2x2), then add A * B_right
            # Rank = 7 (Strassen) + 2 (for B_right) = 9
            rank = 9
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Strassen terms for A * B_left (2x2 * 2x2)
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # For B_right (2x1), use schoolbook: 2 multiplications
            # C13 = A11*B13 + A12*B23
            # C23 = A21*B13 + A22*B23
            
            # Strassen terms (columns 0-6)
            terms_strassen = [
                {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)]},
                {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1), (1,1,-1)]},
                {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'C': [(0,1,1), (1,1,1)]},
                {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'C': [(0,0,1), (1,0,1)]},
                {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1), (0,1,1)]},
                {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]},
                {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1)]}
            ]
            
            for t_idx, term in enumerate(terms_strassen):
                for (i, j, coeff) in term['A']:
                    U[i*m + j, t_idx] += coeff
                for (j, k, coeff) in term['B']:
                    V[j*p + k, t_idx] += coeff
                for (k, i, coeff) in term['C']:
                    W[k*n + i, t_idx] += coeff
            
            # Schoolbook terms for B_right (columns 7-8)
            # C13 = A11*B13 + A12*B23
            U[0*2+0, 7] = 1  # A11
            V[0*3+2, 7] = 1  # B13
            W[2*2+0, 7] = 1  # C13
            
            U[0*2+1, 8] = 1  # A12
            V[1*3+2, 8] = 1  # B23
            W[2*2+0, 8] = 1  # C13
            
            # C23 = A21*B13 + A22*B23
            U[1*2+0, 7] = 1  # A21
            U[1*2+1, 8] = 1  # A22
            W[2*2+1, 7] = 1  # C23
            W[2*2+1, 8] = 1  # C23
            
            return U, V, W
        
        # If any dimension is odd, fall back to schoolbook
        if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        W[k*n + i, col] = 1
                        col += 1
            return U, V, W
        
        # Recursive Strassen case
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        U_sub, V_sub, W_sub = build_decomp(n2, m2, p2)
        rank_sub = U_sub.shape[1]
        total_rank = 7 * rank_sub
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # The 7 Strassen terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each term: A_blocks [(block, coeff)], B_blocks, C_blocks
        terms = [
            {'A': [(11, 1), (22, 1)], 'B': [(11, 1), (22, 1)], 'C': [(11, 1), (22, 1)]},
            {'A': [(21, 1), (22, 1)], 'B': [(11, 1)], 'C': [(21, 1), (22, -1)]},
            {'A': [(11, 1)], 'B': [(12, 1), (22, -1)], 'C': [(12, 1), (22, 1)]},
            {'A': [(22, 1)], 'B': [(21, 1), (11, -1)], 'C': [(11, 1), (21, 1)]},
            {'A': [(11, 1), (12, 1)], 'B': [(22, 1)], 'C': [(11, -1), (12, 1)]},
            {'A': [(21, 1), (11, -1)], 'B': [(11, 1), (12, 1)], 'C': [(22, 1)]},
            {'A': [(12, 1), (22, -1)], 'B': [(21, 1), (22, 1)], 'C': [(11, 1)]}
        ]
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            
            # Process U entries
            for u_idx in range(n2 * m2):
                u_val = U_sub[u_idx]
                if np.all(u_val == 0):
                    continue
                i = u_idx // m2
                j = u_idx % m2
                for block, coeff in term['A']:
                    if block == 11:
                        g_u = i * m + j
                    elif block == 12:
                        g_u = i * m + (j + m2)
                    elif block == 21:
                        g_u = (i + n2) * m + j
                    elif block == 22:
                        g_u = (i + n2) * m + (j + m2)
                    else:
                        continue
                    U[g_u, base_col:base_col+rank_sub] += u_val * coeff
            
            # Process V entries
            for v_idx in range(m2 * p2):
                v_val = V_sub[v_idx]
                if np.all(v_val == 0):
                    continue
                j = v_idx // p2
                k = v_idx % p2
                for block, coeff in term['B']:
                    if block == 11:
                        g_v = j * p + k
                    elif block == 12:
                        g_v = j * p + (k + p2)
                    elif block == 21:
                        g_v = (j + m2) * p + k
                    elif block == 22:
                        g_v = (j + m2) * p + (k + p2)
                    else:
                        continue
                    V[g_v, base_col:base_col+rank_sub] += v_val * coeff
            
            # Process W entries
            for w_idx in range(p2 * n2):
                w_val = W_sub[w_idx]
                if np.all(w_val == 0):
                    continue
                k = w_idx // n2
                i = w_idx % n2
                for block, coeff in term['C']:
                    if block == 11:
                        g_w = k * n + i
                    elif block == 12:
                        g_w = (k + p2) * n + i
                    elif block == 21:
                        g_w = k * n + (i + n2)
                    elif block == 22:
                        g_w = (k + p2) * n + (i + n2)
                    else:
                        continue
                    W[g_w, base_col:base_col+rank_sub] += w_val * coeff
        
        return U, V, W
    
    U, V, W = build_decomp(n, m, p)
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
