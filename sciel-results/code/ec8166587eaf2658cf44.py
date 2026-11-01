# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank is 14 (two independent 2x2x2 Strassen decompositions)
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define Strassen decomposition for 2x2 * 2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
    # W indices for block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
    
    # Strassen terms (u, v, w) where u,v,w are vectors of length 4
    # M1 = (a+d)(e+i)
    # M2 = (c+d)(e)
    # M3 = a(f-i)
    # M4 = d(g-e)
    # M5 = (a+b)(i)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+i)
    
    # Output reconstruction:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the 7 terms as lists of (u_coeff, v_coeff, w_coeff)
    # Each coeff is a dict {index: value} or a vector
    
    def get_strassen_terms():
        terms = []
        # Term 1: (a+d)(e+i)
        u1 = {0: 1, 3: 1}
        v1 = {0: 1, 3: 1}
        w1 = {0: 1, 3: 1} # C11 += 1, C22 += 1
        terms.append((u1, v1, w1))
        
        # Term 2: (c+d)(e)
        u2 = {2: 1, 3: 1}
        v2 = {0: 1}
        w2 = {2: 1, 3: -1} # C21 += 1, C22 -= 1
        terms.append((u2, v2, w2))
        
        # Term 3: a(f-i)
        u3 = {0: 1}
        v3 = {1: 1, 3: -1}
        w3 = {1: 1, 3: 1} # C12 += 1, C22 += 1
        terms.append((u3, v3, w3))
        
        # Term 4: d(g-e)
        u4 = {3: 1}
        v4 = {2: 1, 0: -1}
        w4 = {0: 1, 2: 1} # C11 += 1, C21 += 1
        terms.append((u4, v4, w4))
        
        # Term 5: (a+b)(i)
        u5 = {0: 1, 1: 1}
        v5 = {3: 1}
        w5 = {0: -1, 1: 1} # C11 -= 1, C12 += 1
        terms.append((u5, v5, w5))
        
        # Term 6: (c-a)(e+f)
        u6 = {2: 1, 0: -1}
        v6 = {0: 1, 1: 1}
        w6 = {3: 1} # C22 += 1
        terms.append((u6, v6, w6))
        
        # Term 7: (b-d)(g+i)
        u7 = {1: 1, 3: -1}
        v7 = {2: 1, 3: 1}
        w7 = {0: 1} # C11 += 1
        terms.append((u7, v7, w7))
        
        return terms

    terms = get_strassen_terms()
    
    # Apply to Block 1 (cols 0,1 of B, cols 0,1 of C)
    # B indices: 0,1,4,5 (row 0: 0,1; row 1: 4,5)
    # C indices: 0,1,2,3 (col 0: 0,2; col 1: 1,3) -> Wait, W[k*n+i]
    # C[0,0] -> W[0*2+0]=0
    # C[0,1] -> W[1*2+0]=2
    # C[1,0] -> W[0*2+1]=1
    # C[1,1] -> W[1*2+1]=3
    
    # Block 1 mapping:
    # B local (r,c) -> Global V index: r*4 + c
    # C local (r,c) -> Global W index: c*2 + r
    
    for i, (u, v, w) in enumerate(terms):
        # Add to U
        for idx, val in u.items():
            U[idx, i] += val
        # Add to V (offset 0)
        for idx, val in v.items():
            V[idx, i] += val
        # Add to W (offset 0)
        for idx, val in w.items():
            # W index mapping: local (r,c) -> global c*2+r
            # In terms, w keys are 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # So 0->0, 1->2, 2->1, 3->3
            w_global = [0, 2, 1, 3][idx]
            W[w_global, i] += val
            
    # Apply to Block 2 (cols 2,3 of B, cols 2,3 of C)
    # B indices: 2,3,6,7
    # C indices: 4,5,6,7
    # W mapping:
    # C[0,2] -> W[2*2+0]=4
    # C[0,3] -> W[3*2+0]=6
    # C[1,2] -> W[2*2+1]=5
    # C[1,3] -> W[3*2+1]=7
    
    for i, (u, v, w) in enumerate(terms):
        d = 7 + i
        # Add to U
        for idx, val in u.items():
            U[idx, d] += val
        # Add to V (offset 2)
        for idx, val in v.items():
            V[idx + 2, d] += val
        # Add to W (offset 2 in cols)
        for idx, val in w.items():
            # Local (r,c) -> Global W index: (c+2)*2 + r
            # 0:(0,0) -> 4*2+0 = 4? No.
            # C[0,2] is row 0, col 2. W[2*2+0] = 4.
            # C[0,3] is row 0, col 3. W[3*2+0] = 6.
            # C[1,2] is row 1, col 2. W[2*2+1] = 5.
            # C[1,3] is row 1, col 3. W[3*2+1] = 7.
            w_global = [4, 6, 5, 7][idx]
            W[w_global, d] += val

    # Verify
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
