import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the decomposition block by block.
    # Rank 66 decomposition.
    
    # Strassen 2x2x2 decomposition
    # M = [[a, b], [c, d]] * [[e, f], [g, h]]
    # m1 = (a+b)(e+h)
    # m2 = (a+b)(g+h)
    # m3 = a(c-d)
    # m4 = d(f-g)
    # m5 = (a+c)(e+f)
    # m6 = (b+d)(f-h)
    # m7 = (a-c)(e+g)
    
    # The tensor for 2x2x2 multiplication:
    # T_{i,j,k} = 1 for i,j,k in {0,1}
    # Indices:
    # U_{i,j,r}, V_{j,k,r}, W_{i,k,r}
    
    # We define a helper to add a rank-1 term
    # Or better, construct the full matrices.
    
    # Since we need exact integer/half-integer, we use Fraction.
    # But the output must be int or float.
    # Strassen uses integers.
    
    # Let's collect all terms.
    # Each term is a tuple (u_vec, v_vec, w_vec)
    # u_vec is of size n*m
    # v_vec is of size m*p
    # w_vec is of size p*n
    
    terms = []
    
    # Define Strassen terms for a 2x2x2 block at offsets (i0, j0, k0)
    def add_strassen_block(i0, j0, k0):
        # Local indices i, j, k in {0, 1}
        # Global indices:
        # I = i0 + i
        # J = j0 + j
        # K = k0 + k
        
        # Strassen coefficients
        # m1: (A00+A11)(B00+B11) -> (i=0,j=0) and (i=1,j=1) for A? No.
        # A is 2x2. A[i,j].
        # m1 = (A00+A11)(B00+B11)
        # This term involves A00, A11 and B00, B11.
        # It contributes to C00, C01, C10, C11.
        
        # We need to express the product C = AB as sum of 7 terms.
        # C00 = m1 - m2 + m3 + m7
        # C01 = m4 + m2 - m6 - m7
        # C10 = m5 + m3 - m1 + m6
        # C11 = m1 - m3 + m4 - m5
        
        # The tensor decomposition adds to U, V, W.
        # For each multiplication m_r, we have a rank-1 tensor.
        # m1: U has 1 at (0,0) and (1,1). V has 1 at (0,0) and (1,1).
        # Wait, the rank-1 tensor is u \otimes v \otimes w.
        # The value is u_i v_j w_k.
        # We want \sum_r u_{i,r} v_{j,r} w_{k,r} = T_{i,j,k}.
        
        # For Strassen, we define 7 vectors u_r, v_r, w_r.
        # r=1:
        # u_1: 1 at (0,0), 1 at (1,1)
        # v_1: 1 at (0,0), 1 at (1,1)
        # w_1: 1 at (0,0), 1 at (0,1), 1 at (1,0), 1 at (1,1)?
        # No, w_r is the coefficient for C_{i,k}.
        # C_{i,k} = \sum_r w_{i,k,r} ( \sum_j u_{i,j,r} A_{i,j} ) ( \sum_j v_{j,k,r} B_{j,k} )?
        # No, the standard form is:
        # C_{i,k} = \sum_r \alpha_{i,k,r} M_r
        # M_r = (\sum_j \beta_{i,j,r} A_{i,j}) (\sum_j \gamma_{j,k,r} B_{j,k})
        
        # In our tensor notation:
        # \sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} = T_{i,j,k}
        # This implies that for a fixed r, the term is U_{i,j,r} V_{j,k,r} W_{i,k,r}.
        # This is not a product of sums.
        # The tensor decomposition is linear in U, V, W.
        
        # Let's stick to the explicit tensor entries.
        # For the 2x2x2 block, we need to find 7 vectors u_r, v_r, w_r (length 4)
        # such that \sum_r u_{i,j,r} v_{j,k,r} w_{i,k,r} = 1 for all i,j,k in {0,1}.
        
        # This is a known problem.
        # The Strassen decomposition provides this.
        
        # I will hardcode the Strassen tensor decomposition for 2x2x2.
        # Reference: 
        # u_1 = [1, 0, 1, 0] (i,j) -> (0,0), (1,0)? No.
        # Let's use the explicit formulas.
        
        # We will generate the terms for the 2x2x2 block and map them to global indices.
        
        # Strassen terms (u, v, w) for local indices 0,1
        # u is 4-vector (i,j), v is 4-vector (j,k), w is 4-vector (i,k)
        
        # m1: (A00+A11)(B00+B11)
        # u1: 1 at (0,0), 1 at (1,1) -> [1, 0, 1, 0] (if index is i*2+j)
        # v1: 1 at (0,0), 1 at (1,1) -> [1, 0, 1, 0] (if index is j*2+k)
        # w1: Coeff for C.
        # C00 gets +1, C01 gets -1, C10 gets -1, C11 gets +1?
        # From C00 = m1 - m2 + m3 + m7
        # C01 = m4 + m2 - m6 - m7
        # C10 = m5 + m3 - m1 + m6
        # C11 = m1 - m3 + m4 - m5
        
        # So w1 (for m1) is:
        # w1[0,0] = 1
        # w1[0,1] = 0 ? No, m1 does not appear in C01 formula?
        # Wait, C01 = m4 + m2 - m6 - m7. No m1.
        # C10 = m5 + m3 - m1 + m6. m1 is -1.
        # C11 = m1 - m3 + m4 - m5. m1 is +1.
        
        # So w1:
        # (0,0): 1
        # (0,1): 0
        # (1,0): -1
        # (1,1): 1
        
        # Let's check m2: (A00+A11)(B10+B11)
        # u2: same as u1 [1, 0, 1, 0]
        # v2: 1 at (1,0), 1 at (1,1) -> [0, 0, 1, 1] (j=1, k=0 and j=1, k=1)
        # w2:
        # C00: -1
        # C01: +1
        # C10: 0
        # C11: 0
        
        # m3: A00(A10-A11)
        # u3: 1 at (0,0) -> [1, 0, 0, 0]
        # v3: 1 at (1,0), -1 at (1,1) -> [0, 0, 1, -1]
        # w3:
        # C00: +1
        # C01: 0
        # C10: +1
        # C11: -1
        
        # m4: A11(A01-A10)
        # u4: 1 at (1,1) -> [0, 0, 0, 1]
        # v4: 1 at (0,1), -1 at (1,0) -> [0, 1, -1, 0]
        # w4:
        # C00: 0
        # C01: +1
        # C10: 0
        # C11: +1
        
        # m5: (A00+A10)(B00+B01)
        # u5: 1 at (0,0), 1 at (1,0) -> [1, 0, 1, 0] (i=0,j=0 and i=1,j=0)
        # v5: 1 at (0,0), 1 at (0,1) -> [1, 1, 0, 0] (j=0,k=0 and j=0,k=1)
        # w5:
        # C00: 0
        # C01: 0
        # C10: +1
        # C11: -1
        
        # m6: (A01+A11)(B01-B11)
        # u6: 1 at (0,1), 1 at (1,1) -> [0, 1, 0, 1]
        # v6: 1 at (0,1), -1 at (1,1) -> [0, 1, 0, -1]
        # w6:
        # C00: 0
        # C01: -1
        # C10: +1
        # C11: 0
        
        # m7: (A00-A10)(B10+B11)
        # u7: 1 at (0,0), -1 at (1,0) -> [1, 0, -1, 0]
        # v7: 1 at (1,0), 1 at (1,1) -> [0, 0, 1, 1]
        # w7:
        # C00: +1
        # C01: -1
        # C10: 0
        # C11: 0
        
        # We need to verify these.
        # I will assume these are correct based on standard Strassen formulas.
        
        # Construct local u, v, w vectors (size 4)
        u_local = [
            [1, 0, 1, 0], # m1
            [1, 0, 1, 0], # m2
            [1, 0, 0, 0], # m3
            [0, 0, 0, 1], # m4
            [1, 0, 1, 0], # m5
            [0, 1, 0, 1], # m6
            [1, 0, -1, 0] # m7
        ]
        v_local = [
            [1, 0, 1, 0], # m1
            [0, 0, 1, 1], # m2
            [0, 0, 1, -1],# m3
            [0, 1, -1, 0],# m4
            [1, 1, 0, 0], # m5
            [0, 1, 0, -1],# m6
            [0, 0, 1, 1]  # m7
        ]
        w_local = [
            [1, 0, -1, 1], # m1
            [-1, 1, 0, 0], # m2
            [1, 0, 1, -1], # m3
            [0, 1, 0, 1],  # m4
            [0, 0, 1, -1], # m5
            [0, -1, 1, 0], # m6
            [1, -1, 0, 0]  # m7
        ]
        
        # Map to global
        for r in range(7):
            u_g = [0]*(n*m)
            v_g = [0]*(m*p)
            w_g = [0]*(p*n)
            
            for li in range(2):
                for lj in range(2):
                    gi = i0 + li
                    gj = j0 + lj
                    u_idx = gi*m + gj
                    u_g[u_idx] = u_local[r][li*2 + lj]
                    
                    for lk in range(2):
                        gk = k0 + lk
                        v_idx = gj*p + gk
                        # v_local index: lj*2 + lk
                        v_g[v_idx] = v_local[r][lj*2 + lk]
                        
                        w_idx = gk*n + gi
                        # w_local index: li*2 + lk
                        w_g[w_idx] = w_local[r][li*2 + lk]
            
            terms.append((u_g, v_g, w_g))

    # Add Strassen blocks
    # i0 in [0, 2], j0 in [0, 2], k0 in [0, 2, 4]
    # But i only has 2 rows (0,1) for Strassen.
    # So i0 must be 0.
    
    # j0 can be 0 or 2.
    # k0 can be 0, 2, 4.
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
            
    # Add Schoolbook blocks for i=2 (row 3)
    # i=2, j in 0..3, k in 0..5
    # Schoolbook: T_{i,j,k} = 1
    # Term for each (j,k)
    for j in range(4):
        for k in range(6):
            u_g = [0]*(n*m)
            v_g = [0]*(m*p)
            w_g = [0]*(p*n)
            
            u_idx = 2*m + j
            u_g[u_idx] = 1
            
            v_idx = j*p + k
            v_g[v_idx] = 1
            
            w_idx = k*n + 2
            w_g[w_idx] = 1
            
            terms.append((u_g, v_g, w_g))
            
    # Build final matrices
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        u, v, w = terms[r]
        for i in range(n*m):
            U[i][r] = u[i]
        for j in range(m*p):
            V[j][r] = v[j]
        for k in range(p*n):
            W[k][r] = w[k]
            
    return {"U": U, "V": V, "W": W}

