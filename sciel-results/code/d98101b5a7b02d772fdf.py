"""Strassen-based rank-49 decomposition for 2x2 * 2x14 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy:
    - Treat the 2x14 matrix B as 7 independent 2x2 blocks (columns 0-1, 2-3, ..., 12-13).
    - Apply the 7-multiplication Strassen algorithm to each 2x2 block.
    - This gives rank = 7 * 7 = 49.
    - The output C is 2x14, with C[:, 2l] = 2x2 Strassen result on block l, column 0.
    - The output C is 2x14, with C[:, 2l+1] = 2x2 Strassen result on block l, column 1.
    
    Index conventions:
    - U shape (n*m, rank) = (4, 49), row index i*m + j
    - V shape (m*p, rank) = (28, 49), row index j*p + k
    - W shape (p*n, rank) = (28, 49), row index k*n + i
    
    Strassen formulas for 2x2 matrices:
    Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    C = A*B = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    
    Strassen's 7 products:
    p1 = (a+d)(e+h)  -> used in C00, C11
    p2 = d(g+h)       -> used in C10, C11
    p3 = (a+b)h       -> used in C01, C11
    p4 = (c+d)e       -> used in C00, C10
    p5 = (a-c)(h-g)   -> used in C00, C01, C11
    p6 = (b-c)e       -> used in C00, C10
    p7 = (a-b)f       -> used in C01, C11
    
    Then:
    C00 = p1 + p4 - p5 + p6
    C01 = p3 - p5 + p7
    C10 = p4 + p2 - p6
    C11 = p1 + p3 - p2 + p5
    """
    import time as _time
    _start = _time.time()
    _deadline = _start + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    
    num_blocks = p // 2  # 7 blocks of 2x2
    rank = 7 * num_blocks  # 49
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen products for one 2x2 block:
    # p1 = (a+d)(e+h): U: a,d=1; V: e,h=1; W: C00,C11
    # p2 = d(g+h):     U: d=1;    V: g,h=1; W: C10,C11
    # p3 = (a+b)h:     U: a,b=1;  V: h=1;   W: C01,C11
    # p4 = (c+d)e:     U: c,d=1;  V: e=1;   W: C00,C10
    # p5 = (a-c)(h-g): U: a=1,c=-1; V: h=1,g=-1; W: C00,C01
    # p6 = (b-c)e:     U: b=1,c=-1; V: e=1; W: C00,C10
    # p7 = (a-b)f:     U: a=1,b=-1; V: f=1; W: C01,C11
    
    # U vectors (length 4, indices: a=0, b=1, c=2, d=3)
    U_p1 = np.array([1, 0, 0, 1])    # a+d
    U_p2 = np.array([0, 0, 0, 1])    # d
    U_p3 = np.array([1, 1, 0, 0])    # a+b
    U_p4 = np.array([0, 0, 1, 1])    # c+d
    U_p5 = np.array([1, 0, -1, 0])   # a-c
    U_p6 = np.array([0, 1, -1, 0])   # b-c
    U_p7 = np.array([1, -1, 0, 0])   # a-b
    
    U_strassen = [U_p1, U_p2, U_p3, U_p4, U_p5, U_p6, U_p7]
    
    # For each block l (0..6), and each Strassen product q (0..6):
    for l in range(num_blocks):
        # Block l corresponds to columns 2l and 2l+1 of B
        # B has rows j=0,1 and columns k=0..13
        # For block l: B[0, 2l] = e, B[0, 2l+1] = f, B[1, 2l] = g, B[1, 2l+1] = h
        
        for q in range(7):
            col = l * 7 + q
            
            U[:, col] = U_strassen[q]
            
            # V vector (length 28, index j*p + k)
            v = np.zeros(mp)
            
            if q == 0:  # p1: e+h
                v[0 * p + 2*l] = 1      # e = B[0, 2l]
                v[1 * p + 2*l + 1] = 1  # h = B[1, 2l+1]
            elif q == 1:  # p2: g+h
                v[1 * p + 2*l] = 1      # g = B[1, 2l]
                v[1 * p + 2*l + 1] = 1  # h = B[1, 2l+1]
            elif q == 2:  # p3: h
                v[1 * p + 2*l + 1] = 1  # h = B[1, 2l+1]
            elif q == 3:  # p4: e
                v[0 * p + 2*l] = 1      # e = B[0, 2l]
            elif q == 4:  # p5: h-g
                v[1 * p + 2*l] = -1     # g = B[1, 2l]
                v[1 * p + 2*l + 1] = 1  # h = B[1, 2l+1]
            elif q == 5:  # p6: e
                v[0 * p + 2*l] = 1      # e = B[0, 2l]
            elif q == 6:  # p7: f
                v[0 * p + 2*l + 1] = 1  # f = B[0, 2l+1]
            
            V[:, col] = v
            
            # W vector (length 28, index k*n + i)
            w = np.zeros(pn)
            
            # C00: k=2l, i=0 -> index 2l*n + 0 = 4l
            # C01: k=2l+1, i=0 -> index (2l+1)*n + 0 = 4l+2
            # C10: k=2l, i=1 -> index 2l*n + 1 = 4l+1
            # C11: k=2l+1, i=1 -> index (2l+1)*n + 1 = 4l+3
            
            C00_idx = 4*l
            C01_idx = 4*l + 2
            C10_idx = 4*l + 1
            C11_idx = 4*l + 3
            
            if q == 0:  # p1 -> +C00 +C11
                w[C00_idx] = 1
                w[C11_idx] = 1
            elif q == 1:  # p2 -> +C10 -C11
                w[C10_idx] = 1
                w[C11_idx] = -1
            elif q == 2:  # p3 -> +C01 +C11
                w[C01_idx] = 1
                w[C11_idx] = 1
            elif q == 3:  # p4 -> +C00 +C10
                w[C00_idx] = 1
                w[C10_idx] = 1
            elif q == 4:  # p5 -> -C00 -C01 +C11
                w[C00_idx] = -1
                w[C01_idx] = -1
                w[C11_idx] = 1
            elif q == 5:  # p6 -> +C00 -C10
                w[C00_idx] = 1
                w[C10_idx] = -1
            elif q == 6:  # p7 -> +C01 +C11
                w[C01_idx] = 1
                w[C11_idx] = 1
            
            W[:, col] = w
    
    # Check deadline before verification
    if _time.time() > _deadline:
        # Return current best (Strassen) without verification
        U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
        V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
        W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Verify correctness
    def verify_decomp(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(Uf.shape[1]):
                        s += Uf[i*m + j, r] * Vf[j*p + k, r] * Wf[k*n + i, r]
                    if abs(s - 1.0) > 1e-12:
                        return False, (i, j, k, s)
        return True, None
    
    is_valid, info = verify_decomp(U, V, W)
    if not is_valid:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    # Convert to lists with integer values where possible
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
