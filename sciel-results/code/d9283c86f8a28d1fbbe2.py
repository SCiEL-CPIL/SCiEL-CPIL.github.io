import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 120.0

def _build_tensor(n, m, p):
    T = np.zeros((n * p, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = k * n + i
                u_idx = i * m + j
                v_idx = j * p + k
                T[out_idx, u_idx, v_idx] = 1.0
    return T

def _construct_strassen_decomposition(n, m, p):
    """
    Constructs a rank-minimal (or near-minimal) decomposition by splitting 
    the multiplication into 2x2 blocks where Strassen applies, and schoolbook elsewhere.
    
    For 2x4x7:
    Split M1 (2x4) into [M1_0, M1_1] (each 2x2)
    Split M2 (4x7) into [M2_0; M2_1] (each 2x7)
    Result C (2x7) = M1_0 M2_0 + M1_1 M2_1
    
    We can decompose the 2x2x7 multiplication.
    R(2,2,7) can be bounded by R(2,2,2)*R(1,1,3) + ... ? No, simply:
    Split M2_0 (2x7) into [M2_0_0, M2_0_1, M2_0_2] (each 2x2, 2x2, 2x1)
    Split M2_1 (2x7) into [M2_1_0, M2_1_1, M2_1_2] (each 2x2, 2x2, 2x1)
    
    Term 1: M1_0 (2x2) * M2_0_0 (2x2) -> Rank 7 (Strassen)
    Term 2: M1_0 (2x2) * M2_0_1 (2x2) -> Rank 7 (Strassen)
    Term 3: M1_0 (2x2) * M2_0_2 (2x1) -> Rank 4 (Schoolbook 2x2x1)
    Term 4: M1_1 (2x2) * M2_1_0 (2x2) -> Rank 7 (Strassen)
    Term 5: M1_1 (2x2) * M2_1_1 (2x2) -> Rank 7 (Strassen)
    Term 6: M1_1 (2x2) * M2_1_2 (2x1) -> Rank 4 (Schoolbook 2x2x1)
    
    Total Rank = 7 + 7 + 4 + 7 + 7 + 4 = 36.
    
    This is significantly better than 56.
    """
    # We will build the factor matrices U, V, W directly.
    # U shape: (n*m, R), V shape: (m*p, R), W shape: (p*n, R)
    
    # Helper to add a rank-1 term
    # But Strassen is rank-7, so we need to add 7 rank-1 terms per 2x2x2 block.
    
    # Strassen 2x2x2 decomposition:
    # M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]]
    # M1 M2 = [[ae+bg, af+dh], [ce+dg, cf+bh]]
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # We need to map these to the global indices.
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    def add_strassen_block(m1_i, m1_j, m2_j, m2_k, out_i, out_k):
        """
        Adds Strassen decomposition for M1[i, j] * M2[j, k]
        m1_i: row index of M1 (0..n-1)
        m1_j: col index of M1 (0..m-1)
        m2_j: row index of M2 (0..m-1)
        m2_k: col index of M2 (0..p-1)
        out_i: row index of Output (0..n-1)
        out_k: col index of Output (0..p-1)
        
        Assumes m1, m2, out are 2x2 blocks.
        M1 block is at rows [m1_i, m1_i+1], cols [m1_j, m1_j+1]
        M2 block is at rows [m2_j, m2_j+1], cols [m2_k, m2_k+1]
        Out block is at rows [out_i, out_i+1], cols [out_k, out_k+1]
        """
        
        # Indices in flattened arrays
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # Strassen terms for 2x2x2
        # p1 = (a+d)(e+h)
        # a = M1[0,0], d = M1[1,1]
        # e = M2[0,0], h = M2[1,1]
        
        # We define 7 terms.
        # Each term is a rank-1 update.
        # Term t: U[:, t] * V[:, t] * W[:, t] contributes to the tensor.
        # Actually, the tensor decomposition is T = sum_t U_t V_t W_t.
        # Here U, V, W are vectors.
        
        # Let's construct the 7 vectors for this block.
        
        # p1: (a+d)(e+h)
        # U: a at (m1_i, m1_j), d at (m1_i+1, m1_j+1)
        # V: e at (m2_j, m2_k), h at (m2_j+1, m2_k+1)
        # W: 1 at (out_k, out_i), 1 at (out_k+1, out_i) -- wait, W is output factor.
        # The output of p1 is added to C11, C21, C22.
        # C11 gets +p1. C21 gets 0. C22 gets +p1.
        # Wait, Strassen formula:
        # C11 = p1 + p4 - p5 + p7
        # C12 = p3 + p5
        # C21 = p2 + p4
        # C22 = p1 - p2 + p3 + p6
        
        # So for p1:
        # W contributions:
        # C11 (out_i, out_k) -> +1
        # C12 (out_i, out_k+1) -> 0
        # C21 (out_i+1, out_k) -> 0
        # C22 (out_i+1, out_k+1) -> +1
        
        # U contributions for p1:
        # a (M1[0,0]) -> index u_idx(m1_i, m1_j) -> 1
        # d (M1[1,1]) -> index u_idx(m1_i+1, m1_j+1) -> 1
        
        # V contributions for p1:
        # e (M2[0,0]) -> index v_idx(m2_j, m2_k) -> 1
        # h (M2[1,1]) -> index v_idx(m2_j+1, m2_k+1) -> 1
        
        # Let's build these 7 terms explicitly.
        
        # Term 1: p1
        u1 = np.zeros(n*m); u1[u_idx(m1_i, m1_j)] = 1; u1[u_idx(m1_i+1, m1_j+1)] = 1
        v1 = np.zeros(m*p); v1[v_idx(m2_j, m2_k)] = 1; v1[v_idx(m2_j+1, m2_k+1)] = 1
        w1 = np.zeros(p*n); w1[w_idx(out_k, out_i)] = 1; w1[w_idx(out_k+1, out_i+1)] = 1
        add_term(u1, v1, w1)
        
        # Term 2: p2 = (c+d)e
        # c = M1[1,0], d = M1[1,1]
        # e = M2[0,0]
        # C21 gets +p2.
        u2 = np.zeros(n*m); u2[u_idx(m1_i+1, m1_j)] = 1; u2[u_idx(m1_i+1, m1_j+1)] = 1
        v2 = np.zeros(m*p); v2[v_idx(m2_j, m2_k)] = 1
        w2 = np.zeros(p*n); w2[w_idx(out_k, out_i+1)] = 1
        add_term(u2, v2, w2)
        
        # Term 3: p3 = a(f-h)
        # a = M1[0,0]
        # f = M2[0,1], h = M2[1,1]
        # C12 gets +p3, C22 gets +p3.
        u3 = np.zeros(n*m); u3[u_idx(m1_i, m1_j)] = 1
        v3 = np.zeros(m*p); v3[v_idx(m2_j, m2_k+1)] = 1; v3[v_idx(m2_j+1, m2_k+1)] = -1
        w3 = np.zeros(p*n); w3[w_idx(out_k+1, out_i)] = 1; w3[w_idx(out_k+1, out_i+1)] = 1
        add_term(u3, v3, w3)
        
        # Term 4: p4 = d(g-e)
        # d = M1[1,1]
        # g = M2[1,0], e = M2[0,0]
        # C11 gets +p4, C21 gets +p4.
        u4 = np.zeros(n*m); u4[u_idx(m1_i+1, m1_j+1)] = 1
        v4 = np.zeros(m*p); v4[v_idx(m2_j+1, m2_k)] = 1; v4[v_idx(m2_j, m2_k)] = -1
        w4 = np.zeros(p*n); w4[w_idx(out_k, out_i)] = 1; w4[w_idx(out_k, out_i+1)] = 1
        add_term(u4, v4, w4)
        
        # Term 5: p5 = (a+b)h
        # a = M1[0,0], b = M1[0,1]
        # h = M2[1,1]
        # C11 gets -p5, C12 gets +p5.
        u5 = np.zeros(n*m); u5[u_idx(m1_i, m1_j)] = 1; u5[u_idx(m1_i, m1_j+1)] = 1
        v5 = np.zeros(m*p); v5[v_idx(m2_j+1, m2_k+1)] = 1
        w5 = np.zeros(p*n); w5[w_idx(out_k, out_i)] = -1; w5[w_idx(out_k+1, out_i)] = 1
        add_term(u5, v5, w5)
        
        # Term 6: p6 = (c-a)(e+f)
        # c = M1[1,0], a = M1[0,0]
        # e = M2[0,0], f = M2[0,1]
        # C22 gets +p6.
        u6 = np.zeros(n*m); u6[u_idx(m1_i+1, m1_j)] = 1; u6[u_idx(m1_i, m1_j)] = -1
        v6 = np.zeros(m*p); v6[v_idx(m2_j, m2_k)] = 1; v6[v_idx(m2_j, m2_k+1)] = 1
        w6 = np.zeros(p*n); w6[w_idx(out_k+1, out_i+1)] = 1
        add_term(u6, v6, w6)
        
        # Term 7: p7 = (b-d)(g+h)
        # b = M1[0,1], d = M1[1,1]
        # g = M2[1,0], h = M2[1,1]
        # C11 gets +p7.
        u7 = np.zeros(n*m); u7[u_idx(m1_i, m1_j+1)] = 1; u7[u_idx(m1_i+1, m1_j+1)] = -1
        v7 = np.zeros(m*p); v7[v_idx(m2_j+1, m2_k)] = 1; v7[v_idx(m2_j+1, m2_k+1)] = 1
        w7 = np.zeros(p*n); w7[w_idx(out_k, out_i)] = 1
        add_term(u7, v7, w7)

    def add_schoolbook_2x2x1_block(m1_i, m1_j, m2_j, m2_k, out_i, out_k):
        """
        Adds schoolbook decomposition for M1 (2x2) * M2 (2x1) -> Out (2x1)
        m1_i, m1_j: top-left of M1 block
        m2_j, m2_k: top-left of M2 block (col k)
        out_i, out_k: top-left of Out block
        """
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # C1 = a*e + b*f
        # C2 = c*e + d*f
        # a=M1[0,0], b=M1[0,1], c=M1[1,0], d=M1[1,1]
        # e=M2[0,0], f=M2[1,0]
        
        # Term 1: a*e -> C1
        u1 = np.zeros(n*m); u1[u_idx(m1_i, m1_j)] = 1
        v1 = np.zeros(m*p); v1[v_idx(m2_j, m2_k)] = 1
        w1 = np.zeros(p*n); w1[w_idx(out_k, out_i)] = 1
        add_term(u1, v1, w1)
        
        # Term 2: b*f -> C1
        u2 = np.zeros(n*m); u2[u_idx(m1_i, m1_j+1)] = 1
        v2 = np.zeros(m*p); v2[v_idx(m2_j+1, m2_k)] = 1
        w2 = np.zeros(p*n); w2[w_idx(out_k, out_i)] = 1
        add_term(u2, v2, w2)
        
        # Term 3: c*e -> C2
        u3 = np.zeros(n*m); u3[u_idx(m1_i+1, m1_j)] = 1
        v3 = np.zeros(m*p); v3[v_idx(m2_j, m2_k)] = 1
        w3 = np.zeros(p*n); w3[w_idx(out_k, out_i+1)] = 1
        add_term(u3, v3, w3)
        
        # Term 4: d*f -> C2
        u4 = np.zeros(n*m); u4[u_idx(m1_i+1, m1_j+1)] = 1
        v4 = np.zeros(m*p); v4[v_idx(m2_j+1, m2_k)] = 1
        w4 = np.zeros(p*n); w4[w_idx(out_k, out_i+1)] = 1
        add_term(u4, v4, w4)

    # Dimensions: n=2, m=4, p=7
    
    # Block 1: M1[0,0:2] * M2[0:2, 0:2] -> Out[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: M1[0,0:2] * M2[0:2, 2:4] -> Out[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # Block 3: M1[0,0:2] * M2[0:2, 4:5] -> Out[0:2, 4:5] (2x2x1)
    add_schoolbook_2x2x1_block(0, 0, 0, 4, 0, 4)
    
    # Block 4: M1[0,0:2] * M2[0:2, 5:6] -> Out[0:2, 5:6] (2x2x1)
    add_schoolbook_2x2x1_block(0, 0, 0, 5, 0, 5)
    
    # Block 5: M1[0,0:2] * M2[0:2, 6:7] -> Out[0:2, 6:7] (2x2x1)
    add_schoolbook_2x2x1_block(0, 0, 0, 6, 0, 6)
    
    # Block 6: M1[1,0:2] * M2[0:2, 0:2] -> Out[1:2, 0:2]
    add_strassen_block(1, 0, 0, 0, 1, 0)
    
    # Block 7: M1[1,0:2] * M2[0:2, 2:4] -> Out[1:2, 2:4]
    add_strassen_block(1, 0, 0, 2, 1, 2)
    
    # Block 8: M1[1,0:2] * M2[0:2, 4:5] -> Out[1:2, 4:5] (2x2x1)
    add_schoolbook_2x2x1_block(1, 0, 0, 4, 1, 4)
    
    # Block 9: M1[1,0:2] * M2[0:2, 5:6] -> Out[1:2, 5:6] (2x2x1)
    add_schoolbook_2x2x1_block(1, 0, 0, 5, 1, 5)
    
    # Block 10: M1[1,0:2] * M2[0:2, 6:7] -> Out[1:2, 6:7] (2x2x1)
    add_schoolbook_2x2x1_block(1, 0, 0, 6, 1, 6)
    
    # Convert lists of vectors to matrices
    U = np.array(U).T
    V = np.array(V).T
    W = np.array(W).T
    
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Try the constructive approach
    U, V, W = _construct_strassen_decomposition(n, m, p)
    
    # Verify
    T = _build_tensor(n, m, p)
    recon = np.einsum('it,jt,kt->ijk', U, V, W)
    
    if np.allclose(recon, T, atol=1e-6):
        # Success
        # Convert to list of lists
        def _to_int_or_half(M):
            out = []
            for row in M:
                newrow = []
                for x in row:
                    v = round(2 * x) / 2.0
                    if abs(v - round(v)) < 1e-6:
                        v = float(round(v))
                    newrow.append(v)
                out.append(newrow)
            return out
            
        return {"U": _to_int_or_half(U), "V": _to_int_or_half(V), "W": _to_int_or_half(W)}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        def _to_int_or_half(M):
            out = []
            for row in M:
                newrow = []
                for x in row:
                    v = round(2 * x) / 2.0
                    if abs(v - round(v)) < 1e-6:
                        v = float(round(v))
                    newrow.append(v)
                out.append(newrow)
            return out
        return {"U": _to_int_or_half(U), "V": _to_int_or_half(V), "W": _to_int_or_half(W)}

