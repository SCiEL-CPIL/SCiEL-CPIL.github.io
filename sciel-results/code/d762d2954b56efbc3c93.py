import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # We decompose M_{2,3,10} into 5 blocks of M_{2,2,2} (Strassen) and 10 blocks of M_{2,1,1}.
    # A = [A11 A12 A13; A21 A22 A23]
    # B = [B11 B12 ... B110; B21 B22 ... B210; B31 B32 ... B310]
    # C = A B
    # C = [A11 A12] [B11 ... B110; B21 ... B210] + [A13; A23] [B31 ... B310]
    
    # Part 1: M_{2,2,10} using 5 Strassen blocks
    # Part 2: M_{2,1,10} using 10 simple blocks
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term (u_idx, v_idx, w_idx, coeff)
    # We will store terms as tuples (u, v, w, coeff) and then build matrices
    
    terms = []
    
    # Strassen coefficients for M_{2,2,2}
    # m1 = (a+d)(e+h) -> u: a,d; v: e,h; w: C11(+), C22(+)
    # m2 = (c+d)e    -> u: c,d; v: e;    w: C21(+), C22(-)
    # m3 = a(f-h)    -> u: a;    v: f,h;  w: C12(+), C22(+)
    # m4 = d(g-e)    -> u: d;    v: g,e;  w: C11(+), C21(+)
    # m5 = (a+b)h    -> u: a,b;  v: h;    w: C11(-), C12(+)
    # m6 = (c-a)(e+f)-> u: c,a;  v: e,f;  w: C22(+)
    # m7 = (b-d)(g+h)-> u: b,d;  v: g,h;  w: C11(+)
    
    # Indices for A (2x3):
    # A11=0, A12=1, A13=2
    # A21=3, A22=4, A23=5
    
    # Indices for B (3x10):
    # B1k = 0*k + 0 = 0..9? No, B is 3x10.
    # V index: j*p + k. j in 0..2, k in 0..9.
    # B11 (j=0,k=0) -> 0
    # B12 (j=0,k=1) -> 1
    # ...
    # B21 (j=1,k=0) -> 10
    # B31 (j=2,k=0) -> 20
    
    # Indices for C (2x10):
    # W index: k*n + i. k in 0..9, i in 0..1.
    # C11 (k=0, i=0) -> 0
    # C12 (k=1, i=0) -> 2
    # C21 (k=0, i=1) -> 1
    # C22 (k=1, i=1) -> 3
    
    # We will iterate through 5 blocks of 2 columns each for the Strassen part.
    # Block t (0..4) handles columns 2t and 2t+1 of B (indices 2t, 2t+1)
    # And rows 0,1 of B (j=0,1).
    # And rows 0,1 of A (i=0,1).
    
    for t in range(5):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        # B indices for this block
        # e = B1k1 = 0*p + k1 = k1
        # f = B1k2 = 0*p + k2 = k2
        # g = B2k1 = 1*p + k1 = 10 + k1
        # h = B2k2 = 1*p + k2 = 10 + k2
        
        # A indices
        a = 0 # A11
        b = 1 # A12
        c = 3 # A21
        d = 4 # A22
        
        # C indices for this block
        # C1k1 = k1*n + 0 = 2*k1
        # C1k2 = k2*n + 0 = 2*k2
        # C2k1 = k1*n + 1 = 2*k1 + 1
        # C2k2 = k2*n + 1 = 2*k2 + 1
        
        c11 = 2 * k1
        c12 = 2 * k2
        c21 = 2 * k1 + 1
        c22 = 2 * k2 + 1
        
        e = k1
        f = k2
        g = 10 + k1
        h = 10 + k2
        
        # Add Strassen terms
        # m1 = (a+d)(e+h)
        terms.append((a, e, c11, 1))
        terms.append((a, h, c11, 1))
        terms.append((d, e, c11, 1))
        terms.append((d, h, c11, 1))
        
        terms.append((a, e, c22, 1))
        terms.append((a, h, c22, 1))
        terms.append((d, e, c22, 1))
        terms.append((d, h, c22, 1))
        
        # m2 = (c+d)e
        terms.append((c, e, c21, 1))
        terms.append((d, e, c21, 1))
        
        terms.append((c, e, c22, -1))
        terms.append((d, e, c22, -1))
        
        # m3 = a(f-h)
        terms.append((a, f, c12, 1))
        terms.append((a, h, c12, -1))
        
        terms.append((a, f, c22, 1))
        terms.append((a, h, c22, -1))
        
        # m4 = d(g-e)
        terms.append((d, g, c11, 1))
        terms.append((d, e, c11, -1))
        
        terms.append((d, g, c21, 1))
        terms.append((d, e, c21, -1))
        
        # m5 = (a+b)h
        terms.append((a, h, c11, -1))
        terms.append((b, h, c11, -1))
        
        terms.append((a, h, c12, 1))
        terms.append((b, h, c12, 1))
        
        # m6 = (c-a)(e+f)
        terms.append((c, e, c22, 1))
        terms.append((c, f, c22, 1))
        terms.append((a, e, c22, -1))
        terms.append((a, f, c22, -1))
        
        # m7 = (b-d)(g+h)
        terms.append((b, g, c11, 1))
        terms.append((b, h, c11, 1))
        terms.append((d, g, c11, -1))
        terms.append((d, h, c11, -1))
        
    # Part 2: M_{2,1,10}
    # A13 = 2, A23 = 5
    # B3k = 2*p + k = 20 + k
    
    for k in range(10):
        b3k = 20 + k
        c1k = 2 * k
        c2k = 2 * k + 1
        
        # C1k += A13 * B3k
        terms.append((2, b3k, c1k, 1))
        # C2k += A23 * B3k
        terms.append((5, b3k, c2k, 1))
        
    # Now build U, V, W
    # We need to consolidate terms with same (u,v) or just keep them separate?
    # The tensor decomposition is a sum of rank-1 tensors.
    # U, V, W are matrices of shape (dim, rank).
    # Each column r corresponds to a term.
    # If we have multiple terms, we can just append them as new columns.
    # The rank is the number of columns.
    
    # However, we can merge terms with the same U index and V index?
    # No, the rank is the number of terms in the decomposition.
    # We can simplify by summing coefficients for identical (u,v) pairs?
    # If two terms have same u and v, they contribute to the same (u,v) entry in the "product" space?
    # No, the tensor is T = sum_r U[:,r] V[:,r] W[:,r]^T.
    # If we have two terms with same u and v, they are linearly dependent in the U and V factors?
    # Actually, if U[:,r1] == U[:,r2] and V[:,r1] == V[:,r2], we can merge them into one column with W = W1 + W2.
    # This reduces the rank.
    
    # Let's collect terms and merge.
    from collections import defaultdict
    
    term_map = defaultdict(lambda: [0.0] * (n*p)) # W index is p*n
    
    for u, v, w, coeff in terms:
        key = (u, v)
        term_map[key][w] += coeff
        
    # Now create U, V, W
    keys = list(term_map.keys())
    rank = len(keys)
    
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    for idx, (u, v) in enumerate(keys):
        U[u][idx] = 1.0
        V[v][idx] = 1.0
        w_vec = term_map[(u, v)]
        for w_idx in range(p*n):
            if w_vec[w_idx] != 0.0:
                W[w_idx][idx] = w_vec[w_idx]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
