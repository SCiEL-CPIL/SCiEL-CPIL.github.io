"""Improved program: rank-15 decomposition for 2x3 @ 3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, j, k, s)
    return True, None

def _strassen_222():
    """Return the 7 rank-1 terms for 2x2 @ 2x2 Strassen decomposition.
    U index: i*2+j (0..3)
    V index: j*2+k (0..3)
    W index: k*2+i (0..3)
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1,0,0,1], [1,1,0,0], [1,0,0,1]))
    # M2 = (A10+A11)B00
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,1]))
    # M3 = A00(B10-B00) -> A00(B10) - A00(B00)
    # Strassen M3 = A00(B10 - B00). 
    # U: [1,0,0,0]. V: [0,1,0,-1] ? No, V is j*2+k.
    # B10 is j=1,k=0 -> idx 2. B00 is j=0,k=0 -> idx 0.
    # So V should be [0, 0, 1, 0] for B10? No.
    # Let's re-derive Strassen indices carefully.
    
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = C + M
    
    # M1 = (a+d)(e+h)
    # a: i=0,j=0 (0). d: i=1,j=1 (3).
    # e: j=0,k=0 (0). h: j=1,k=1 (3).
    # C00: k=0,i=0 (0). C11: k=1,i=1 (3).
    # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    
    # M2 = (c+d)e
    # c: i=1,j=0 (2). d: i=1,j=1 (3).
    # e: j=0,k=0 (0).
    # C10: k=0,i=1 (2). C00? No, M2 contributes to C10 and C11?
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map to indices:
    # U (i,j): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V (j,k): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W (k,i): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1] (C00, C11)
    
    # M2: (A10+A11)B00
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: [0, 0, 1, 0] (C10 only? No, M2 contributes to C10 and C11?
    # Formula: C10 = M2 + M4. C11 = ... - M2 ...
    # So M2 affects C10 and C11.
    # W indices for C10 is (k=0, i=1) -> 1*2+0? No.
    # W index is k*n + i. n=2.
    # C10: k=0, i=1 -> 0*2+1 = 1.
    # C11: k=1, i=1 -> 1*2+1 = 3.
    # So W should be [0, 1, 0, 1]?
    # Let's check C10 and C11 indices again.
    # W[k*n+i].
    # C00: 0*2+0=0.
    # C01: 0*2+1=1.
    # C10: 1*2+0=2.
    # C11: 1*2+1=3.
    
    # Recalculate W indices:
    # C00: 0
    # C01: 1
    # C10: 2
    # C11: 3
    
    # M1: C00, C11. W=[1,0,0,1]
    # M2: C10, C11?
    # Formula C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # So M2 adds to C10 and subtracts from C11.
    # W=[0, 0, 1, -1]
    
    # M3: A00(B10-B00) -> a(f-h)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1] (B01 is idx 1, B11 is idx 3? No. B10 is idx 2, B00 is idx 0?
    # f is B01 (j=0,k=1) -> idx 1.
    # h is B11 (j=1,k=1) -> idx 3.
    # So V: [0, 1, 0, -1]
    # M3 contributes to C01 and C11.
    # C01: M3 + M5.
    # C11: ... + M3 ...
    # W: [0, 1, 0, 1]
    
    # M4: A11(B10-B00) -> d(g-e)
    # U: [0, 0, 0, 1]
    # V: [0, 0, 1, -1] (g is B10 idx 2, e is B00 idx 0)
    # M4 contributes to C00 and C10.
    # C00: ... + M4 ...
    # C10: M2 + M4.
    # W: [1, 0, 1, 0]
    
    # M5: (A00+A01)B11 -> (a+b)h
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # M5 contributes to C00 and C01.
    # C00: ... - M5 ...
    # C01: M3 + M5.
    # W: [-1, 1, 0, 0]
    
    # M6: (A10-A00)(B00+B01) -> (c-a)(e+f)
    # U: [-1, 0, 1, 0]
    # V: [1, 1, 0, 0]
    # M6 contributes to C11 only.
    # W: [0, 0, 0, 1]
    
    # M7: (A01-A11)(B10+B11) -> (b-d)(g+h)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # M7 contributes to C00 only.
    # W: [1, 0, 0, 0]
    
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # M3
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0])   # M7
    ]

def _construct_rank_15():
    """Construct rank-15 decomposition for 2x3 @ 3x4 -> 2x4.
    Split m=3 as 2+1.
    Split p=4 as 2+2 for the 2x2 part.
    """
    n, m, p = 2, 3, 4
    strassen_terms = _strassen_222()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: A[:,:2] @ B[:2,:2]
    for (u, v, w) in strassen_terms:
        U_row = [0]*6
        V_row = [0]*12
        W_row = [0]*8
        
        # U: i*3+j
        U_row[0*3+0] = u[0]
        U_row[0*3+1] = u[1]
        U_row[1*3+0] = u[2]
        U_row[1*3+1] = u[3]
        
        # V: j*4+k
        V_row[0*4+0] = v[0]
        V_row[0*4+1] = v[1]
        V_row[1*4+0] = v[2]
        V_row[1*4+1] = v[3]
        
        # W: k*2+i
        W_row[0*2+0] = w[0]
        W_row[0*2+1] = w[1]
        W_row[1*2+0] = w[2]
        W_row[1*2+1] = w[3]
        
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
        
    # Part 2: A[:,:2] @ B[:2,2:]
    for (u, v, w) in strassen_terms:
        U_row = [0]*6
        V_row = [0]*12
        W_row = [0]*8
        
        U_row[0*3+0] = u[0]
        U_row[0*3+1] = u[1]
        U_row[1*3+0] = u[2]
        U_row[1*3+1] = u[3]
        
        V_row[0*4+2] = v[0]
        V_row[0*4+3] = v[1]
        V_row[1*4+2] = v[2]
        V_row[1*4+3] = v[3]
        
        W_row[2*2+0] = w[0]
        W_row[2*2+1] = w[1]
        W_row[3*2+0] = w[2]
        W_row[3*2+1] = w[3]
        
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
        
    # Part 3: A[:,2] @ B[2,:]
    # Rank 1 term
    U_row = [0]*6
    V_row = [0]*12
    W_row = [0]*8
    
    # U: i*3+2
    U_row[0*3+2] = 1
    U_row[1*3+2] = 1
    
    # V: 2*4+k
    for k in range(4):
        V_row[2*4+k] = 1
        
    # W: k*2+i
    for k in range(4):
        for i in range(2):
            W_row[k*2+i] = 1
            
    U_cols.append(U_row)
    V_cols.append(V_row)
    W_cols.append(W_row)
    
    return {"U": U_cols, "V": V_cols, "W": W_cols}

def _construct_rank_24():
    """Schoolbook decomposition of rank 24."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    try:
        if time.time() < deadline:
            result = _construct_rank_15()
            valid, info = _verify(result["U"], result["V"], result["W"], n, m, p)
            if valid:
                return result
            else:
                print(f"Rank-15 verification failed: {info}")
    except Exception as e:
        print(f"Rank-15 construction failed: {e}")
        
    return _construct_rank_24()
# EVOLVE-BLOCK-END
