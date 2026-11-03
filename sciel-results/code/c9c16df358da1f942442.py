import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = 3, 3, 4
    # Max rank estimate: 7+1+7+1+4+1+4+1 = 26
    rank = 26
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= rank:
            return
        U[:, col] += u_vec
        V[:, col] += v_vec
        W[:, col] += w_vec
        col += 1

    # Helper to get index for A[i, k]
    def idx_A(i, k):
        return i * m + k
        
    # Helper to get index for B[k, j]
    def idx_B(k, j):
        return k * p + j
        
    # Helper to get index for C[i, j] based on W[k*n+i] convention?
    # Prompt: W[k*n+i,r]
    # Failed code analysis suggested W[j*n + i] for C[i,j].
    # Let's verify the "official trace convention".
    # Usually trace is sum C[i,i].
    # If W index is k*n+i, and k is col, i is row?
    # If C is n x p.
    # If index = j * n + i, then for n=3, p=4:
    # C[0,0] -> 0*3+0=0
    # C[0,1] -> 1*3+0=3
    # C[1,0] -> 0*3+1=1
    # This matches the failed code's comments:
    # C[0,0] -> W[0]
    # C[0,1] -> W[3]
    # C[1,0] -> W[1]
    # So W_idx = j * n + i.
    
    def idx_C(i, j):
        return j * n + i

    # --- Block 1: C[0:2, 0:2] from A[0:2, 0:2] B[0:2, 0:2] (Strassen) ---
    # A indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B indices: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # C indices: (0,0)=0, (0,1)=3, (1,0)=1, (1,1)=4
    
    # M1 = (A00+A11)(B00+B11)
    u = np.zeros(n*m); u[idx_A(0,0)]=1; u[idx_A(1,1)]=1
    v = np.zeros(m*p); v[idx_B(0,0)]=1; v[idx_B(1,1)]=1
    w = np.zeros(p*n); w[idx_C(0,0)]=1; w[idx_C(1,1)]=1
    add_term(u, v, w)
    
    # M2 = (A10+A11)B00
    u = np.zeros(n*m); u[idx_A(1,0)]=1; u[idx_A(1,1)]=1
    v = np.zeros(m*p); v[idx_B(0,0)]=1
    w = np.zeros(p*n); w[idx_C(1,0)]=1; w[idx_C(1,1)]=-1
    add_term(u, v, w)
    
    # M3 = A00(B01-B11)
    u = np.zeros(n*m); u[idx_A(0,0)]=1
    v = np.zeros(m*p); v[idx_B(0,1)]=1; v[idx_B(1,1)]=-1
    w = np.zeros(p*n); w[idx_C(0,1)]=1; w[idx_C(1,1)]=1
    add_term(u, v, w)
    
    # M4 = A11(B10-B00)
    u = np.zeros(n*m); u[idx_A(1,1)]=1
    v = np.zeros(m*p); v[idx_B(1,0)]=1; v[idx_B(0,0)]=-1
    w = np.zeros(p*n); w[idx_C(0,0)]=1
    add_term(u, v, w)
    
    # M5 = (A00+A01)B11
    u = np.zeros(n*m); u[idx_A(0,0)]=1; u[idx_A(0,1)]=1
    v = np.zeros(m*p); v[idx_B(1,1)]=1
    w = np.zeros(p*n); w[idx_C(0,0)]=-1; w[idx_C(0,1)]=1
    add_term(u, v, w)
    
    # M6 = (A10-A00)(B00+B01)
    u = np.zeros(n*m); u[idx_A(1,0)]=1; u[idx_A(0,0)]=-1
    v = np.zeros(m*p); v[idx_B(0,0)]=1; v[idx_B(0,1)]=1
    w = np.zeros(p*n); w[idx_C(1,1)]=1
    add_term(u, v, w)
    
    # M7 = (A01-A11)(B10+B11)
    u = np.zeros(n*m); u[idx_A(0,1)]=1; u[idx_A(1,1)]=-1
    v = np.zeros(m*p); v[idx_B(1,0)]=1; v[idx_B(1,1)]=1
    w = np.zeros(p*n); w[idx_C(0,0)]=-1; w[idx_C(1,0)]=1
    add_term(u, v, w)

    # --- Block 2: C[0:2, 0:2] from A[0:2, 2] B[2, 0:2] (Outer Product) ---
    # A indices: (0,2)=2, (1,2)=5
    # B indices: (2,0)=8, (2,1)=9
    # C indices: (0,0)=0, (0,1)=3, (1,0)=1, (1,1)=4
    # Rank 1 term
    u = np.zeros(n*m); u[idx_A(0,2)]=1; u[idx_A(1,2)]=1
    v = np.zeros(m*p); v[idx_B(2,0)]=1; v[idx_B(2,1)]=1
    w = np.zeros(p*n); w[idx_C(0,0)]=1; w[idx_C(0,1)]=1; w[idx_C(1,0)]=1; w[idx_C(1,1)]=1
    add_term(u, v, w)

    # --- Block 3: C[0:2, 2:4] from A[0:2, 0:2] B[0:2, 2:4] (Strassen) ---
    # A indices: same as Block 1
    # B indices: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # C indices: (0,2)=6, (0,3)=9, (1,2)=7, (1,3)=10
    
    # M1 = (A00+A11)(B02+B13)
    u = np.zeros(n*m); u[idx_A(0,0)]=1; u[idx_A(1,1)]=1
    v = np.zeros(m*p); v[idx_B(0,2)]=1; v[idx_B(1,3)]=1
    w = np.zeros(p*n); w[idx_C(0,2)]=1; w[idx_C(1,3)]=1
    add_term(u, v, w)
    
    # M2 = (A10+A11)B02
    u = np.zeros(n*m); u[idx_A(1,0)]=1; u[idx_A(1,1)]=1
    v = np.zeros(m*p); v[idx_B(0,2)]=1
    w = np.zeros(p*n); w[idx_C(1,2)]=1; w[idx_C(1,3)]=-1
    add_term(u, v, w)
    
    # M3 = A00(B03-B13)
    u = np.zeros(n*m); u[idx_A(0,0)]=1
    v = np.zeros(m*p); v[idx_B(0,3)]=1; v[idx_B(1,3)]=-1
    w = np.zeros(p*n); w[idx_C(0,3)]=1; w[idx_C(1,3)]=1
    add_term(u, v, w)
    
    # M4 = A11(B12-B02)
    u = np.zeros(n*m); u[idx_A(1,1)]=1
    v = np.zeros(m*p); v[idx_B(1,2)]=1; v[idx_B(0,2)]=-1
    w = np.zeros(p*n); w[idx_C(0,2)]=1
    add_term(u, v, w)
    
    # M5 = (A00+A01)B13
    u = np.zeros(n*m); u[idx_A(0,0)]=1; u[idx_A(0,1)]=1
    v = np.zeros(m*p); v[idx_B(1,3)]=1
    w = np.zeros(p*n); w[idx_C(0,2)]=-1; w[idx_C(0,3)]=1
    add_term(u, v, w)
    
    # M6 = (A10-A00)(B02+B03)
    u = np.zeros(n*m); u[idx_A(1,0)]=1; u[idx_A(0,0)]=-1
    v = np.zeros(m*p); v[idx_B(0,2)]=1; v[idx_B(0,3)]=1
    w = np.zeros(p*n); w[idx_C(1,3)]=1
    add_term(u, v, w)
    
    # M7 = (A01-A11)(B12+B13)
    u = np.zeros(n*m); u[idx_A(0,1)]=1; u[idx_A(1,1)]=-1
    v = np.zeros(m*p); v[idx_B(1,2)]=1; v[idx_B(1,3)]=1
    w = np.zeros(p*n); w[idx_C(0,2)]=-1; w[idx_C(1,2)]=1
    add_term(u, v, w)

    # --- Block 4: C[0:2, 2:4] from A[0:2, 2] B[2, 2:4] (Outer Product) ---
    # A indices: (0,2)=2, (1,2)=5
    # B indices: (2,2)=10, (2,3)=11
    # C indices: (0,2)=6, (0,3)=9, (1,2)=7, (1,3)=10
    # Rank 1 term
    u = np.zeros(n*m); u[idx_A(0,2)]=1; u[idx_A(1,2)]=1
    v = np.zeros(m*p); v[idx_B(2,2)]=1; v[idx_B(2,3)]=1
    w = np.zeros(p*n); w[idx_C(0,2)]=1; w[idx_C(0,3)]=1; w[idx_C(1,2)]=1; w[idx_C(1,3)]=1
    add_term(u, v, w)

    # --- Block 5: C[2, 0:2] from A[2, 0:2] B[0:2, 0:2] (Rank 4) ---
    # A indices: (2,0)=6, (2,1)=7
    # B indices: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # C indices: (2,0)=2, (2,1)=5
    # Terms:
    # C[2,0] += A[2,0]B[0,0] + A[2,1]B[1,0]
    # C[2,1] += A[2,0]B[0,1] + A[2,1]B[1,1]
    
    u = np.zeros(n*m); u[idx_A(2,0)]=1
    v = np.zeros(m*p); v[idx_B(0,0)]=1
    w = np.zeros(p*n); w[idx_C(2,0)]=1
    add_term(u, v, w)
    
    u = np.zeros(n*m); u[idx_A(2,1)]=1
    v = np.zeros(m*p); v[idx_B(1,0)]=1
    w = np.zeros(p*n); w[idx_C(2,0)]=1
    add_term(u, v, w)
    
    u = np.zeros(n*m); u[idx_A(2,0)]=1
    v = np.zeros(m*p); v[idx_B(0,1)]=1
    w = np.zeros(p*n); w[idx_C(2,1)]=1
    add_term(u, v, w)
    
    u = np.zeros(n*m); u[idx_A(2,1)]=1
    v = np.zeros(m*p); v[idx_B(1,1)]=1
    w = np.zeros(p*n); w[idx_C(2,1)]=1
    add_term(u, v, w)

    # --- Block 6: C[2, 0:2] from A[2, 2] B[2, 0:2] (Rank 1) ---
    # A indices: (2,2)=8
    # B indices: (2,0)=8, (2,1)=9
    # C indices: (2,0)=2, (2,1)=5
    u = np.zeros(n*m); u[idx_A(2,2)]=1
    v = np.zeros(m*p); v[idx_B(2,0)]=1; v[idx_B(2,1)]=1
    w = np.zeros(p*n); w[idx_C(2,0)]=1; w[idx_C(2,1)]=1
    add_term(u, v, w)

    # --- Block 7: C[2, 2:4] from A[2, 0:2] B[0:2, 2:4] (Rank 4) ---
    # A indices: (2,0)=6, (2,1)=7
    # B indices: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # C indices: (2,2)=8, (2,3)=11
    
    u = np.zeros(n*m); u[idx_A(2,0)]=1
    v = np.zeros(m*p); v[idx_B(0,2)]=1
    w = np.zeros(p*n); w[idx_C(2,2)]=1
    add_term(u, v, w)
    
    u = np.zeros(n*m); u[idx_A(2,1)]=1
    v = np.zeros(m*p); v[idx_B(1,2)]=1
    w = np.zeros(p*n); w[idx_C(2,2)]=1
    add_term(u, v, w)
    
    u = np.zeros(n*m); u[idx_A(2,0)]=1
    v = np.zeros(m*p); v[idx_B(0,3)]=1
    w = np.zeros(p*n); w[idx_C(2,3)]=1
    add_term(u, v, w)
    
    u = np.zeros(n*m); u[idx_A(2,1)]=1
    v = np.zeros(m*p); v[idx_B(1,3)]=1
    w = np.zeros(p*n); w[idx_C(2,3)]=1
    add_term(u, v, w)

    # --- Block 8: C[2, 2:4] from A[2, 2] B[2, 2:4] (Rank 1) ---
    # A indices: (2,2)=8
    # B indices: (2,2)=10, (2,3)=11
    # C indices: (2,2)=8, (2,3)=11
    u = np.zeros(n*m); u[idx_A(2,2)]=1
    v = np.zeros(m*p); v[idx_B(2,2)]=1; v[idx_B(2,3)]=1
    w = np.zeros(p*n); w[idx_C(2,2)]=1; w[idx_C(2,3)]=1
    add_term(u, v, w)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
