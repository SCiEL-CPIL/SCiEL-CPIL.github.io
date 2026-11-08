import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    timeout = 130.0  # Safety margin (evaluator kills at 150s)
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    np_ = n * p  # 10
    
    # Target tensor entries
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    target_set = set(target_indices)
    
    # Schoolbook solution as fallback
    R_school = n * m * p  # 30
    U_school = np.zeros((nm, R_school))
    V_school = np.zeros((mp, R_school))
    W_school = np.zeros((np_, R_school))
    for r, (a, b, c) in enumerate(target_indices):
        U_school[a, r] = 1
        V_school[b, r] = 1
        W_school[c, r] = 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # === Strassen-based rank 29 decomposition ===
    # Use Strassen for the 2x2x2 sub-problem: i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: a=i*3+j -> A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: b=j*5+k -> B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    # C indices: c=k*2+i -> C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    
    R = 29
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((np_, R))
    
    col = 0
    # Strassen 7 products
    # M1 = (A00-A11)(B11-B00) -> C00: +1, C11: +1
    U[0, col] = 1; U[4, col] = -1
    V[6, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    # M2 = (A00+A11)B01 -> C01: +1
    U[0, col] = 1; U[4, col] = 1
    V[1, col] = 1
    W[2, col] = 1
    col += 1
    # M3 = A00(B00-B11) -> C01: +1, C11: +1
    U[0, col] = 1
    V[0, col] = 1; V[6, col] = -1
    W[2, col] = 1; W[3, col] = 1
    col += 1
    # M4 = A11(B00+B11) -> C00: +1, C10: +1
    U[4, col] = 1
    V[0, col] = 1; V[6, col] = 1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    # M5 = (A00+A01)B11 -> C00: -1, C11: +1
    U[0, col] = 1; U[1, col] = 1
    V[6, col] = 1
    W[0, col] = -1; W[3, col] = 1
    col += 1
    # M6 = (A10-A00)(B00+B01) -> C10: +1
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[1, col] = 1
    col += 1
    # M7 = (A01+A11)(B10-B11) -> C00: +1, C11: +1
    U[1, col] = 1; U[4, col] = 1
    V[5, col] = 1; V[6, col] = -1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    
    # Schoolbook for remaining 22 terms
    sub_block = set([(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) in sub_block:
                    continue
                a = i*m+j; b = j*p+k; c = k*n+i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Verify
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    valid = True
    for a, b, c in target_indices:
        if abs(T_rec[a, b, c] - 1.0) > 1e-9:
            valid = False
            break
    if valid:
        for a in range(nm):
            for b in range(mp):
                for c in range(np_):
                    if (a,b,c) in target_set:
                        continue
                    if abs(T_rec[a,b,c]) > 1e-9:
                        valid = False
                        break
            if not valid:
                break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
