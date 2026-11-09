import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    def build_strassen_block(i1, i2, j1, j2, k1, k2):
        """
        Return 7 Strassen terms for a 2x2x2 block.
        Each term is a dict with 'U', 'V', 'W' lists of (index, value) pairs.
        
        The tensor must satisfy:
        sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k) in block.
        
        Strassen's 7 products for C = A * B (2x2):
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)(B00)
        M3 = (A00)(B01-B11)
        M4 = (A11)(B10-B00)
        M5 = (A00+A01)(B11)
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 - M4 + M5 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        
        In our tensor: A[i,j] -> U[i*m+j], B[j,k] -> V[j*p+k], C[k,i] -> W[k*n+i]
        Local: a=(i,j) with i in {i1,i2}, j in {j1,j2}
               b=(j,k) with j in {j1,j2}, k in {k1,k2}
               c=(k,i) with k in {k1,k2}, i in {i1,i2}
        
        Mapping: A00=U[i1*m+j1], A01=U[i1*m+j2], A10=U[i2*m+j1], A11=U[i2*m+j2]
                 B00=V[j1*p+k1], B01=V[j1*p+k2], B10=V[j2*p+k1], B11=V[j2*p+k2]
                 C00=W[k1*n+i1], C01=W[k2*n+i1], C10=W[k1*n+i2], C11=W[k2*n+i2]
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00 gets +1, C01 gets +1, C10 gets +1, C11 gets +1
        # But we need to distribute so that the product U*V*W gives the right contributions
        # Term r: U[i1,j1]=1, U[i2,j2]=1; V[j1,k1]=1, V[j2,k2]=1; W[k1,i1]=1, W[k2,i2]=1
        # This gives 1 at (i1,j1,k1), (i1,j1,k2), (i2,j2,k1), (i2,j2,k2)
        terms.append({
            'U': [(i1*m+j1, 1), (i2*m+j2, 1)],
            'V': [(j1*p+k1, 1), (j2*p+k2, 1)],
            'W': [(k1*n+i1, 1), (k2*n+i2, 1)]
        })
        
        # M2 = (A10+A11)(B00)
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10 gets +1, C11 gets -1
        terms.append({
            'U': [(i2*m+j1, 1), (i2*m+j2, 1)],
            'V': [(j1*p+k1, 1)],
            'W': [(k1*n+i2, 1), (k2*n+i2, -1)]
        })
        
        # M3 = (A00)(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01 gets +1, C11 gets +1
        terms.append({
            'U': [(i1*m+j1, 1)],
            'V': [(j1*p+k2, 1), (j2*p+k2, -1)],
            'W': [(k2*n+i1, 1), (k2*n+i2, 1)]
        })
        
        # M4 = (A11)(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00 gets +1, C10 gets +1
        terms.append({
            'U': [(i2*m+j2, 1)],
            'V': [(j2*p+k1, 1), (j1*p+k1, -1)],
            'W': [(k1*n+i1, 1), (k1*n+i2, 1)]
        })
        
        # M5 = (A00+A01)(B11)
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00 gets +1, C01 gets +1
        terms.append({
            'U': [(i1*m+j1, 1), (i1*m+j2, 1)],
            'V': [(j2*p+k2, 1)],
            'W': [(k1*n+i1, 1), (k2*n+i1, 1)]
        })
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C10 gets +1
        terms.append({
            'U': [(i2*m+j1, 1), (i1*m+j1, -1)],
            'V': [(j1*p+k1, 1), (j1*p+k2, 1)],
            'W': [(k1*n+i2, 1)]
        })
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00 gets -1
        terms.append({
            'U': [(i1*m+j2, 1), (i2*m+j2, -1)],
            'V': [(j2*p+k1, 1), (j2*p+k2, 1)],
            'W': [(k1*n+i1, -1)]
        })
        
        return terms
    
    def build_decomposition():
        # Define blocks
        i_blocks = [[0, 1], [2, 3]]
        j_blocks = [[0, 1], [2, 3], [4]]
        k_blocks = [[0, 1], [2, 3], [4]]
        
        # Collect all terms
        terms = []
        
        for I in i_blocks:
            for J in j_blocks:
                for K in k_blocks:
                    si, sj, sk = len(I), len(J), len(K)
                    
                    if si == 2 and sj == 2 and sk == 2:
                        # Use Strassen
                        i1, i2 = I
                        j1, j2 = J
                        k1, k2 = K
                        terms.extend(build_strassen_block(i1, i2, j1, j2, k1, k2))
                    else:
                        # Standard method
                        for i in I:
                            for j in J:
                                for k in K:
                                    terms.append({
                                        'U': [(i*m+j, 1)],
                                        'V': [(j*p+k, 1)],
                                        'W': [(k*n+i, 1)]
                                    })
        
        # Build matrices
        rank = len(terms)
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        for col, term in enumerate(terms):
            for idx, val in term['U']:
                U[idx][col] += val
            for idx, val in term['V']:
                V[idx][col] += val
            for idx, val in term['W']:
                W[idx][col] += val
        
        return U, V, W, rank
    
    U, V, W, rank = build_decomposition()
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is correct."""
        nm, mp, pn = n*m, m*p, p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to standard decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_rank = rank
    
    # Quick check: remove any all-zero columns
    for r in range(current_rank - 1, -1, -1):
        if np.all(U_np[:, r] == 0) and np.all(V_np[:, r] == 0) and np.all(W_np[:, r] == 0):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            current_rank -= 1
    
    # Convert back to lists with integer entries
    U_final = []
    for i in range(nm):
        row = []
        for r in range(current_rank):
            val = U_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            else:
                row.append(val)
        U_final.append(row)
    
    V_final = []
    for i in range(mp):
        row = []
        for r in range(current_rank):
            val = V_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            else:
                row.append(val)
        V_final.append(row)
    
    W_final = []
    for i in range(pn):
        row = []
        for r in range(current_rank):
            val = W_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            else:
                row.append(val)
        W_final.append(row)
    
    return {"U": U_final, "V": V_final, "W": W_final}
