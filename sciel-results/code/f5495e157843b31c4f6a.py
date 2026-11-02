import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, safety margin
    
    n, m, p = 3, 4, 6
    terms = []
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Strassen 2x2x2: A[i1:i3, j1:j3] * B[j1:j3, k1:k3] -> C[i1:i3, k1:k3]"""
        a, b = (i1, j1), (i1, j2)
        c, d = (i2, j1), (i2, j2)
        e, f = (j1, k1), (j1, k2)
        g, h = (j2, k1), (j2, k2)
        C11, C12 = (i1, k1), (i1, k2)
        C21, C22 = (i2, k1), (i2, k2)
        
        # M1 = (a+d)(e+h)
        terms.append((
            {a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}
        ))
        # M2 = (c+d)e
        terms.append((
            {c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}
        ))
        # M3 = a(f-h)
        terms.append((
            {a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}
        ))
        # M4 = d(g-e)
        terms.append((
            {d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}
        ))
        # M5 = (a+b)h
        terms.append((
            {a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}
        ))
        # M6 = (c-a)(e+f)
        terms.append((
            {c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}
        ))
        # M7 = (b-d)(g+h)
        terms.append((
            {b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}
        ))
    
    def add_schoolbook(i, j, k):
        terms.append((
            {(i, j): 1}, {(j, k): 1}, {(i, k): 1}
        ))
    
    # Row 0: schoolbook (4 middle * 6 right = 24 terms)
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: Strassen on 2x2x2 blocks
    # Split m=4 into two 2-col groups, p=6 into three 2-col groups
    for j1 in [0, 2]:
        for k1 in [0, 2, 4]:
            if not time_ok():
                break
            add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1)
        if not time_ok():
            break
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff
    
    # Try to reduce rank via SVD on the combined factor structure
    if time_ok():
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # The decomposition is valid if U @ diag(1) @ V^T reconstructs the tensor.
        # We can try to find a lower-rank equivalent by checking linear dependencies.
        # Build the "bilinear" form: for each term t, we have u_t, v_t, w_t.
        # The tensor T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # If we can find coefficients such that some columns are linear combos, we can reduce.
        
        # Simple approach: check if any column is a linear combination of others
        # by computing the rank of [U; V; W] stacked appropriately.
        # Actually, a cleaner check: the rank of the decomposition is at most the
        # minimum of rank(U), rank(V), rank(W) after appropriate reshaping.
        
        # Let's just check numerical rank of each factor
        tol = 1e-10
        rU = np.linalg.matrix_rank(U_np, tol=tol)
        rV = np.linalg.matrix_rank(V_np, tol=tol)
        rW = np.linalg.matrix_rank(W_np, tol=tol)
        
        # If any factor has rank < 66, we might be able to reduce
        # But reducing one factor requires adjusting others, which is complex.
        # For now, just verify correctness.
        
        # Verify the decomposition
        # T[i,j,k] should equal sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # For a random test:
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for t in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback: return schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        U[i * m + j][idx] = 1
                        V[j * p + k][idx] = 1
                        W[k * n + i][idx] = 1
    
    return {"U": U, "V": V, "W": W}
