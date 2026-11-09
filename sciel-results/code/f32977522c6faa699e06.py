import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition matrices
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # 00
        [0, 0, 0, 0, 1, 0, 1],   # 01
        [0, 1, 0, 0, 0, 1, 0],   # 10
        [1, 1, 0, 1, 0, 0, -1]   # 11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # 00
        [0, 0, 1, 0, 0, 1, 0],   # 01
        [0, 0, 0, 1, 0, 0, 1],   # 10
        [1, 0, -1, 0, 1, 0, 1]   # 11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # 0: C_00
        [0, 1, 0, 1, 0, 0, 0],   # 1: C_10
        [0, 0, 1, 0, 1, 0, 0],   # 2: C_01
        [1, -1, 1, 0, 0, 1, 0]   # 3: C_11
    ]

    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U and V
    for I in range(2):
        for J in range(2):
            block_idx = I * 2 + J
            for i in range(2):
                for j in range(2):
                    inner_idx = i * 2 + j
                    u_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    v_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    
                    for r in range(7):
                        ub = U2[block_idx][r]
                        vb = V2[block_idx][r]
                        if ub == 0 and vb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            ui = U2[inner_idx][s]
                            vi = V2[inner_idx][s]
                            if ui == 0 and vi == 0:
                                continue
                            U[u_idx][c] = ub * ui
                            V[v_idx][c] = vb * vi

    # Fill W
    for I in range(2):
        for J in range(2):
            block_idx = J * 2 + I
            for i in range(2):
                for j in range(2):
                    inner_idx = j * 2 + i
                    w_idx = (J * 2 + j) * 4 + (I * 2 + i)
                    
                    for r in range(7):
                        wb = W2[block_idx][r]
                        if wb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            wi = W2[inner_idx][s]
                            if wi == 0:
                                continue
                            W[w_idx][c] = wb * wi

    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for a in range(n):
            for b in range(p):
                for c in range(m):
                    lhs = 0.0
                    for r in range(rank):
                        lhs += U[a * m + c][r] * V[c * p + b][r]
                    rhs = W[b * n + a][r]  # This is wrong, need to sum over r
                    # Actually, the decomposition is:
                    # C[a,b] = sum_r U[a*m+c, r] * V[c*p+b, r] * W[b*n+a, r]
                    # Wait, that's not right either. Let me think about this more carefully.
                    pass
        # The tensor decomposition for matrix multiplication:
        # T_{a,c,b} = sum_r U_{a,c,r} * V_{c,b,r} * W_{a,b,r}
        # where T is the matrix multiplication tensor
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # Index convention: U[i,j] where i = a*m + c, j = r
        #                   V[i,j] where i = c*p + b, j = r  
        #                   W[i,j] where i = b*n + a, j = r (trace convention W[k*n+i] where k=col, i=row)
        # So C[a,b] = sum_c sum_r U[a*m+c, r] * V[c*p+b, r] * W[b*n+a, r]
        # But wait, that would give C[a,b] = sum_{c,r} U[a*m+c,r] * V[c*p+b,r] * W[b*n+a,r]
        # This doesn't look right. The standard form is:
        # C = A * B, so C[a,b] = sum_c A[a,c] * B[c,b]
        # The tensor is T_{a,c,b} = A[a,c] * B[c,b]
        # A rank-r decomposition: T_{a,c,b} = sum_r U[a,c,r] * V[c,b,r] * W[a,b,r]
        # So C[a,b] = sum_c sum_r U[a*m+c, r] * V[c*p+b, r] * W[a,b, r]
        # But W should be indexed as W[b*n+a] per the trace convention...
        # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed by (k, i) where k is column and i is row of output?
        # Or is it W[row, col]? The convention W[k*n+i] suggests linear index k*n+i
        # If k is the column index and i is the row index, then W[b*n+a] means row=a, col=b
        # So W[a,b] = W[b*n+a]
        # Then C[a,b] = sum_c sum_r U[a*m+c, r] * V[c*p+b, r] * W[b*n+a, r]
        # Let's verify with a simple case
        
        import numpy as np
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randint(-10, 10, (n, m)).astype(float)
        B = np.random.randint(-10, 10, (m, p)).astype(float)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for a in range(n):
            for b in range(p):
                for c in range(m):
                    for r in range(rank):
                        C_computed[a, b] += U_arr[a * m + c, r] * V_arr[c * p + b, r] * W_arr[b * n + a, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if Strassen construction is wrong
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for a in range(n):
            for c in range(m):
                for b in range(p):
                    r = a * m * p + c * p + b
                    U[a * m + c][r] = 1
                    V[c * p + b][r] = 1
                    W[b * n + a][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by zeroing out columns that don't contribute
    # This is a greedy approach - try removing each column and see if it still works
    def can_remove_column(U, V, W, col_to_remove, n, m, p):
        rank = len(U[0])
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        for row in U_test:
            row.pop(col_to_remove)
        for row in V_test:
            row.pop(col_to_remove)
        for row in W_test:
            row.pop(col_to_remove)
        return verify(U_test, V_test, W_test, n, m, p)
    
    # Try to remove redundant columns
    removed = True
    while removed and time.time() < deadline:
        removed = False
        for col in range(len(U[0])):
            if can_remove_column(U, V, W, col, n, m, p):
                for row in U:
                    row.pop(col)
                for row in V:
                    row.pop(col)
                for row in W:
                    row.pop(col)
                removed = True
                break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
