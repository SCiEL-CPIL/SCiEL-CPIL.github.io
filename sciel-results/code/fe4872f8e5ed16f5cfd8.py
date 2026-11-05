"""Matrix multiplication 5x5x5 - rank-minimal decomposition using Strassen hybrid."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: C[i,j] = sum_r U[i,r]*V[j,r]*W[r] where W encodes the product structure.
    Actually for tensor: T[i,j,k] = sum_r U[i,r]*V[k,r]*W[j,r] 
    where i=row of A (0..1), k=col of A/row of B (0..1), j=col of B (0..1)
    W index: k*n+i with n=2, so W index = k*2+i
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    n = 2
    rank = 7
    U = np.zeros((n*n, rank))  # A part: index i*n+k -> but convention is i*m+j, m=2, so i*2+j
    V = np.zeros((n*n, rank))  # B part: index k*n+j, n=2, so k*2+j
    W = np.zeros((n*n, rank))  # C part: index j*n+i, n=2, so j*2+i
    
    # U[i*m+j, r] where m=2 (shared dim size), i in 0..1, j in 0..1
    # V[k*p+l, r] where p=2, k in 0..1, l in 0..1
    # W[l*n+i, r] where n=2, l in 0..1, i in 0..1
    
    # M1 = (A00+A11)*(B00+B11)
    U[0,0] = 1  # A00: i=0,j=0
    U[3,0] = 1  # A11: i=1,j=1
    V[0,0] = 1  # B00: k=0,l=0
    V[3,0] = 1  # B11: k=1,l=1
    W[0,0] = 1  # C00: l=0,i=0
    W[3,0] = 1  # C11: l=1,i=1
    
    # M2 = (A10+A11)*B00
    U[2,1] = 1  # A10: i=1,j=0
    U[3,1] = 1  # A11: i=1,j=1
    V[0,1] = 1  # B00: k=0,l=0
    W[2,1] = 1  # C10: l=0,i=1
    
    # M3 = A00*(B01-B11)
    U[0,2] = 1  # A00
    V[1,2] = 1  # B01: k=0,l=1
    V[3,2] = -1 # B11: k=1,l=1
    W[1,2] = 1  # C01: l=1,i=0
    
    # M4 = A11*(B10-B00)
    U[3,3] = 1  # A11
    V[2,3] = 1  # B10: k=1,l=0
    V[0,3] = -1 # B00: k=0,l=0
    W[0,3] = 1  # C00
    W[2,3] = 1  # C10
    
    # M5 = (A00+A01)*B11
    U[0,4] = 1  # A00
    U[1,4] = 1  # A01: i=0,j=1
    V[3,4] = 1  # B11
    W[0,4] = -1 # C00
    W[1,4] = 1  # C01
    
    # M6 = (A10-A00)*(B00+B01)
    U[2,5] = 1  # A10
    U[0,5] = -1 # A00
    V[0,5] = 1  # B00
    V[1,5] = 1  # B01
    W[3,5] = 1  # C11
    
    # M7 = (A01-A11)*(B10+B11)
    U[1,6] = 1  # A01
    U[3,6] = -1 # A11
    V[2,6] = 1  # B10
    V[3,6] = 1  # B11
    W[0,6] = 1  # C00
    W[3,6] = 1  # C11
    
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for n x m times m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _concat_factors(factors_list):
    """Concatenate list of (U,V,W) factor triples horizontally.
    Each triple has U: (rows_u, r), V: (rows_v, r), W: (rows_w, r)
    Returns (U_all, V_all, W_all) with total rank = sum of r's.
    """
    if not factors_list:
        return [], [], []
    U_list = []
    V_list = []
    W_list = []
    for U, V, W in factors_list:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
    # Concatenate along axis 1 (columns)
    U_all = [row + row2 for row, row2 in zip(*[U_list[0]]*1)]  # placeholder
    # Do it properly
    rank_total = sum(len(W) for _,_,W in factors_list)
    rows_u = len(factors_list[0][0])
    rows_v = len(factors_list[0][1])
    rows_w = len(factors_list[0][2])
    U_all = [[0]*rank_total for _ in range(rows_u)]
    V_all = [[0]*rank_total for _ in range(rows_v)]
    W_all = [[0]*rank_total for _ in range(rows_w)]
    offset = 0
    for U, V, W in factors_list:
        r = len(W)
        for i in range(rows_u):
            for c in range(r):
                U_all[i][offset+c] = U[i][c]
        for i in range(rows_v):
            for c in range(r):
                V_all[i][offset+c] = V[i][c]
        for i in range(rows_w):
            for c in range(r):
                W_all[i][offset+c] = W[i][c]
        offset += r
    return U_all, V_all, W_all

def _block_decompose_5x5x5():
    """Block decomposition of 5x5x5 matmul using partition [2,2,1].
    
    Partition: rows of A = [2,2,1], cols of A/rows of B = [2,2,1], cols of B = [2,2,1]
    
    For each block (i,k) of A (size ri x ck) and block (k,j) of B (size ck x dj),
    we need a ri x ck x dj tensor decomposition.
    
    Block sizes:
    - (2,2)x(2,2): 2x2x2 -> Strassen rank 7
    - (2,2)x(2,1): 2x2x1 -> schoolbook rank 4
    - (2,1)x(2,2): 2x1x2 -> schoolbook rank 4
    - (2,1)x(2,1): 2x1x1 -> schoolbook rank 2
    - (1,2)x(2,2): 1x2x2 -> schoolbook rank 4
    - (1,2)x(2,1): 1x2x1 -> schoolbook rank 2
    - (1,1)x(2,2): 1x1x2 -> schoolbook rank 2
    - (1,1)x(2,1): 1x1x1 -> schoolbook rank 1
    - (2,2)x(1,2): 2x2x2 -> Strassen rank 7  [wait, this is same as first]
    
    Actually the blocks are indexed by (i,k,j) where i indexes row-blocks of A/C,
    k indexes col-blocks of A / row-blocks of B, j indexes col-blocks of B/C.
    
    Partition sizes: s = [2,2,1]
    For each (i,k,j): A block is s[i] x s[k], B block is s[k] x s[j], C block is s[i] x s[j]
    The tensor is s[i] x s[k] x s[j]
    
    Total rank = sum over all (i,k,j) of rank(s[i] x s[k] x s[j])
    """
    s = [2, 2, 1]
    n = m = p = 5
    
    # Precompute decompositions for each unique (ri, ck, dj)
    decomp_cache = {}
    def get_decomp(ri, ck, dj):
        key = (ri, ck, dj)
        if key not in decomp_cache:
            if ri == 2 and ck == 2 and dj == 2:
                decomp_cache[key] = _strassen_2x2x2()
            else:
                decomp_cache[key] = _schoolbook_nm_p(ri, ck, dj)
        return decomp_cache[key]
    
    # For each block triple (i, k, j), we have a local decomposition.
    # We need to embed it into the global coordinate system.
    # 
    # Global U index: row_in_A * m + col_in_A, where row_in_A is 0..4, col_in_A is 0..4
    #   Block i covers rows [sum(s[0:i]), sum(s[0:i+1])) of A
    #   Block k covers cols [sum(s[0:k]), sum(s[0:k+1])) of A
    # Global V index: row_in_B * p + col_in_B
    #   Block k covers rows [sum(s[0:k]), sum(s[0:k+1])) of B
    #   Block j covers cols [sum(s[0:j]), sum(s[0:j+1])) of B
    # Global W index: col_in_C * n + row_in_C  (W[k*n+i] convention)
    #   Block j covers cols [sum(s[0:j]), sum(s[0:j+1])) of C
    #   Block i covers rows [sum(s[0:i]), sum(s[0:i+1])) of C
    
    row_offsets = [0, s[0], s[0]+s[1]]  # [0, 2, 4]
    col_offsets = [0, s[0], s[0]+s[1]]  # [0, 2, 4]
    
    all_factors = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                ri, ck, dj = s[i], s[k], s[j]
                U_local, V_local, W_local = get_decomp(ri, ck, dj)
                
                # Embed into global coordinates
                r = len(W_local)
                U_global = [[0]*(r) for _ in range(ri*ck)]
                V_global = [[0]*(r) for _ in range(ck*dj)]
                W_global = [[0]*(r) for _ in range(dj*ri)]
                
                for local_i in range(ri):
                    for local_k in range(ck):
                        global_i = row_offsets[i] + local_i
                        global_k = col_offsets[k] + local_k
                        for c in range(r):
                            U_global[local_i*ck+local_k][c] = U_local[local_i*ck+local_k][c]
                
                for local_k in range(ck):
                    for local_j in range(dj):
                        global_k = row_offsets[k] + local_k
                        global_j = col_offsets[j] + local_j
                        for c in range(r):
                            V_global[local_k*dj+local_j][c] = V_local[local_k*dj+local_j][c]
                
                for local_j in range(dj):
                    for local_i in range(ri):
                        global_j = col_offsets[j] + local_j
                        global_i = row_offsets[i] + local_i
                        for c in range(r):
                            W_global[local_j*ri+local_i][c] = W_local[local_j*ri+local_i][c]
                
                all_factors.append((U_global, V_global, W_global))
    
    # Now concatenate all factors, but they have different sizes.
    # We need to map each local factor to the full global size.
    # Better approach: build the full U, V, W directly.
    
    total_rank = sum(len(W) for _,_,W in all_factors)
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                ri, ck, dj = s[i], s[k], s[j]
                U_local, V_local, W_local = get_decomp(ri, ck, dj)
                r = len(W_local)
                
                for local_i in range(ri):
                    for local_k in range(ck):
                        global_i = row_offsets[i] + local_i
                        global_k = col_offsets[k] + local_k
                        for c in range(r):
                            U_full[global_i*m + global_k][offset+c] = U_local[local_i*ck+local_k][c]
                
                for local_k in range(ck):
                    for local_j in range(dj):
                        global_k = row_offsets[k] + local_k
                        global_j = col_offsets[j] + local_j
                        for c in range(r):
                            V_full[global_k*p + global_j][offset+c] = V_local[local_k*dj+local_j][c]
                
                for local_j in range(dj):
                    for local_i in range(ri):
                        global_j = col_offsets[j] + local_j
                        global_i = row_offsets[i] + local_i
                        for c in range(r):
                            W_full[global_j*n + global_i][offset+c] = W_local[local_j*ri+local_i][c]
                
                offset += r
    
    return {"U": U_full, "V": V_full, "W": W_full}

def construct():
    """Construct rank-minimal decomposition for 5x5x5 matrix multiplication."""
    start_time = time.time()
    
    # Try block decomposition with Strassen hybrid
    try:
        result = _block_decompose_5x5x5()
        rank = len(result["U"][0])
        
        # Verify correctness with a quick check
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        # Check: T[i,j,k] = sum_r U[i,r] * V[k,r] * W[j,r]
        # where i = row of A (0..4), j = col of B (0..4), k = shared dim (0..4)
        # But our convention: U index = i*m+j where i=row_A, j=col_A (shared)
        # V index = k*p+l where k=row_B (shared), l=col_B
        # W index = l*n+i where l=col_C, i=row_C
        # 
        # The tensor T[a,b,c] for matmul: C = A*B
        # C[i,j] = sum_k A[i,k]*B[k,j]
        # So T[i,k,j] = A[i,k]*B[k,j] contributes to C[i,j]
        # In our factorization: T[i,k,j] = sum_r U[i*5+k, r] * V[k*5+j, r] * W[j*5+i, r]
        
        # Verify with random matrices
        np.random.seed(42)
        A = np.random.randint(-5, 5, (5,5)).astype(float)
        B = np.random.randint(-5, 5, (5,5)).astype(float)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for r in range(rank):
                        C_computed[i,j] += U[i*5+k, r] * V[k*5+j, r] * W[j*5+i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook
            n, m, p = DIMENSIONS
            rank_sb = n*m*p
            U_sb = [[0]*rank_sb for _ in range(n*m)]
            V_sb = [[0]*rank_sb for _ in range(m*p)]
            W_sb = [[0]*rank_sb for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j][col] = 1
                        V_sb[j*p+k][col] = 1
                        W_sb[k*n+i][col] = 1
                        col += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        elapsed = time.time() - start_time
        if elapsed > 140:
            # Safety margin
            pass
        
        return result
        
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank_sb = n*m*p
        U_sb = [[0]*rank_sb for _ in range(n*m)]
        V_sb = [[0]*rank_sb for _ in range(m*p)]
        W_sb = [[0]*rank_sb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
