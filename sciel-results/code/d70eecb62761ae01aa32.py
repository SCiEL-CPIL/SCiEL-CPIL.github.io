"""Strassen-based decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]


def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 multiplication with rank 7 (Strassen).
    U shape (4,7): indexed [i*2+j, c]
    V shape (4,7): indexed [j*2+k, c]
    W shape (4,7): indexed [k*2+i, c]
    """
    # Strassen's 7 products for (A 2x2) * (B 2x2) = C 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4, M1-M2+M3+M6]]
    
    # U maps input A[i,j] to coefficients in the 7 products
    # Product 1: a+d  -> U[0,0]=1 (a), U[3,0]=1 (d)  [row i*2+j]
    # Product 2: (c+d)*e -> U[2,1]=1 (c), U[3,1]=1 (d)
    # Product 3: a*(f-h) -> U[0,2]=1 (a)
    # Product 4: d*(g-e) -> U[3,3]=1 (d)
    # Product 5: (a+b)*h -> U[0,4]=1 (a), U[1,4]=1 (b)
    # Product 6: (c-a)*(e+f) -> U[2,5]=1 (c), U[0,5]=-1 (a)
    # Product 7: (b-d)*(g+h) -> U[1,6]=1 (b), U[3,6]=-1 (d)
    
    U = np.zeros((4, 7))
    U[0, 0] = 1; U[3, 0] = 1       # M1: a+d
    U[2, 1] = 1; U[3, 1] = 1       # M2: c+d
    U[0, 2] = 1                     # M3: a
    U[3, 3] = 1                     # M4: d
    U[0, 4] = 1; U[1, 4] = 1       # M5: a+b
    U[2, 5] = 1; U[0, 5] = -1      # M6: c-a
    U[1, 6] = 1; U[3, 6] = -1      # M7: b-d
    
    # V maps input B[j,k] to coefficients
    # Product 1: e+h -> V[0,0]=1 (e), V[3,0]=1 (h)  [row j*2+k]
    # Product 2: e   -> V[0,1]=1 (e)
    # Product 3: f-h -> V[1,2]=1 (f), V[3,2]=-1 (h)
    # Product 4: g-e -> V[2,3]=1 (g), V[0,3]=-1 (e)
    # Product 5: h   -> V[3,4]=1 (h)
    # Product 6: e+f -> V[0,5]=1 (e), V[1,5]=1 (f)
    # Product 7: g+h -> V[2,6]=1 (g), V[3,6]=1 (h)
    
    V = np.zeros((4, 7))
    V[0, 0] = 1; V[3, 0] = 1       # M1: e+h
    V[0, 1] = 1                     # M2: e
    V[1, 2] = 1; V[3, 2] = -1      # M3: f-h
    V[2, 3] = 1; V[0, 3] = -1      # M4: g-e
    V[3, 4] = 1                     # M5: h
    V[0, 5] = 1; V[1, 5] = 1       # M6: e+f
    V[2, 6] = 1; V[3, 6] = 1       # M7: g+h
    
    # W maps output C[i,k] to coefficients
    # C[0,0] = M1+M4-M5+M7 -> W[0,0]=1, W[3,0]=1, W[0,4]=-1, W[3,0]... 
    # W indexed by [k*2+i, c]
    # C[0,0]: k=0,i=0 -> W[0,0]=1(M1), W[0,3]=1(M4), W[0,4]=-1(M5), W[0,6]=1(M7)
    # C[0,1]: k=0,i=1 -> W[2,2]=1(M3), W[2,4]=1(M5)
    # C[1,0]: k=1,i=0 -> W[1,1]=1(M2), W[1,3]=1(M4)
    # C[1,1]: k=1,i=1 -> W[3,0]=1(M1), W[3,1]=-1(M2), W[3,2]=1(M3), W[3,5]=1(M6)
    
    W = np.zeros((4, 7))
    # C[0,0]: k*n+i = 0*2+0 = 0
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C[0,1]: k*n+i = 0*2+1 = 1
    W[1, 2] = 1; W[1, 4] = 1
    # C[1,0]: k*n+i = 1*2+0 = 2
    W[2, 1] = 1; W[2, 3] = 1
    # C[1,1]: k*n+i = 1*2+1 = 3
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _validate(U, V, W, n=2, m=3, p=10):
    """
    Validate that U @ V^T contracted with W gives the matrix multiplication tensor.
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the full tensor contraction
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape to check
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Contract over c
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Should be all 1s
    expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T - expected))
    return max_err < 1e-9, max_err


def _build_strassen():
    """
    Build Strassen decomposition for 2x3 * 3x10.
    Partition the 10 output columns into 5 pairs, apply Strassen to each 2x3x2 block.
    The 3 middle rows are partitioned: for each pair of output columns, we use
    all 3 middle rows (but Strassen is for 2x2, so we handle the 3rd row separately).
    
    Actually, let me think more carefully. We have A (2x3) * B (3x10) = C (2x10).
    
    For the 3 middle dimension, we can't directly use 2x2 Strassen on a 2x3*3x2 block.
    Instead, let's partition differently:
    - Split the 3 middle rows into a 2-row group and a 1-row group.
    - For the 2-row group, we can use Strassen-like tricks.
    
    Alternative: Split output into 5 pairs. For each pair of output columns (k, k+1),
    we have a sub-problem: A (2x3) * B_sub (3x2) = C_sub (2x2).
    
    For 2x3 * 3x2, the naive rank is 2*3*2 = 12. Can we do better?
    
    Let me think about this as a tensor of size 2x3x2. The rank of this tensor...
    Actually, for a 2x3*3x2 multiplication, we can write it as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    One approach: split the 3 middle dimension into 2+1.
    For the 2x2*2x2 part (rows 0,1 of middle), use Strassen (rank 7).
    For the remaining row (row 2 of middle), use naive (rank 4: 2*1*2).
    Total per block: 7 + 4 = 11. For 5 blocks: 55.
    
    But wait, that's what the current approach does. Let me verify.
    
    Actually, let me try a different partition. What if we group the middle dimension
    differently across blocks?
    
    For block b (output columns 2b, 2b+1):
    C[i, 2b+k] = sum_{j=0}^{2} A[i,j] * B[j, 2b+k]
    
    = A[i,0]*B[0,2b+k] + A[i,1]*B[1,2b+k] + A[i,2]*B[2,2b+k]
    
    We can use Strassen on the first two terms (treating them as a 2x2*2x2 problem
    where the "2x2" A is [[A[i,0], A[i,1]] for i=0,1] and "2x2" B is 
    [[B[0,2b], B[0,2b+1]], [B[1,2b], B[1,2b+1]]]) plus the third term naively.
    
    Strassen for the 2x2 part: rank 7
    Naive for the 1x2 part: A[i,2]*B[2,2b+k] for i in {0,1}, k in {0,1}: rank 4
    Total per block: 11
    Total: 5 * 11 = 55
    
    That matches the current best. To do better, we need to find linear dependencies
    across blocks or use a smarter decomposition.
    
    Let me try: what if we use a 3x3 Strassen-like decomposition for the full 2x3*3x2?
    The rank of the 2x3x2 multiplication tensor... 
    """
    n, m, p = 2, 3, 10
    
    # Build the full Strassen decomposition
    # 5 blocks, each using Strassen on 2x2 sub-part + naive on remaining row
    sU, sV, sW = _strassen_2x2x2()
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    for block in range(5):
        k_start = block * 2  # output columns k_start, k_start+1
        
        # Strassen part: uses middle rows 0,1
        # U_strassen: 2x2 input (rows 0,1 of A) -> 7 products
        # V_strassen: 2x2 input (cols k_start, k_start+1 of B, rows 0,1) -> 7 products
        # W_strassen: 2x2 output -> 7 products
        
        # Map to full indices:
        # U: rows i*m+j = i*3+j for i in {0,1}, j in {0,1}
        # V: rows j*p+k = j*10+k for j in {0,1}, k in {k_start, k_start+1}
        # W: rows k*n+i = k*2+i for k in {0,1} (local), i in {0,1}
        
        for c in range(7):
            # U contribution
            U_col = np.zeros(n * m)
            # sU is 4x7, indexed [i*2+j, c]
            for i in range(2):
                for j in range(2):
                    U_col[i * m + j] = sU[i * 2 + j, c]
            
            # V contribution
            V_col = np.zeros(m * p)
            # sV is 4x7, indexed [j*2+k, c]
            for j in range(2):
                for k_local in range(2):
                    V_col[j * p + k_start + k_local] = sV[j * 2 + k_local, c]
            
            # W contribution
            W_col = np.zeros(p * n)
            # sW is 4x7, indexed [k*2+i, c]
            for k_local in range(2):
                for i in range(2):
                    W_col[(k_start + k_local) * n + i] = sW[k_local * 2 + i, c]
            
            U_list.append(U_col)
            V_list.append(V_col)
            W_list.append(W_col)
            total_rank += 1
        
        # Naive part: uses middle row 2
        # A[i,2] * B[2, k_start+k] for i in {0,1}, k in {0,1}
        for i in range(2):
            for k_local in range(2):
                k_global = k_start + k_local
                
                U_col = np.zeros(n * m)
                U_col[i * m + 2] = 1  # A[i, 2]
                
                V_col = np.zeros(m * p)
                V_col[2 * p + k_global] = 1  # B[2, k_global]
                
                W_col = np.zeros(p * n)
                W_col[k_global * n + i] = 1  # C[i, k_global]
                
                U_list.append(U_col)
                V_list.append(V_col)
                W_list.append(W_col)
                total_rank += 1
    
    U = np.array(U_list).T  # (n*m, total_rank)
    V = np.array(V_list).T  # (m*p, total_rank)
    W = np.array(W_list).T  # (p*n, total_rank)
    
    # Validate
    valid, err = _validate(U, V, W)
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try to reduce rank by finding linear dependencies
    # Check if any columns in U, V, W are linearly dependent
    # We can try to find a subset of columns that still spans the same tensor
    # This is equivalent to finding the tensor rank, which is hard.
    # Instead, let's try a greedy approach: remove columns that are linear combinations.
    
    # For each column c, check if removing it still gives a valid decomposition.
    # This is O(rank * rank^2) which is manageable for rank ~55.
    
    U_float = U.copy()
    V_float = V.copy()
    W_float = W.copy()
    current_rank = total_rank
    
    # Try removing columns one by one (greedy)
    removed = [False] * current_rank
    improved = True
    while improved:
        improved = False
        for c in range(current_rank):
            if removed[c]:
                continue
            # Try removing column c
            mask = [i for i in range(current_rank) if not removed[i] and i != c]
            if len(mask) < 1:
                continue
            U_sub = U_float[:, mask]
            V_sub = V_float[:, mask]
            W_sub = W_float[:, mask]
            
            valid_sub, _ = _validate(U_sub, V_sub, W_sub)
            if valid_sub:
                removed[c] = True
                improved = True
                break
    
    if improved:
        mask = [i for i in range(current_rank) if not removed[i]]
        U = U_float[:, mask]
        V = V_float[:, mask]
        W = W_float[:, mask]
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition
    try:
        U, V, W = _build_strassen()
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            # Too close to deadline, return schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
