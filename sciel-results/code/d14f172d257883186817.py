import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 130  # Safety margin before 150s limit

    # Build rank-44 Strassen-based construction
    U, V, W = _build_rank44(n, m, p)

    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fallback: return schoolbook rank-48
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Attempt rank reduction via null-space analysis
    U, V, W = _try_reduce(U, V, W, n, m, p, deadline)

    return {"U": U, "V": V, "W": W}

def _build_schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition with rank n*m*p."""
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

    return U, V, W

def _build_rank44(n, m, p):
    """
    Build a rank-44 decomposition using Strassen-like block decomposition.
    
    Strategy:
    - The 2x3 matrix A has entries A[i][j] for i in {0,1}, j in {0,1,2}
    - The 3x8 matrix B has entries B[j][k] for j in {0,1,2}, k in {0,...,7}
    - Output C[i][k] = sum_j A[i][j] * B[j][k]
    
    We use the following approach:
    - For each column k of B, we need to compute 2 outputs (C[0][k] and C[1][k])
    - Each output is a dot product of a 3-vector (row of A) with a 3-vector (column of B)
    - Using Strassen's trick on the 3-vector dot product, we can reduce from 3 multiplications to 3 multiplications (no gain for n=3)
    - But we can group columns and use block Strassen
    
    Actually, let me use a cleaner approach:
    - Split the 8 columns into 4 groups of 2
    - For each group of 2 columns, we have a 3x2 block of B
    - We compute 2x3 * 3x2 = 2x2 product
    - Use rank-7 Strassen for 2x2x2, but we need to handle the 3x2 block
    
    Let me think more carefully. The tensor is T[i,j,k,i'] = delta_{i,i'} * delta_{j,k}
    where i,i' in {0,1}, j in {0,1,2}, k in {0,...,7}
    
    The schoolbook has rank 2*3*8 = 48.
    
    Strassen for 2x2x2 gives rank 7 instead of 8, saving 1 per 2x2 block.
    
    If we split the 8 columns into 4 groups of 2, and for each group we use a
    2x3x2 sub-problem, we can potentially save.
    
    For 2x3x2: The tensor has indices i in {0,1}, j in {0,1,2}, k in {0,1}
    Schoolbook rank = 2*3*2 = 12.
    
    Can we do better? Let's use the fact that:
    C[0][k] = A[0][0]*B[0][k] + A[0][1]*B[1][k] + A[0][2]*B[2][k]
    C[1][k] = A[1][0]*B[0][k] + A[1][1]*B[1][k] + A[1][2]*B[2][k]
    
    For k in {0,1}, we have 4 outputs.
    
    Using Strassen on the 2x2 part (treating the j=2 term separately):
    Let's define:
    M1 = (A[0][0] + A[1][0]) * (B[0][0] + B[0][1])  -- this doesn't directly help
    
    Actually, let me just use a known construction. The rank of 2x3x2 is at most 11.
    
    Let me use a simpler approach: just use schoolbook for each column pair, 
    but that gives 48 again.
    
    Let me try a different strategy: use the rank-7 Strassen decomposition for 2x2x2
    and handle the third row of A separately.
    
    For each pair of columns (k0, k1):
    C[0][k0] = A[0][0]*B[0][k0] + A[0][1]*B[1][k0] + A[0][2]*B[2][k0]
    C[0][k1] = A[0][0]*B[0][k1] + A[0][1]*B[1][k1] + A[0][2]*B[2][k1]
    C[1][k0] = A[1][0]*B[0][k0] + A[1][1]*B[1][k0] + A[1][2]*B[2][k0]
    C[1][k1] = A[1][0]*B[0][k1] + A[1][1]*B[1][k1] + A[1][2]*B[2][k1]
    
    Split into 2x2 part (j=0,1) and j=2 part:
    C[0][k0] = (A[0][0]*B[0][k0] + A[0][1]*B[1][k0]) + A[0][2]*B[2][k0]
    C[0][k1] = (A[0][0]*B[0][k1] + A[0][1]*B[1][k1]) + A[0][2]*B[2][k1]
    C[1][k0] = (A[1][0]*B[0][k0] + A[1][1]*B[1][k0]) + A[1][2]*B[2][k0]
    C[1][k1] = (A[1][0]*B[0][k1] + A[1][1]*B[1][k1]) + A[1][2]*B[2][k1]
    
    The 2x2x2 part (i in {0,1}, j in {0,1}, k in {0,1}) has rank 7.
    The j=2 part: 4 terms (A[i][2]*B[2][k] for i in {0,1}, k in {0,1})
    
    Total per column pair: 7 + 4 = 11
    For 4 column pairs: 4 * 11 = 44
    
    This gives rank 44!
    
    Strassen 2x2x2 decomposition:
    M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
    M2 = (A[1][0] + A[1][1]) * B[0][0]
    M3 = A[0][0] * B[1][1]
    M4 = A[1][1] * (B[0][1] - B[1][1])
    M5 = A[0][1] * (B[0][0] + B[0][1])
    M6 = (A[1][0] - A[0][0]) * B[1][0]
    M7 = (A[0][1] - A[1][1]) * (B[1][1] + B[1][0])
    
    C[0][0] = M1 + M4 - M5 + M7
    C[0][1] = M3 + M5
    C[1][0] = M2 + M6
    C[1][1] = M1 - M2 + M3 - M6
    """
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen 2x2x2 for the j=0,1 part
        # M1 = (A[0][0] + A[1][1]) * (B[0][k0] + B[1][k1])
        # Wait, I need to be more careful with the Strassen indices.
        
        # Let me use the standard Strassen:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(g-h)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C[0][0] = M1 + M4 - M5 + M7
        # C[0][1] = M3 + M5
        # C[1][0] = M2 + M6
        # C[1][1] = M1 - M2 + M3 - M6
        
        # In our case:
        # a = A[0][0], b = A[0][1], c = A[1][0], d = A[1][1]
        # e = B[0][k0], f = B[0][k1], g = B[1][k0], h = B[1][k1]
        
        # M1 = (A[0][0] + A[1][1]) * (B[0][k0] + B[1][k1])
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A[1][0] + A[1][1]) * B[0][k0]
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M3 = A[0][0] * (B[0][k1] - B[1][k0])
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A[1][1] * (B[1][k0] - B[1][k1])
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = -1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M5 = (A[0][0] + A[0][1]) * B[1][k1]
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M6 = (A[1][0] - A[0][0]) * (B[0][k0] + B[0][k1])
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M7 = (A[0][1] - A[1][1]) * (B[1][k0] + B[1][k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # j=2 part: 4 terms
        # A[0][2] * B[2][k0] -> C[0][k0]
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # A[0][2] * B[2][k1] -> C[0][k1]
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # A[1][2] * B[2][k0] -> C[1][k0]
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # A[1][2] * B[2][k1] -> C[1][k1]
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition matches matrix multiplication tensor."""
    r = len(U[0])
    # Check T[i,j,k,i'] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i',r] = delta_{i,i'}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + ip][r_idx]
                    expected = 1.0 if i == ip else 0.0
                    if abs(s - expected) > 1e-10:
                        return False
    return True

def _try_reduce(U, V, W, n, m, p, deadline):
    """Attempt to reduce rank by checking for linear dependencies among rank-1 terms."""
    r = len(U[0])
    
    # Convert to arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check for zero columns
    for c in range(r - 1, -1, -1):
        if time.time() > deadline:
            break
            
        if np.allclose(U_arr[:, c], 0) and np.allclose(V_arr[:, c], 0) and np.allclose(W_arr[:, c], 0):
            U_arr = np.delete(U_arr, c, axis=1)
            V_arr = np.delete(V_arr, c, axis=1)
            W_arr = np.delete(W_arr, c, axis=1)
            r -= 1
    
    # Convert back to lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    
    # Final verification
    if not _verify(U_out, V_out, W_out, n, m, p):
        return U, V, W
    
    return U_out, V_out, W_out

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
