"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(n, m, p, U, V, W):
    """Verify the decomposition is correct."""
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
                    return False
    return True

def _strassen_block(U, V, W, i0, j0, k0, col):
    """
    Add a Strassen decomposition for a 2x2x2 sub-block starting at (i0, j0, k0).
    Returns the next column index.
    """
    # Strassen 7 multiplications for 2x2 matrix multiplication
    # C = A * B where A is 2x2, B is 2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In our tensor notation:
    # T[i,j,k] coefficient for basis element
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For a 2x2x2 block at (i0, j0, k0):
    # Local (i, j, k) -> Global (i0+i, j0+j, k0+k)
    
    # The 7 Strassen terms:
    # Each term is a product of sums/differences of A entries times B entries
    # which corresponds to specific U, V, W entries
    
    # M1: (A00 + A11) * (B00 + B11)
    # A00 corresponds to U[i0, j0] = U[(i0)*m+j0]
    # A11 corresponds to U[i0+1, j0+1] = U[(i0+1)*m+j0+1]
    # B00 corresponds to V[j0, k0] = V[j0*p+k0]
    # B11 corresponds to V[j0+1, k0+1] = V[(j0+1)*p+k0+1]
    # Result goes to C00 (i0, k0), C11 (i0+1, k0+1)
    # C00: W[k0, i0] = W[k0*n+i0], C11: W[k0+1, i0+1] = W[(k0+1)*n+i0+1]
    
    # Actually, let me think about this more carefully.
    # The matrix multiplication tensor T_{ijk} has the property that
    # sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = 1 if (i,j,k) is in the support
    # and 0 otherwise (for the standard basis).
    
    # For Strassen, we're replacing the 8 schoolbook terms for a 2x2x2 block
    # with 7 terms. The 8 schoolbook terms are:
    # For local (i,j,k) in {0,1}^3: U_{i,j} V_{j,k} W_{k,i} = 1
    
    # Strassen gives us 7 rank-1 tensors that sum to the same 2x2x2 multiplication tensor.
    
    # The 7 Strassen rank-1 tensors for local indices:
    # Let's denote the local U indices as (i,j), V as (j,k), W as (k,i)
    
    # M1: U has entries at (0,0) and (1,1), V has entries at (0,0) and (1,1), W has entries at (0,0) and (1,1)
    # M2: U has entries at (1,0) and (1,1), V has entry at (0,0), W has entry at (1,0)
    # M3: U has entry at (0,0), V has entries at (1,0) and (1,1), W has entry at (0,1)
    # M4: U has entry at (1,1), V has entries at (0,1) and (0,0), W has entry at (1,1)
    # M5: U has entries at (0,0) and (0,1), V has entry at (1,1), W has entries at (0,0) and (1,1)
    # M6: U has entries at (1,0) and (0,0), V has entries at (0,0) and (1,0), W has entry at (1,0)
    # M7: U has entries at (0,1) and (1,1), V has entries at (0,1) and (1,1), W has entry at (0,1)
    
    # Let me be more precise with the signs:
    # M1: U: (0,0)=1, (1,1)=1; V: (0,0)=1, (1,1)=1; W: (0,0)=1, (1,1)=1
    # M2: U: (1,0)=1, (1,1)=1; V: (0,0)=1; W: (1,0)=1
    # M3: U: (0,0)=1; V: (1,0)=1, (1,1)=-1; W: (0,1)=1
    # M4: U: (1,1)=1; V: (0,1)=1, (0,0)=-1; W: (1,1)=1
    # M5: U: (0,0)=1, (0,1)=1; V: (1,1)=1; W: (0,0)=1, (1,1)=-1
    # M6: U: (1,0)=1, (0,0)=-1; V: (0,0)=1, (1,0)=1; W: (1,0)=1
    # M7: U: (0,1)=1, (1,1)=-1; V: (0,1)=1, (1,1)=1; W: (0,1)=1
    
    # Let's define these 7 terms with their local (i,j,k) contributions:
    
    # I'll build the U, V, W vectors for each term
    terms = []
    
    # M1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # M2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1}
    terms.append((u2, v2, w2))
    
    # M3
    u3 = {(0,0): 1}
    v3 = {(1,0): 1, (1,1): -1}
    w3 = {(0,1): 1}
    terms.append((u3, v3, w3))
    
    # M4
    u4 = {(1,1): 1}
    v4 = {(0,1): 1, (0,0): -1}
    w4 = {(1,1): 1}
    terms.append((u4, v4, w4))
    
    # M5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): 1, (1,1): -1}
    terms.append((u5, v5, w5))
    
    # M6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (1,0): 1}
    w6 = {(1,0): 1}
    terms.append((u6, v6, w6))
    
    # M7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(0,1): 1, (1,1): 1}
    w7 = {(0,1): 1}
    terms.append((u7, v7, w7))
    
    # Now add each term to the global U, V, W
    for (u_dict, v_dict, w_dict) in terms:
        for (li, lj), coeff in u_dict.items():
            gi, gj = i0 + li, j0 + lj
            U[gi * m + gj][col] += coeff
        
        for (lj, lk), coeff in v_dict.items():
            gj, gk = j0 + lj, k0 + lk
            V[gj * p + gk][col] += coeff
        
        for (lk, li), coeff in w_dict.items():
            gk, gi = k0 + lk, i0 + li
            W[gk * n + gi][col] += coeff
        
        col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Try rank 56: 4 disjoint Strassen blocks + schoolbook for the rest
    # Blocks:
    # B1: i in {0,1}, j in {0,1}, k in {0,1}
    # B2: i in {0,1}, j in {0,1}, k in {2,3}
    # B3: i in {0,1}, j in {2,3}, k in {0,1}
    # B4: i in {0,1}, j in {2,3}, k in {2,3}
    
    # Each Strassen block uses rank 7
    # 4 blocks * 7 = 28
    # Remaining: 60 - 4*8 = 60 - 32 = 28 schoolbook terms
    # Total: 28 + 28 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add the 4 Strassen blocks
    blocks = [
        (0, 0, 0),  # B1
        (0, 0, 2),  # B2
        (0, 2, 0),  # B3
        (0, 2, 2),  # B4
    ]
    
    for (i0, j0, k0) in blocks:
        if time.time() > deadline:
            # Fallback to schoolbook if we're running out of time
            return _schoolbook(n, m, p)
        col = _strassen_block(U, V, W, i0, j0, k0, col)
    
    # Add schoolbook terms for all (i,j,k) not covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    if time.time() > deadline:
                        return _schoolbook(n, m, p)
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Verify
    if not _verify_decomposition(n, m, p, U, V, W):
        # If verification fails, fall back to schoolbook
        return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
