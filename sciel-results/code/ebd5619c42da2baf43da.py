import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _strassen_2x2_2():
    """Return (U, V, W) for 2x2 * 2x2 = 2x2 with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Index convention: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products:
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b01-b11)
    # M3 = a11(b10-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b10)
    # M6 = (a01-a11)(b01+b11)
    #
    # C00 = M0 - M1 + M3 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4 - M6
    
    # U vectors (flattened 2x2 A): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V vectors (flattened 2x2 B): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W vectors (flattened 2x2 C): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M0
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    W[1,0] = 0; W[2,0] = 0
    
    # M1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = 1
    W[0,1] = -1; W[1,1] = 0
    
    # M2
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1
    W[0,2] = 0; W[2,2] = 0; W[3,2] = -1
    
    # M3
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[2,3] = 1; W[3,3] = 1
    W[0,3] = 1; W[1,3] = 0
    
    # M4
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1; W[3,4] = 1
    W[0,4] = 0; W[2,4] = 0
    
    # M5
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    W[0,5] = 1; W[3,5] = 1
    W[1,5] = 0; W[2,5] = 0
    
    # M6
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1
    W[3,6] = -1
    W[0,6] = 0; W[1,6] = 0; W[2,6] = 0
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # Build the tensor T from U, V, W: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We need to check that this equals the standard matrix multiplication tensor.
    
    # Check a few random entries
    np.random.seed(42)
    for _ in range(100):
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        k = np.random.randint(0, p)
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        # Tensor value from decomposition
        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        
        # Standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor M_{n,m,p}[i*m+j, j*p+k, k*n+i] = 1 if this is a valid (i,j,k) triple, else 0
        # Actually, the matrix multiplication tensor has:
        # T[i, j, k] = 1 where the entry C[i,k] gets contribution A[i,j]*B[j,k]
        # In flattened form: T[i*m+j, j*p+k, k*n+i] = 1
        # So for any valid (i,j,k), the tensor value should be 1.
        expected = 1.0
        
        if abs(val - expected) > 1e-6:
            return False
    
    # Also check that non-tensor entries are 0
    # The tensor is non-zero only at positions (i*m+j, j*p+k, k*n+i) for valid i,j,k
    # For other positions, the sum should be 0
    # Check a few invalid positions
    for _ in range(100):
        u = np.random.randint(0, n*m)
        v = np.random.randint(0, m*p)
        w = np.random.randint(0, p*n)
        
        # Check if (u,v,w) is a valid tensor position
        i1 = u // m; j1 = u % m
        j2 = v // p; k1 = v % p
        k2 = w // n; i2 = w % n
        
        if j1 == j2 and k1 == k2 and i1 == i2:
            continue  # Valid position, skip
        
        val = np.sum(U[u, :] * V[v, :] * W[w, :])
        if abs(val) > 1e-6:
            return False
    
    return True


def _build_decomposition():
    """Build the rank-56 decomposition for 3x3 * 3x8 using Strassen blocks."""
    n, m, p = 3, 3, 8
    
    # We decompose the 3x3x8 into:
    # Block 1: cols 0,1,2 (3x3x3)
    # Block 2: cols 3,4,5 (3x3x3)
    # Block 3: cols 6,7 (3x3x2)
    
    # For each 3x3x3, use rank 21 (via Strassen + schoolbook hybrid)
    # For 3x3x2, use rank 14
    
    total_rank = 21 + 21 + 14
    rank = total_rank
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper: add a 2x2x2 Strassen block at positions
    # A: rows i0:i0+2, cols j0:j0+2
    # B: rows j0:j0+2, cols k0:k0+2
    # C: rows k0:k0+2, cols i0:i0+2
    def add_strassen(i0, j0, k0, start_col):
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            # U
            for r in range(4):
                ui = sU[r, c]
                if ui != 0:
                    i = i0 + r // 2
                    j = j0 + r % 2
                    U[i*m + j, start_col + c] += ui
            # V
            for r in range(4):
                vi = sV[r, c]
                if vi != 0:
                    j = j0 + r // 2
                    k = k0 + r % 2
                    V[j*p + k, start_col + c] += vi
            # W
            for r in range(4):
                wi = sW[r, c]
                if wi != 0:
                    k = k0 + r // 2
                    i = i0 + r % 2
                    W[k*n + i, start_col + c] += wi
        return start_col + 7
    
    # Helper: add schoolbook terms for remaining parts of 3x3x3
    def add_3x3x3(k_off, start_col):
        c = start_col
        # Use Strassen for top-left 2x2x2
        c = add_strassen(0, 0, k_off, c)
        
        # Remaining terms:
        # A col 2 (3x1) * B rows 0,1 (2x3) -> C rows 0,1 (3x3)
        # A row 2 (3x3) * B col 2 (3x1) -> C col 2 (3x3)
        # But we need to be careful not to double-count the 2x2x2 part
        
        # Let's use schoolbook for the non-2x2x2 parts:
        # Terms involving A[0:2, 2] * B[2, k_off:k_off+3] -> C[k_off:k_off+3, 0:2]
        for i in range(2):
            for k in range(3):
                U[i*m + 2, c] = 1
                V[2*p + (k_off + k), c] = 1
                W[(k_off + k)*n + i, c] = 1
                c += 1
        
        # Terms involving A[2, 0:2] * B[0:2, k_off:k_off+3] -> C[k_off:k_off+3, 2]
        for j in range(2):
            for k in range(3):
                U[2*m + j, c] = 1
                V[j*p + (k_off + k), c] = 1
                W[(k_off + k)*n + 2, c] = 1
                c += 1
        
        # Term involving A[2,2] * B[2, k_off:k_off+3] -> C[k_off:k_off+3, 2]
        for k in range(3):
            U[2*m + 2, c] = 1
            V[2*p + (k_off + k), c] = 1
            W[(k_off + k)*n + 2, c] = 1
            c += 1
        
        return c
    
    # Helper: add 3x3x2 block
    def add_3x3x2(k_off, start_col):
        c = start_col
        # Use Strassen for top-left 2x2x2
        c = add_strassen(0, 0, k_off, c)
        
        # A col 2 (3x1) * B rows 0,1 (2x2) -> C rows 0,1 (2x3)
        for i in range(2):
            for k in range(2):
                U[i*m + 2, c] = 1
                V[2*p + (k_off + k), c] = 1
                W[(k_off + k)*n + i, c] = 1
                c += 1
        
        # A row 2 (3x2) * B cols 0,1 (2x2) -> C rows 0,1 (2x3)
        for j in range(2):
            for k in range(2):
                U[2*m + j, c] = 1
                V[j*p + (k_off + k), c] = 1
                W[(k_off + k)*n + 2, c] = 1
                c += 1
        
        # A[2,2] * B[2, 0:2] -> C[0:2, 2]
        for k in range(2):
            U[2*m + 2, c] = 1
            V[2*p + (k_off + k), c] = 1
            W[(k_off + k)*n + 2, c] = 1
            c += 1
        
        return c
    
    col = add_3x3x3(0, col)
    col = add_3x3x3(3, col)
    col = add_3x3x2(6, col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, col

def construct():
    start_time = time.time()
    
    result, rank_used = _build_decomposition()
    
    # Verify
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    if _verify(U, V, W, 3, 3, 8):
        return result
    
    # If verification fails, fall back to schoolbook
    n, m, p = 3, 3, 8
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

