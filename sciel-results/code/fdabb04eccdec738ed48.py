import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 8]
_DEADLINE = time.time() + 140  # 140s safety margin before 150s kill

def _build_strassen56():
    """
    Rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    The 2x4 * 4x8 product is partitioned into 8 independent 2x2 * 2x2
    sub-problems by splitting A's 4 columns into 2 groups of 2 and B's
    8 columns into 4 groups of 2. Each sub-problem uses the rank-7
    Strassen decomposition, giving total rank 8 * 7 = 56.
    
    Output convention: W[k*n + i, r] where k is column index, i is row index.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for s in range(2):  # s=0,1: split A columns / B rows into 2x2 blocks
        for j in range(4):  # j=0..3: split B columns into 2x2 blocks
            # A_s: rows 0-1, cols 2s, 2s+1 of A (2x4)
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            # B_{s,j}: rows 2s, 2s+1; cols 2j, 2j+1 of B (4x8)
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            # C block: rows 0-1, cols 2j, 2j+1; global index = col*n + row
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # Strassen U (A-side) coefficients
            # m1 = (a00+a11)*(b00+b11)
            U[a00][col] += 1
            U[a11][col] += 1
            # m2 = (a10+a11)*b00
            U[a10][col+1] += 1
            U[a11][col+1] += 1
            # m3 = a00*(b01-b11)
            U[a00][col+2] += 1
            # m4 = a11*b10
            U[a11][col+3] += 1
            # m5 = (a00+a01)*b11
            U[a00][col+4] += 1
            U[a01][col+4] += 1
            # m6 = (a10-a00)*(b00+b01)
            U[a10][col+5] += 1
            U[a00][col+5] += -1
            # m7 = (a01-a11)*(b10+b11)
            U[a01][col+6] += 1
            U[a11][col+6] += -1
            
            # Strassen V (B-side) coefficients
            V[b00][col] += 1
            V[b11][col] += 1
            V[b00][col+1] += 1
            V[b01][col+2] += 1
            V[b11][col+2] += -1
            V[b10][col+3] += 1
            V[b00][col+3] += -1
            V[b11][col+4] += 1
            V[b00][col+5] += 1
            V[b01][col+5] += 1
            V[b10][col+6] += 1
            V[b11][col+6] += 1
            
            # Strassen W (C-side) coefficients
            # c00 = m1 + m4 - m5 + m7
            W[c00][col] += 1
            W[c00][col+3] += 1
            W[c00][col+4] += -1
            W[c00][col+6] += 1
            # c01 = m3 + m5
            W[c01][col+2] += 1
            W[c01][col+4] += 1
            # c10 = m2 + m4
            W[c10][col+1] += 1
            W[c10][col+3] += 1
            # c11 = m1 - m2 + m3 + m6
            W[c11][col] += 1
            W[c11][col+1] += -1
            W[c11][col+2] += 1
            W[c11][col+5] += 1
            
            col += 7
    
    return {"U": U, "V": V, "W": W}


def _validate_decomp(U, V, W, n=2, m=4, p=8):
    """
    Validate that the decomposition exactly computes matrix multiplication.
    U: (n*m) x rank
    V: (m*p) x rank  
    W: (p*n) x rank, indexed as W[k*n + i, r]
    
    For each rank term r, the contribution to C[i,j] is:
    sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * V[k*p+j, r])
    
    This must equal the standard matrix product.
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each rank term, compute the outer product contribution
    # U[i*m+k, r] * V[k*p+j, r] gives contribution to C[i,j]
    # Then multiply by W[j*n+i, r]
    
    # Build the full multiplication tensor check
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Decomposition: C[i,j] = sum_r W[j*n+i, r] * sum_k U[i*m+k, r] * V[k*p+j, r]
    
    # Check against random test matrices
    np.random.seed(42)
    for trial in range(3):
        A_test = np.random.randint(-5, 5, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 5, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # sum_k U[i*m+k, r] * A[i,k]
            # Actually: the intermediate product for rank term r is
            # (A @ U[:,r]) * (V[:,r] @ B) but that's not quite right.
            # 
            # Let me think again. The tensor decomposition means:
            # T_{i,j,k} = sum_r U_{i,k,r} * V_{k,j,r} * W_{i,j,r}
            # where T_{i,j,k} is the coefficient such that
            # C_{i,j} = sum_k A_{i,k} * B_{k,j}
            #
            # Wait, that's not standard. Let me reconsider.
            #
            # Standard: C_{i,j} = sum_k A_{i,k} B_{k,j}
            # This is a trilinear form in A, B, C.
            # T_{i,k,j} = delta (it's just the identity in k for each i,j)
            #
            # The decomposition says:
            # sum_k A_{i,k} B_{k,j} = sum_r (sum_k U_{i,k,r} A_{i,k}) * (sum_k V_{k,j,r} B_{k,j}) * W_{i,j,r}
            #
            # Wait no. Let me re-read the interface.
            # U has shape (n*m, rank) - indexed by [i*m+k, r]
            # V has shape (m*p, rank) - indexed by [k*p+j, r]
            # W has shape (p*n, rank) - indexed by [j*n+i, r]
            #
            # So the decomposition is:
            # C_{i,j} = sum_r W_{j*n+i, r} * (sum_k U_{i*m+k, r} * A_{i,k}) * (sum_k V_{k*p+j, r} * B_{k,j})
            
            a_proj = np.zeros(n)
            b_proj = np.zeros(p)
            for i in range(n):
                for k in range(m):
                    a_proj[i] += U_arr[i*m+k, r] * A_test[i, k]
            for k in range(m):
                for j in range(p):
                    b_proj[j] += V_arr[k*p+j, r] * B_test[k, j]
            
            for i in range(n):
                for j in range(p):
                    C_computed[i, j] += W_arr[j*n+i, r] * a_proj[i] * b_proj[j]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    
    return True


def _try_reduce_rank(base_decomp, target_rank, max_time=60):
    """
    Try to reduce the rank of the decomposition from 56 to target_rank.
    Uses a simple approach: try to merge pairs of rank terms.
    """
    start_time = time.time()
    U = [row[:] for row in base_decomp["U"]]
    V = [row[:] for row in base_decomp["V"]]
    W = [row[:] for row in base_decomp["W"]]
    rank = len(U[0])
    
    if rank <= target_rank:
        return base_decomp
    
    # Try to find pairs of rank terms that can be merged
    # Two rank terms r1 and r2 can be merged if there exists a scalar c
    # such that U[:,r1] = c*U[:,r2], V[:,r1] = c*V[:,r2], W[:,r1] = c*W[:,r2]
    # Or more generally, we can add a linear combination.
    
    # Simpler approach: try removing each rank term and checking if
    # the remaining terms can be adjusted to still work.
    # This is complex, so let's try a different approach.
    
    # Try: for each pair (i,j) of rank terms, see if we can express
    # term i as a linear combination of other terms.
    
    # Actually, let's try a more practical approach:
    # Check if any rank term is a linear combination of others.
    
    for r_remove in range(rank):
        if time.time() - start_time > max_time:
            break
        if rank - 1 < target_rank:
            break
            
        # Try removing rank term r_remove
        U_new = [row[:r_remove] + row[r_remove+1:] for row in U]
        V_new = [row[:r_remove] + row[r_remove+1:] for row in V]
        W_new = [row[:r_remove] + row[r_remove+1:] for row in W]
        
        # Check if this still works (it won't in general, but let's verify)
        # This is expensive, so only do it if we're close to target
        if rank - 1 <= target_rank + 2:
            if _validate_decomp(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                continue
    
    if rank == target_rank:
        return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Starts with rank-56 Strassen-based decomposition and attempts to reduce rank.
    """
    base = _build_strassen56()
    
    # Try to reduce rank from 56 to 55
    if time.time() < _DEADLINE - 30:
        reduced = _try_reduce_rank(base, 55, max_time=60)
        if reduced is not None and _validate_decomp(reduced["U"], reduced["V"], reduced["W"]):
            return reduced
    
    # Fall back to the known rank-56 solution
    return base
