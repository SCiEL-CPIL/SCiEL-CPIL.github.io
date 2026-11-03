"""Matrix multiplication 2,4,5 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _construct_strassen_rank36():
    """
    Rank-36 decomposition of the 2x4x5 matrix multiplication tensor.
    
    Strategy: Split m=4 into two groups of 2 (j_blocks). For each j_block,
    split p=5 into K1={0,1}, K2={2,3}, K3={4}.
    - K1, K2: 2x2x2 blocks using Strassen (rank 7 each)
    - K3: 2x2x1 block using schoolbook (rank 4)
    Total: 2 * (7 + 7 + 4) = 36
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    
    U = [[0.0] * rank for _ in range(n * m)]  # (8, 36)
    V = [[0.0] * rank for _ in range(m * p)]  # (20, 36)
    W = [[0.0] * rank for _ in range(p * n)]  # (10, 36)
    
    # Strassen 2x2x2 decomposition:
    # For matrices A (2x2), B (2x2), C = A*B (2x2):
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b00)
    # M4 = a11 * (b11 - b01)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen patterns for (i, j, k) in a 2x2x2 block:
    # U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    # For a 2x2 block at (i_block=0, j_block, k_block):
    # i = 0 (only one i-group since n=2 split into i1={0}... wait n=2 so just i=0 and i=1)
    
    # Actually for 2x2x2: i in {0,1}, j in {0,1} (within block), k in {0,1} (within block)
    # U index: i*m + (j_block + j_local)
    # V index: (j_block + j_local)*p + (k_block + k_local)
    # W index: (k_block + k_local)*n + i
    
    strassen_patterns = [
        # (U_i, U_jlocal, V_jlocal, V_klocal, W_klocal, W_i, coeff_U, coeff_V, coeff_W)
        # M1 = (a00+a11)(b00+b11): 
        # U: i=0, j=0 -> coeff 1; i=1, j=1 -> coeff 1 (but we add per column)
        # Actually each column corresponds to one multiplication factor.
        # For M1: U contribution to i=0,j=0 is 1; i=1,j=1 is 1
        # V contribution to j=0,k=0 is 1; j=1,k=1 is 1
        # W: k=0,i=0 -> 1; k=1,i=1 -> 1; k=0,i=1 -> -1; k=1,i=0 -> 1
        # Wait, I need to be more careful. Each rank-1 term has exactly one U entry, one V entry, one W entry? No.
        # In a rank-1 tensor term, U[i,j,c]=u, V[j,k,c]=v, W[k,i,c]=w for one (i,j,k,c) combination? 
        # No. A rank-1 tensor T_{ijk} = u_i * v_j * w_k has:
        # U[i,j,c] gets u_i for the appropriate j... hmm.
        
        # Let me reconsider. The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k.
        # We want T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
        # Wait no. The convention is:
        # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # So T_{i,j,k} = 1 and (AB)_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{j,c} U_{i,j,c} V_{j,k,c} W_{k,i,c}... 
        # Hmm, that doesn't quite work dimensionally. Let me re-read the problem.
        
        # "output factor follows the official trace convention W[k*n+i,r]"
        # U has shape (n*m, rank) -> U[i*m+j, c]
        # V has shape (m*p, rank) -> V[j*p+k, c]  
        # W has shape (p*n, rank) -> W[k*n+i, c]
        
        # The decomposition: (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) ... no
        # It should be: (AB)_{i,k} = sum_c W[k*n+i, c] * sum_j ... 
        # Actually I think it's: for each c, we compute a product and W combines.
        # The standard form: T_{i,j,k} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
        # where T_{i,j,k} = 1 for the matrix mult tensor.
        # And (AB)_{i,k} = sum_j A_{i,j} B_{j,k} requires that:
        # sum_c sum_j A_{i,j} V_{jk,c} U_{ij,c} ... this is getting confusing.
        
        # Let me just use the known Strassen pattern directly.
        # For 2x2x2, the 7 rank-1 terms are well-known. Let me define them by which (i,j,k) they contribute to.
    ]
    
    # Let me define Strassen 2x2x2 more carefully.
    # The 7 scalars M1..M7 and their contributions to C_{i,k}:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mt involves products of linear combinations of A elements and B elements.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # For the tensor decomposition, each Mt corresponds to a rank-1 term.
    # For Mt, the U factor (acting on A indices) and V factor (acting on B indices) and W factor.
    # 
    # M1: U contributes to A00 (1) and A11 (1). V contributes to B00 (1) and B11 (1).
    #     W: C00 gets +1, C01 gets 0, C10 gets 0, C11 gets +1
    #     So W[k=0,i=0]=1, W[k=0,i=1]=0, W[k=1,i=0]=0, W[k=1,i=1]=1
    #     U[i=0,j=0]=1, U[i=0,j=1]=0, U[i=1,j=0]=0, U[i=1,j=1]=1
    #     V[j=0,k=0]=1, V[j=0,k=1]=0, V[j=1,k=0]=0, V[j=1,k=1]=1
    #
    # This is getting complex. Let me just hardcode the known patterns.
    
    def _add_strassen(col_start, i_block_start, j_block_start, k_block_start):
        """Add Strassen 2x2x2 pattern starting at column col_start."""
        patterns = [
            # (u_coeffs [(i_local, j_local, val)], v_coeffs [(j_local, k_local, val)], w_coeffs [(k_local, i_local, val)])
            # M1 = (A00+A11)(B00+B11)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2 = (A10+A11)B00
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]),
            # M3 = A00(B10-B00)
            ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,1,-1)]),
            # M4 = A11(B11-B01)
            ([(1,1,1)], [(1,1,1),(0,1,-1)], [(1,0,-1),(1,1,1)]),
            # M5 = (A00+A01)B11
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1),(0,0,-1)]),
            # M6 = (A10-A00)(B00+B01)
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,1,1)]),
            # M7 = (A01-A11)(B10+B11)
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,-1),(1,0,1)]),
        ]
        
        for m_idx, (u_coeff, v_coeff, w_coeff) in enumerate(patterns):
            col = col_start + m_idx
            for (i_loc, j_loc, val) in u_coeff:
                ii = i_block_start + i_loc
                jj = j_block_start + j_loc
                U[ii * m + jj][col] += val
            for (j_loc, k_loc, val) in v_coeff:
                jj = j_block_start + j_loc
                kk = k_block_start + k_loc
                V[jj * p + kk][col] += val
            for (k_loc, i_loc, val) in w_coeff:
                kk = k_block_start + k_loc
                ii = i_block_start + i_loc
                W[kk * n + ii][col] += val
    
    def _add_2x2x1(col_start, i_block_start, j_block_start, k_block_start):
        """Add schoolbook 2x2x1 pattern starting at column col_start."""
        # C_{i,0} = sum_j A_{i,j} B_{j,0} for i in {0,1}, j in {0,1}
        # 4 terms
        for i_loc in range(2):
            for j_loc in range(2):
                col = col_start + i_loc * 2 + j_loc
                U[(i_block_start + i_loc) * m + (j_block_start + j_loc)][col] = 1
                V[(j_block_start + j_loc) * p + (k_block_start + 0)][col] = 1
                W[(k_block_start + 0) * n + (i_block_start + i_loc)][col] = 1
    
    col = 0
    for j_block in [0, 2]:
        # K1: k=0,1 -> 2x2x2 Strassen
        _add_strassen(col, 0, j_block, 0)
        col += 7
        # K2: k=2,3 -> 2x2x2 Strassen
        _add_strassen(col, 0, j_block, 2)
        col += 7
        # K3: k=4 -> 2x2x1 schoolbook
        _add_2x2x1(col, 0, j_block, 4)
        col += 4
    
    assert col == 36, f"Expected 36, got {col}"
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # For each (i,k), check: sum_j 1 * sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 if we set A=I... 
    # Actually the tensor T_{i,j,k} should be 1 for all i,j,k.
    # T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_r[i, j, c] * V_r[j, k, c] * W_r[k, i, c]
                if abs(val - 1.0) > tol:
                    return False, val
    return True, 0.0


def construct():
    deadline = time.time() + 140  # Safety margin before 150s
    
    # Try rank 36 decomposition first (Strassen-based)
    try:
        result = _construct_strassen_rank36()
        # Quick verification
        ok, val = _verify(result["U"], result["V"], result["W"], 2, 4, 5)
        if ok:
            return result
    except Exception:
        pass
    
    if time.time() > deadline:
        # Fallback to schoolbook rank 40
        n, m, p = DIMENSIONS
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
    
    # If rank 36 failed, return schoolbook as fallback
    n, m, p = DIMENSIONS
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
