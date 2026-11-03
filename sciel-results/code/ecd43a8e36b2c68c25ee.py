"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    # U: (n*m, rank) -> (n, m, rank)
    U3 = U_arr.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V3 = V_arr.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # Compute the tensor sum: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Result should be 1 for all i,j,k
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Factors are (U_vec, V_vec, W_vec) where each is a 4-element vector.
    U_vec corresponds to A entries (A00, A01, A10, A11)
    V_vec corresponds to B entries (B00, B01, B10, B11)
    W_vec corresponds to C entries (C00, C01, C10, C11)
    """
    factors = []
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A10+A11) * B00 -> C10 += M2, C11 -= M2
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = A00 * (B01-B11) -> C01 += M3, C11 += M3
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = A11 * (B10-B00) -> C00 += M4, C10 += M4
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # M5 = (A00+A01) * B11 -> C01 -= M5, C11 += M5
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # M6 = (A10-A00) * (B00+B01) -> C10 += M6
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (A01-A11) * (B10+B11) -> C00 += M7
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _strassen_4x4x4_factors():
    """
    Generate the 49 rank-1 factors for 4x4 * 4x4 matrix multiplication
    using recursive Strassen.
    Returns list of (U_vec, V_vec, W_vec) where each is a 16-element vector.
    """
    s2 = _strassen_2x2_factors()
    factors = []
    
    for s in range(7):
        for r in range(7):
            u_block = s2[s][0]  # 4 elements: coeffs for A blocks
            v_block = s2[s][1]  # 4 elements: coeffs for B blocks
            w_block = s2[s][2]  # 4 elements: coeffs for C blocks
            
            u_scalar = s2[r][0]
            v_scalar = s2[r][1]
            w_scalar = s2[r][2]
            
            U_vec = [0] * 16
            V_vec = [0] * 16
            W_vec = [0] * 16
            
            # Fill U_vec
            # A is 4x4, split into 2x2 blocks.
            # Block (bi, bj) contains elements A[2*bi + ei, 2*bj + ej]
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    c_block = u_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = u_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            U_vec[i * 4 + j] += c_block * c_scalar
            
            # Fill V_vec
            # B is 4x4, split into 2x2 blocks.
            # Block (bi, bj) contains elements B[2*bi + ei, 2*bj + ej]
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    c_block = v_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = v_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            V_vec[j * 4 + k] += c_block * c_scalar
            
            # Fill W_vec
            # C is 4x4, split into 2x2 blocks.
            # Block (ci, cj) contains elements C[2*ci + ei, 2*cj + ej]
            for ci in range(2):
                for cj in range(2):
                    block_idx = ci * 2 + cj
                    c_block = w_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = w_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            i = 2 * ci + ei
                            k = 2 * cj + ej
                            W_vec[k * 4 + i] += c_block * c_scalar
            
            factors.append((U_vec, V_vec, W_vec))
    
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    Split B into two 4x4 blocks, compute A*B1 and A*B2 each with rank-49 Strassen.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the 49 factors for 4x4*4x4
    s44_factors = _strassen_4x4x4_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # (32, 98)
    
    for t in range(2):
        for f in range(49):
            col = t * 49 + f
            U_vec, V_vec, W_vec = s44_factors[f]
            
            # U: same for both subproblems (A is the same)
            for idx in range(16):
                U[idx][col] += U_vec[idx]
            
            # V: offset by 4*t for columns
            # V_vec is for B[0:4, 0:4]. 
            # For t=0, B is B[0:4, 0:4].
            # For t=1, B is B[0:4, 4:8].
            # V index is j * p + k.
            # In V_vec, index is j * 4 + k.
            for idx in range(16):
                j = idx // 4
                k = idx % 4
                V[j * p + (k + 4 * t)][col] += V_vec[idx]
            
            # W: offset by 4*t for columns (k index in W is k*n+i)
            # W_vec is for C[0:4, 0:4].
            # For t=0, C is C[0:4, 0:4].
            # For t=1, C is C[0:4, 4:8].
            # W index is k * n + i.
            # In W_vec, index is k * 4 + i.
            for idx in range(16):
                k = idx // 4
                i = idx % 4
                W[(k + 4 * t) * n + i][col] += W_vec[idx]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try Strassen-based decomposition
    try:
        result = _try_strassen_4x4_4x8()
        if time.time() < deadline:
            if _verify(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
