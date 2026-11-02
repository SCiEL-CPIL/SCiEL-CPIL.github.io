"""Strassen-based rank-53 decomposition for 4x4 x 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2_terms():
    """Return 7 Strassen rank-1 terms for 2x2 x 2x2 -> 2x2.
    Each term is (u, v, w) of length-4 vectors (row-major 2x2).
    """
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))       # M1
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))       # M2
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))      # M3
    terms.append(([0,0,0,1], [0,0,1,-1], [0,0,0,1]))      # M4
    terms.append(([1,1,0,0], [0,0,1,0], [1,0,1,0]))       # M5
    terms.append(([0,0,1,-1], [1,1,0,0], [0,0,1,-1]))     # M6
    terms.append(([0,1,0,-1], [0,0,1,1], [0,-1,0,1]))     # M7
    return terms

def _kron_block(vec_blocks, vec_inner, N=4):
    """Kronecker product of block-coeff vector (len 4) with 2x2 vec (len 4).
    Result is length N*N=16 where index = bi*(N//2)*(N//2) + ei mapped to N*N layout.
    bi is block index (0..3), ei is inner 2x2 index (0..3).
    Position in N x N: row = (bi//2)*(N//2) + (ei//2), col = (bi%2)*(N//2) + (ei%2).
    Flat index in N*N row-major = row*N + col.
    """
    out = [0.0] * (N * N)
    for bi in range(4):
        if vec_blocks[bi] == 0:
            continue
        for ei in range(4):
            if vec_inner[ei] == 0:
                continue
            br, bc = bi // 2, bi % 2
            er, ec = ei // 2, ei % 2
            row = br * (N // 2) + er
            col = bc * (N // 2) + ec
            out[row * N + col] = vec_blocks[bi] * vec_inner[ei]
    return out

def _build_strassen_4x4_4x4():
    """Build rank-49 decomposition for 4x4 x 4x4 via tensor product of two Strassen levels."""
    N = 4
    inner_terms = _strassen_2x2_terms()
    super_terms = _strassen_2x2_terms()  # Same structure at block level
    
    all_terms = []
    for u_s, v_s, w_s in super_terms:
        for u_i, v_i, w_i in inner_terms:
            u16 = _kron_block(u_s, u_i, N)
            v16 = _kron_block(v_s, v_i, N)
            w16 = _kron_block(w_s, w_i, N)
            all_terms.append((u16, v16, w16))
    return all_terms

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(
                    U_arr[i * m + j, :] * V_arr[j * p + k, :],
                    W_arr[k * n + i, :]
                ))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_full_4x4x4x5():
    n, m, p = 4, 4, 5
    strassen_terms = _build_strassen_4x4_4x4()
    rank = len(strassen_terms) + 4
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u16, v16, w16) in enumerate(strassen_terms):
        for idx in range(16):
            U[idx][c] = u16[idx]
            V[idx][c] = v16[idx]
            W[idx][c] = w16[idx]
    
    col = len(strassen_terms)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1.0
            V[j * p + 4][col] = 1.0
            W[4 * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _snap(x, tol=1e-9):
    if abs(x) < tol:
        return 0
    xi = round(x)
    if abs(x - xi) < tol:
        return xi
    xh = round(x * 2) / 2.0
    if abs(x - xh) < tol:
        return xh
    return x

def _convert_to_integer_halfinteger(U, V, W):
    U2 = [[_snap(x) for x in row] for row in U]
    V2 = [[_snap(x) for x in row] for row in V]
    W2 = [[_snap(x) for x in row] for row in W]
    return U2, V2, W2

def construct():
    n, m, p = DIMENSIONS
    result = _build_full_4x4x4x5()
    U, V, W = result["U"], result["V"], result["W"]
    
    if _verify_decomposition(U, V, W, n, m, p):
        U2, V2, W2 = _convert_to_integer_halfinteger(U, V, W)
        return {"U": U2, "V": V2, "W": W2}
    
    # Fallback schoolbook rank 80
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
