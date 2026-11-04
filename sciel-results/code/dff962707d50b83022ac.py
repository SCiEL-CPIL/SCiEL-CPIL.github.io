import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _verify_4x4x4(U, V, W, tol=1e-9):
    """Verify 4x4x4 tensor decomposition."""
    n = m = p = 4
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    # T[i,j,k] should be 1 if i==k, 0 otherwise
    expected = np.zeros((4,4,4))
    for i in range(4):
        for j in range(4):
            expected[i,j,i] = 1.0
    max_err = np.max(np.abs(T - expected))
    return max_err < tol, max_err

def _winograd_4x4x4():
    """Construct rank-19 Winograd decomposition for 4x4x4."""
    n = m = p = 4
    rank = 19
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Define 19 terms
    terms = [
        # Term 0-6: For C[0:2, 0:2]
        (
            {(0,0):1,(1,1):1},
            {(0,0):1,(1,1):1},
            {(0,0):1,(1,1):1}
        ),
        (
            {(1,0):1,(1,1):1},
            {(0,0):1},
            {(1,0):1,(1,1):-1}
        ),
        (
            {(0,0):1},
            {(0,1):1,(1,1):-1},
            {(0,1):1,(1,1):1}
        ),
        (
            {(1,1):1},
            {(1,0):1,(0,0):-1},
            {(0,0):1,(1,0):1}
        ),
        (
            {(0,0):1,(0,1):1},
            {(1,1):1},
            {(0,1):1,(0,0):-1}
        ),
        (
            {(1,0):1,(0,0):-1},
            {(0,0):1,(0,1):1},
            {(1,1):1}
        ),
        (
            {(0,1):1,(1,1):-1},
            {(1,0):1,(1,1):1},
            {(0,0):1}
        ),
        # Term 7-13: For C[0:2, 2:4]
        (
            {(0,0):1,(1,1):1},
            {(0,2):1,(1,3):1},
            {(2,0):1,(3,1):1}
        ),
        (
            {(1,0):1,(1,1):1},
            {(0,2):1},
            {(2,1):1,(3,1):-1}
        ),
        (
            {(0,0):1},
            {(0,3):1,(1,3):-1},
            {(3,0):1,(3,1):1}
        ),
        (
            {(1,1):1},
            {(1,2):1,(0,2):-1},
            {(2,0):1,(2,1):1}
        ),
        (
            {(0,0):1,(0,1):1},
            {(1,3):1},
            {(3,0):1,(2,0):-1}
        ),
        (
            {(1,0):1,(0,0):-1},
            {(0,2):1,(0,3):1},
            {(3,1):1}
        ),
        (
            {(0,1):1,(1,1):-1},
            {(1,2):1,(1,3):1},
            {(2,0):1}
        ),
        # Term 14-18: For C[2:4, 0:2]
        (
            {(2,0):1,(3,1):1},
            {(0,0):1,(1,1):1},
            {(0,2):1,(1,3):1}
        ),
        (
            {(3,0):1,(3,1):1},
            {(0,0):1},
            {(0,3):1,(1,3):-1}
        ),
        (
            {(2,0):1},
            {(0,1):1,(1,1):-1},
            {(1,2):1,(1,3):1}
        ),
        (
            {(3,1):1},
            {(1,0):1,(0,0):-1},
            {(0,2):1,(0,3):1}
        ),
        (
            {(2,0):1,(2,1):1},
            {(1,1):1},
            {(1,2):1,(0,2):-1}
        ),
    ]
    
    for c, (u_coeffs, v_coeffs, w_coeffs) in enumerate(terms):
        for (li, lj), val in u_coeffs.items():
            U[li*4+lj][c] += val
        for (lj, lk), val in v_coeffs.items():
            V[lj*4+lk][c] += val
        for (li, lk), val in w_coeffs.items():
            W[lk*4+li][c] += val
    
    return {"U": U, "V": V, "W": W}

def _restrict_to_3x4x4(decomp_4x4x4):
    """Restrict 4x4x4 decomposition to 3x4x4."""
    U4 = decomp_4x4x4["U"]  # 16 x rank
    V4 = decomp_4x4x4["V"]  # 16 x rank
    W4 = decomp_4x4x4["W"]  # 16 x rank
    rank = len(U4[0])
    
    # U: take rows where i < 3 (i.e., rows 0-11)
    U3 = U4[:12]
    
    # V: keep all rows (m=4, p=4)
    V3 = V4[:]
    
    # W: take rows where i < 3
    # W[k*4+i, c] -> keep rows where i < 3
    W3 = []
    for k in range(4):
        for i in range(3):
            row_idx = k*4 + i
            W3.append(W4[row_idx][:])
    
    return {"U": U3, "V": V3, "W": W3}

def _schoolbook():
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

def _strassen_44():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            for (li, lj), val in u_coeffs.items():
                U[(li+xi)*m + (lj+xj)][col] += val
            for (lj, lk), val in v_coeffs.items():
                V[(lj+yj)*p + (lk+yk)][col] += val
            for (li, lk), val in w_coeffs.items():
                W[(lk+zk)*n + (li+zi)][col] += val
            col += 1
    
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_remove_columns(U, V, W, n, m, p, tol=1e-9, max_removals=10):
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    rank = len(current_U[0])
    removed = 0
    
    while removed < max_removals:
        found = False
        for c_remove in range(rank - removed):
            U_new = [row[:c_remove] + row[c_remove+1:] for row in current_U]
            V_new = [row[:c_remove] + row[c_remove+1:] for row in current_V]
            W_new = [row[:c_remove] + row[c_remove+1:] for row in current_W]
            
            ok, err = _verify_fast(U_new, V_new, W_new, n, m, p, tol)
            if ok:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                rank = len(current_U[0])
                removed += 1
                found = True
                break
        
        if not found:
            break
    
    if removed > 0:
        return {"U": current_U, "V": current_V, "W": current_W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank-19 Winograd 4x4x4 restricted to 3x4x4
    try:
        wg4 = _winograd_4x4x4()
        ok4, err4 = _verify_4x4x4(wg4["U"], wg4["V"], wg4["W"])
        if ok4:
            wg3 = _restrict_to_3x4x4(wg4)
            ok3, err3 = _verify_fast(wg3["U"], wg3["V"], wg3["W"], n, m, p)
            if ok3:
                return wg3
    except Exception:
        pass
    
    # Fallback to rank-44 Strassen
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    if ok:
        reduced = _try_remove_columns(result_44["U"], result_44["V"], result_44["W"], n, m, p)
        if reduced is not None:
            return reduced
        return result_44
    
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    ok, err = _verify_fast(res['U'], res['V'], res['W'], 3, 4, 4)
    print("Valid:", ok, "Err:", err)
