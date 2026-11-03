import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _build_rank44():
    """
    Build rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for j in range(4):
        for k in range(4):
            U[2*4+j][col] = 1
            V[j*4+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    def strassen_2x2(x, y, r):
        nonlocal col
        U[x[0]][col] += 1; U[x[3]][col] += 1
        V[y[0]][col] += 1; V[y[3]][col] += 1
        W[r[0]][col] += 1; W[r[3]][col] += 1; col += 1
        U[x[2]][col] += 1; U[x[3]][col] += 1
        V[y[0]][col] += 1
        W[r[2]][col] += 1; W[r[3]][col] += -1; col += 1
        U[x[0]][col] += 1
        V[y[2]][col] += 1; V[y[3]][col] += -1
        W[r[1]][col] += 1; W[r[3]][col] += 1; col += 1
        U[x[3]][col] += 1
        V[y[0]][col] += 1; V[y[2]][col] += -1
        W[r[0]][col] += 1; W[r[2]][col] += 1; col += 1
        U[x[0]][col] += 1; U[x[1]][col] += 1
        V[y[3]][col] += 1
        W[r[0]][col] += -1; W[r[1]][col] += 1; col += 1
        U[x[2]][col] += 1; U[x[0]][col] += -1
        V[y[0]][col] += 1; V[y[1]][col] += 1
        W[r[3]][col] += 1; col += 1
        U[x[1]][col] += 1; U[x[3]][col] += -1
        V[y[2]][col] += 1; V[y[3]][col] += 1
        W[r[0]][col] += 1; col += 1
    strassen_2x2([0,1,4,5],[0,1,4,5],[0,1,3,4])
    strassen_2x2([0,1,4,5],[2,3,6,7],[6,7,9,10])
    strassen_2x2([2,3,6,7],[8,9,12,13],[0,1,3,4])
    strassen_2x2([2,3,6,7],[10,11,14,15],[6,7,9,10])
    assert col == 44
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, max_rounds=10):
    """
    Attempt to reduce rank by checking if any column can be removed
    while preserving the tensor identity.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    for _ in range(max_rounds):
        removed = False
        for col_idx in range(Ua.shape[1]):
            U_new = np.delete(Ua, col_idx, axis=1)
            V_new = np.delete(Va, col_idx, axis=1)
            W_new = np.delete(Wa, col_idx, axis=1)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_new[i*m+j], V_new[j*p+k] * W_new[k*n+i])
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                Ua, Va, Wa = U_new, V_new, W_new
                removed = True
                break
        
        if not removed:
            break
    
    return Ua.tolist(), Va.tolist(), Wa.tolist()

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    try:
        result = _build_rank44()
        if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            raise ValueError("Verification failed")
        
        U_red, V_red, W_red = _try_reduce_rank(
            np.array(result["U"], dtype=float),
            np.array(result["V"], dtype=float),
            np.array(result["W"], dtype=float),
            n, m, p
        )
        
        if len(U_red[0]) < len(result["U"][0]):
            result = {"U": U_red, "V": V_red, "W": W_red}
        
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
        return _build_rank44()
    except Exception:
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
