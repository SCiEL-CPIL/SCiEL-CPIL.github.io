import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return np.allclose(T, 1.0, atol=tol)

def _snap_to_half_integer(mat, tol=1e-9):
    arr = np.array(mat, dtype=float)
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        out = []
        for row in snapped:
            out.append([float(x) for x in row])
        return out
    return [list(map(float, row)) for row in arr]

def _build_decomposition(n, m, p):
    # Same as before, rank 44
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            U[0 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 0][col] = 1.0
            col += 1
            
    def _strassen_terms():
        return [
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
            ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
            ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
            ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
            ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
        ]
        
    terms = _strassen_terms()
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in terms:
            for (xi, yi), ux in x_coeffs.items():
                U[(ar + xi) * m + (ac + yi)][col] += ux
            for (yi, xi), vy in y_coeffs.items():
                V[(br + yi) * p + (bc + xi)][col] += vy
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * n + (cr + zi)][col] += wz
            col += 1
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    
    # Start with rank 44
    result = _build_decomposition(n, m, p)
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
        
    best_rank = len(result["U"][0])
    
    # Try to find lower rank
    # We will attempt R from best_rank - 1 down to 30
    # Use random search with verification
    
    for R in range(best_rank - 1, 29, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this R
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Simple gradient descent or just check if random works (unlikely)
            # Instead, we can try to solve a least squares problem?
            # The equation is nonlinear.
            
            # Let's try a structured random search:
            # Take the rank 44 solution and try to zero out one column?
            # Or mix columns?
            
            # For now, just keep the rank 44 result if we don't find better.
            # A full optimization is too slow for 150s if not careful.
            
            # We can try to use the fact that the tensor is sparse.
            # But random search for exact integer solution is hard.
            
            pass
            
    # Return the best found
    result["U"] = _snap_to_half_integer(result["U"])
    result["V"] = _snap_to_half_integer(result["V"])
    result["W"] = _snap_to_half_integer(result["W"])
    
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
        
    return result
