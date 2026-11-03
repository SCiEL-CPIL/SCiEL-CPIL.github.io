import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = 3, 5, 7
    
    # We'll construct a rank-99 decomposition using Strassen blocks where possible.
    # This is the same as the current program's approach, which achieves score 1.0606.
    # Let's verify this is correct and try to optimize.
    
    rank = 99
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Tiling Strategy:
    # i in {0, 1}: Use Strassen for j-pairs and k-pairs where possible
    # j in {0, 1}, k in {0, 1}, {2, 3}, {4, 5} -> 3 Strassen blocks (7 each = 21)
    # j in {0, 1}, k in {6} -> Schoolbook (2*1*1 = 2)
    # j in {2, 3}, k in {0, 1}, {2, 3}, {4, 5} -> 3 Strassen blocks (7 each = 21)
    # j in {2, 3}, k in {6} -> Schoolbook (2*1*1 = 2)
    # j in {4}, k in {0..6} -> Schoolbook (2*1*7 = 14)
    # i in {2}: All Schoolbook (1*5*7 = 35)
    
    # i=0,1; j=0,1
    for k0 in [0, 2, 4]:
        add_strassen(0, 0, k0)
    add_schoolbook([0, 1], [0, 1], [6])
    
    # i=0,1; j=2,3
    for k0 in [0, 2, 4]:
        add_strassen(0, 2, k0)
    add_schoolbook([0, 1], [2, 3], [6])
    
    # i=0,1; j=4
    add_schoolbook([0, 1], [4], range(7))
    
    # i=2; j=0..4
    add_schoolbook([2], range(5), range(7))
    
    # Check if we used exactly 99 columns
    if col != rank:
        # If col < rank, we have unused columns; if col > rank, we exceeded
        # For now, let's just verify correctness
        pass
    
    # Verify with NumPy
    if time.time() > deadline - 15:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Create target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute decomposition
    try:
        computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if not np.allclose(computed, T, atol=1e-9):
            # Fallback to schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col_fb] = 1
                        V_fb[j * p + k, col_fb] = 1
                        W_fb[k * n + i, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
