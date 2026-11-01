"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy import linalg

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    # Initialize with the known rank 36 decomposition
    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    def verify(Ul, Vl, Wl, R):
        Ua = np.array(Ul, dtype=float)
        Va = np.array(Vl, dtype=float)
        Wa = np.array(Wl, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    def verify_fast(Ua, Va, Wa):
        """Vectorized verification"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    base_U, base_V, base_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best = (36, base_U, base_V, base_W)

    if not verify(base_U, base_V, base_W, 36):
        # Fallback: return schoolbook rank 40
        U_sch = [[0.0] * 40 for _ in range(n * m)]
        V_sch = [[0.0] * 40 for _ in range(m * p)]
        W_sch = [[0.0] * 40 for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i*m+j][c] = 1.0
                    V_sch[j*p+k][c] = 1.0
                    W_sch[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_sch, "V": V_sch, "W": W_sch}

    # Convert to numpy for efficient operations
    U_np = np.array(base_U, dtype=float)
    V_np = np.array(base_V, dtype=float)
    W_np = np.array(base_W, dtype=float)

    # Try to reduce rank by finding linear dependencies
    # A column c can be eliminated if its rank-1 tensor is in the span of others
    # i.e., if there exist alpha, beta, gamma such that:
    # U[:,c] = alpha * U[:,c1] for some c1, OR more generally,
    # the tensor T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c] is in the span of {T_r : r != c}
    
    # Better approach: try to express column c as a linear combination
    # For each pair (c, c1), check if we can eliminate c by adjusting c1
    
    def try_eliminate_column(c, c1):
        """Try to eliminate column c by adding it to column c1"""
        if c == c1:
            return None
        
        U_new = U_np.copy()
        V_new = V_np.copy()
        W_new = W_np.copy()
        
        # Add column c to column c1
        U_new[:, c1] += U_new[:, c]
        V_new[:, c1] += V_new[:, c]
        W_new[:, c1] += W_new[:, c]
        
        # Remove column c
        U_new = np.delete(U_new, c, axis=1)
        V_new = np.delete(V_new, c, axis=1)
        W_new = np.delete(W_new, c, axis=1)
        
        if verify_fast(U_new, V_new, W_new):
            return (U_new, V_new, W_new)
        
        # Try subtracting
        U_new = U_np.copy()
        V_new = V_np.copy()
        W_new = W_np.copy()
        
        U_new[:, c1] -= U_new[:, c]
        V_new[:, c1] -= V_new[:, c]
        W_new[:, c1] -= W_new[:, c]
        
        U_new = np.delete(U_new, c, axis=1)
        V_new = np.delete(V_new, c, axis=1)
        W_new = np.delete(W_new, c, axis=1)
        
        if verify_fast(U_new, V_new, W_new):
            return (U_new, V_new, W_new)
        
        return None

    # Systematic search: try all pairs
    found = False
    current_rank = 36
    current_U, current_V, current_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try eliminating one column at a time, greedily
    while True:
        if time.time() > deadline - 15:
            break
        
        eliminated = False
        for c in range(current_rank):
            if time.time() > deadline - 15:
                break
            for c1 in range(current_rank):
                if c == c1:
                    continue
                result = try_eliminate_column(c, c1)
                if result is not None:
                    current_U, current_V, current_W = result
                    current_rank -= 1
                    eliminated = True
                    found = True
                    break
            if eliminated:
                break
        
        if not eliminated:
            break
    
    if found:
        best = (current_rank, 
                current_U.tolist(), 
                current_V.tolist(), 
                current_W.tolist())
    else:
        best = (36, base_U, base_V, base_W)

    rank, bU, bV, bW = best
    return {
        "U": bU,
        "V": bV,
        "W": bW
    }
# EVOLVE-BLOCK-END
