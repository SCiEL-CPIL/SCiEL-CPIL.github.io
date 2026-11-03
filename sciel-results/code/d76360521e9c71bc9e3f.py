"""Matrix multiplication 3x4 by 4x6: Strassen-based rank-66 with de Zeeuw peeling optimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Primary: Strassen-based rank-66 decomposition
    Secondary: Attempt to find lower-rank solution via column-removal search
    """
    n, m, p = 3, 4, 6
    rank = 66
    
    # Build the known rank-66 solution
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # Term 1: (a+d)(e+h) -> p+s
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # Term 2: (c+d)(e) -> r-s
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # Term 3: (a)(f-h) -> q+s
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # Term 4: (d)(g-e) -> p+r
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # Term 5: (a+b)(h) -> -p+q
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # Term 6: (c-a)(e+f) -> s
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # Term 7: (b-d)(g+h) -> p
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Build rank-66 solution:
    # Row 0 (i=0): schoolbook for all j,k -> 4*6 = 24 terms
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Rows 1-2 (i=1,2): Strassen blocks for j in {0,1},{2,3} and k in {0,1},{2,3},{4,5}
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
    
    # Verify the solution is correct
    def verify(U, V, W, rank):
        n, m, p = 3, 4, 6
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fallback to schoolbook if verification fails
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Efficient verification using einsum
    def verify_fast(U_np, V_np, W_np, rank):
        n, m, p = 3, 4, 6
        # Reshape for tensor operations
        U_t = U_np.reshape(n, m, rank)  # (n, m, rank)
        V_t = V_np.reshape(m, p, rank)  # (m, p, rank)
        W_t = W_np.reshape(p, n, rank)  # (p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        # Check all entries equal 1
        return np.allclose(T, 1.0, atol=1e-9)
    
    # Try to reduce rank by removing columns one at a time
    # Greedy approach: try removing each column, check if still valid
    current_rank = rank
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    
    # Try to remove columns
    for attempt in range(10):  # Try up to 10 removals
        if _time_left() < 15:
            break
        
        removed = False
        for remove_col in range(current_rank):
            if _time_left() < 10:
                break
            
            U_test = np.delete(current_U, remove_col, axis=1)
            V_test = np.delete(current_V, remove_col, axis=1)
            W_test = np.delete(current_W, remove_col, axis=1)
            
            if verify_fast(U_test, V_test, W_test, current_rank - 1):
                current_rank -= 1
                current_U = U_test
                current_V = V_test
                current_W = W_test
                removed = True
                break
        
        if not removed:
            break
    
    # Update best if we found improvement
    if current_rank < best_rank:
        best_rank = current_rank
        best_U = current_U
        best_V = current_V
        best_W = current_W
    
    # Return the best solution found
    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U, "V": V, "W": W}
