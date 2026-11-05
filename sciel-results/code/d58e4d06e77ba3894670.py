"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Helper to get U index for A[i, j]
    def u_idx(i, j):
        return i * m + j

    # Helper to get V index for B[j, k]
    def v_idx(j, k):
        return j * p + k

    # Helper to get W index for C[i, k] (stored as k*n+i)
    def w_idx(i, k):
        return k * n + i

    # 1. A11 B11 (2x2x2) -> Rank 7
    # A11: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C11: i in {0,1}, k in {0,1}
    
    # Strassen 2x2x2
    # U: A00, A01, A10, A11
    # V: B00, B01, B10, B11
    # W: C00, C01, C10, C11 (indices k*n+i)
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    add_term(u_idx(0,0), v_idx(0,0), w_idx(0,0), 1, 1, 1)
    add_term(u_idx(1,1), v_idx(1,1), w_idx(0,0), 1, 1, 0) # Wait, add_term increments col. 
    # This approach is wrong. Strassen uses the SAME column for multiple U/V/W entries?
    # No, each rank-1 term is one column.
    # But U, V, W are vectors.
    # So for one term, we set multiple entries in U, V, W for that column.
    
    # Let's redefine add_term to not increment col, but set values.
    # And a separate function to advance col.
    
    # Reset
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def set_term(u_indices_vals, v_indices_vals, w_indices_vals):
        nonlocal col
        for idx, val in u_indices_vals:
            U[idx][col] += val
        for idx, val in v_indices_vals:
            V[idx][col] += val
        for idx, val in w_indices_vals:
            W[idx][col] += val
        col += 1

    # Strassen 2x2x2 for A11 B11
    # Local indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> w_idx(i,k)
    
    # M1
    set_term(
        [(u_idx(0,0), 1), (u_idx(1,1), 1)],
        [(v_idx(0,0), 1), (v_idx(1,1), 1)],
        [(w_idx(0,0), 1), (w_idx(1,1), 1)]
    )
    # M2
    set_term(
        [(u_idx(1,0), 1), (u_idx(1,1), 1)],
        [(v_idx(0,0), 1)],
        [(w_idx(1,0), 1), (w_idx(1,1), -1)]
    )
    # M3
    set_term(
        [(u_idx(0,0), 1)],
        [(v_idx(0,1), 1), (v_idx(1,1), -1)],
        [(w_idx(0,1), 1), (w_idx(1,1), 1)]
    )
    # M4
    set_term(
        [(u_idx(1,1), 1)],
        [(v_idx(1,0), 1), (v_idx(0,0), -1)],
        [(w_idx(0,0), 1), (w_idx(1,0), 1)]
    )
    # M5
    set_term(
        [(u_idx(0,0), 1), (u_idx(0,1), 1)],
        [(v_idx(1,1), 1)],
        [(w_idx(0,0), -1), (w_idx(0,1), 1)]
    )
    # M6
    set_term(
        [(u_idx(1,0), 1), (u_idx(0,0), -1)],
        [(v_idx(0,0), 1), (v_idx(0,1), 1)],
        [(w_idx(1,1), 1)]
    )
    # M7
    set_term(
        [(u_idx(0,1), 1), (u_idx(1,1), -1)],
        [(v_idx(1,0), 1), (v_idx(1,1), 1)],
        [(w_idx(0,0), 1)]
    )

    # 2. A12 B21 (2x2x2) -> Rank 7
    # A12: i in {0,1}, j in {2,3}
    # B21: j in {2,3}, k in {0,1}
    # C12? No, A12 B21 contributes to C11?
    # A is 3x4, B is 4x5.
    # C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # A12 (rows 0,1, cols 2,3) * B21 (rows 2,3, cols 0,1).
    # Contributes to C[i,k] for i in {0,1}, k in {0,1}.
    # So it adds to the C11 block (i in 0,1, k in 0,1).
    
    # Map local 0->2, 1->3 for A cols and B rows.
    def u_idx_12(i, j_local):
        return u_idx(i, j_local + 2)
    def v_idx_21(j_local, k):
        return v_idx(j_local + 2, k)
    # W is same as C11 block: w_idx(i,k)
    
    # M1
    set_term(
        [(u_idx_12(0,0), 1), (u_idx_12(1,1), 1)],
        [(v_idx_21(0,0), 1), (v_idx_21(1,1), 1)],
        [(w_idx(0,0), 1), (w_idx(1,1), 1)]
    )
    # M2
    set_term(
        [(u_idx_12(1,0), 1), (u_idx_12(1,1), 1)],
        [(v_idx_21(0,0), 1)],
        [(w_idx(1,0), 1), (w_idx(1,1), -1)]
    )
    # M3
    set_term(
        [(u_idx_12(0,0), 1)],
        [(v_idx_21(0,1), 1), (v_idx_21(1,1), -1)],
        [(w_idx(0,1), 1), (w_idx(1,1), 1)]
    )
    # M4
    set_term(
        [(u_idx_12(1,1), 1)],
        [(v_idx_21(1,0), 1), (v_idx_21(0,0), -1)],
        [(w_idx(0,0), 1), (w_idx(1,0), 1)]
    )
    # M5
    set_term(
        [(u_idx_12(0,0), 1), (u_idx_12(0,1), 1)],
        [(v_idx_21(1,1), 1)],
        [(w_idx(0,0), -1), (w_idx(0,1), 1)]
    )
    # M6
    set_term(
        [(u_idx_12(1,0), 1), (u_idx_12(0,0), -1)],
        [(v_idx_21(0,0), 1), (v_idx_21(0,1), 1)],
        [(w_idx(1,1), 1)]
    )
    # M7
    set_term(
        [(u_idx_12(0,1), 1), (u_idx_12(1,1), -1)],
        [(v_idx_21(1,0), 1), (v_idx_21(1,1), 1)],
        [(w_idx(0,0), 1)]
    )

    # 3. A11 B12 (2x2x3) -> Rank 11
    # A11: i in {0,1}, j in {0,1}
    # B12: j in {0,1}, k in {2,3,4}
    # C12: i in {0,1}, k in {2,3,4}
    
    # Split B12 into B12_L (k=2,3) and B12_R (k=4)
    # B12_L is 2x2. B12_R is 2x1.
    
    # Strassen on A11 * B12_L
    def v_idx_12L(j, k_local):
        return v_idx(j, k_local + 2)
    def w_idx_12L(i, k_local):
        return w_idx(i, k_local + 2)
        
    # M1
    set_term(
        [(u_idx(0,0), 1), (u_idx(1,1), 1)],
        [(v_idx_12L(0,0), 1), (v_idx_12L(1,1), 1)],
        [(w_idx_12L(0,0), 1), (w_idx_12L(1,1), 1)]
    )
    # M2
    set_term(
        [(u_idx(1,0), 1), (u_idx(1,1), 1)],
        [(v_idx_12L(0,0), 1)],
        [(w_idx_12L(1,0), 1), (w_idx_12L(1,1), -1)]
    )
    # M3
    set_term(
        [(u_idx(0,0), 1)],
        [(v_idx_12L(0,1), 1), (v_idx_12L(1,1), -1)],
        [(w_idx_12L(0,1), 1), (w_idx_12L(1,1), 1)]
    )
    # M4
    set_term(
        [(u_idx(1,1), 1)],
        [(v_idx_12L(1,0), 1), (v_idx_12L(0,0), -1)],
        [(w_idx_12L(0,0), 1), (w_idx_12L(1,0), 1)]
    )
    # M5
    set_term(
        [(u_idx(0,0), 1), (u_idx(0,1), 1)],
        [(v_idx_12L(1,1), 1)],
        [(w_idx_12L(0,0), -1), (w_idx_12L(0,1), 1)]
    )
    # M6
    set_term(
        [(u_idx(1,0), 1), (u_idx(0,0), -1)],
        [(v_idx_12L(0,0), 1), (v_idx_12L(0,1), 1)],
        [(w_idx_12L(1,1), 1)]
    )
    # M7
    set_term(
        [(u_idx(0,1), 1), (u_idx(1,1), -1)],
        [(v_idx_12L(1,0), 1), (v_idx_12L(1,1), 1)],
        [(w_idx_12L(0,0), 1)]
    )
    
    # Standard on A11 * B12_R (2x2x1)
    # B12_R: j in {0,1}, k=4.
    # C12_R: i in {0,1}, k=4.
    def v_idx_12R(j):
        return v_idx(j, 4)
    def w_idx_12R(i):
        return w_idx(i, 4)
        
    # C_i0 = A_i0 B_00 + A_i1 B_10
    # Here B is col vector.
    set_term([(u_idx(0,0), 1)], [(v_idx_12R(0), 1)], [(w_idx_12R(0), 1)])
    set_term([(u_idx(0,1), 1)], [(v_idx_12R(1), 1)], [(w_idx_12R(0), 1)])
    set_term([(u_idx(1,0), 1)], [(v_idx_12R(0), 1)], [(w_idx_12R(1), 1)])
    set_term([(u_idx(1,1), 1)], [(v_idx_12R(1), 1)], [(w_idx_12R(1), 1)])

    # 4. A12 B22 (2x2x3) -> Rank 11
    # A12: i in {0,1}, j in {2,3}
    # B22: j in {2,3}, k in {2,3,4}
    # C22: i in {0,1}, k in {2,3,4}
    
    def u_idx_12_2(i, j_local):
        return u_idx(i, j_local + 2)
    def v_idx_22(j_local, k_local):
        return v_idx(j_local + 2, k_local + 2)
    def w_idx_22(i, k_local):
        return w_idx(i, k_local + 2)
        
    # Strassen on A12 * B22_L (k=2,3)
    # M1
    set_term(
        [(u_idx_12_2(0,0), 1), (u_idx_12_2(1,1), 1)],
        [(v_idx_22(0,0), 1), (v_idx_22(1,1), 1)],
        [(w_idx_22(0,0), 1), (w_idx_22(1,1), 1)]
    )
    # M2
    set_term(
        [(u_idx_12_2(1,0), 1), (u_idx_12_2(1,1), 1)],
        [(v_idx_22(0,0), 1)],
        [(w_idx_22(1,0), 1), (w_idx_22(1,1), -1)]
    )
    # M3
    set_term(
        [(u_idx_12_2(0,0), 1)],
        [(v_idx_22(0,1), 1), (v_idx_22(1,1), -1)],
        [(w_idx_22(0,1), 1), (w_idx_22(1,1), 1)]
    )
    # M4
    set_term(
        [(u_idx_12_2(1,1), 1)],
        [(v_idx_22(1,0), 1), (v_idx_22(0,0), -1)],
        [(w_idx_22(0,0), 1), (w_idx_22(1,0), 1)]
    )
    # M5
    set_term(
        [(u_idx_12_2(0,0), 1), (u_idx_12_2(0,1), 1)],
        [(v_idx_22(1,1), 1)],
        [(w_idx_22(0,0), -1), (w_idx_22(0,1), 1)]
    )
    # M6
    set_term(
        [(u_idx_12_2(1,0), 1), (u_idx_12_2(0,0), -1)],
        [(v_idx_22(0,0), 1), (v_idx_22(0,1), 1)],
        [(w_idx_22(1,1), 1)]
    )
    # M7
    set_term(
        [(u_idx_12_2(0,1), 1), (u_idx_12_2(1,1), -1)],
        [(v_idx_22(1,0), 1), (v_idx_22(1,1), 1)],
        [(w_idx_22(0,0), 1)]
    )
    
    # Standard on A12 * B22_R (k=4)
    def v_idx_22R(j_local):
        return v_idx(j_local + 2, 4)
    def w_idx_22R(i):
        return w_idx(i, 4)
        
    set_term([(u_idx_12_2(0,0), 1)], [(v_idx_22R(0), 1)], [(w_idx_22R(0), 1)])
    set_term([(u_idx_12_2(0,1), 1)], [(v_idx_22R(1), 1)], [(w_idx_22R(0), 1)])
    set_term([(u_idx_12_2(1,0), 1)], [(v_idx_22R(0), 1)], [(w_idx_22R(1), 1)])
    set_term([(u_idx_12_2(1,1), 1)], [(v_idx_22R(1), 1)], [(w_idx_22R(1), 1)])

    # 5. A21 B11 (1x2x2) -> Rank 4
    # A21: i=2, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C21: i=2, k in {0,1}
    
    def u_idx_21(j):
        return u_idx(2, j)
    def w_idx_21(k):
        return w_idx(2, k)
        
    set_term([(u_idx_21(0), 1)], [(v_idx(0,0), 1)], [(w_idx_21(0), 1)])
    set_term([(u_idx_21(1), 1)], [(v_idx(1,0), 1)], [(w_idx_21(0), 1)])
    set_term([(u_idx_21(0), 1)], [(v_idx(0,1), 1)], [(w_idx_21(1), 1)])
    set_term([(u_idx_21(1), 1)], [(v_idx(1,1), 1)], [(w_idx_21(1), 1)])

    # 6. A22 B21 (1x2x2) -> Rank 4
    # A22: i=2, j in {2,3}
    # B21: j in {2,3}, k in {0,1}
    # C21: i=2, k in {0,1}
    
    def u_idx_22(j_local):
        return u_idx(2, j_local + 2)
    def v_idx_21_2(j_local, k):
        return v_idx(j_local + 2, k)
        
    set_term([(u_idx_22(0), 1)], [(v_idx_21_2(0,0), 1)], [(w_idx_21(0), 1)])
    set_term([(u_idx_22(1), 1)], [(v_idx_21_2(1,0), 1)], [(w_idx_21(0), 1)])
    set_term([(u_idx_22(0), 1)], [(v_idx_21_2(0,1), 1)], [(w_idx_21(1), 1)])
    set_term([(u_idx_22(1), 1)], [(v_idx_21_2(1,1), 1)], [(w_idx_21(1), 1)])

    # 7. A21 B12 (1x2x3) -> Rank 6
    # A21: i=2, j in {0,1}
    # B12: j in {0,1}, k in {2,3,4}
    # C22: i=2, k in {2,3,4}
    
    def w_idx_22_2(k_local):
        return w_idx(2, k_local + 2)
        
    for j in range(2):
        for k_local in range(3):
            set_term(
                [(u_idx_21(j), 1)],
                [(v_idx(j, k_local + 2), 1)],
                [(w_idx_22_2(k_local), 1)]
            )

    # 8. A22 B22 (1x2x3) -> Rank 6
    # A22: i=2, j in {2,3}
    # B22: j in {2,3}, k in {2,3,4}
    # C22: i=2, k in {2,3,4}
    
    for j_local in range(2):
        for k_local in range(3):
            set_term(
                [(u_idx_22(j_local), 1)],
                [(v_idx_22(j_local, k_local), 1)],
                [(w_idx_22_2(k_local), 1)]
            )

    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
