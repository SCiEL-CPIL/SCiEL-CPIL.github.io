"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Block decomposition:
    # A (4x5) = [A1 (4x4), A2 (4x1)]
    # B (5x5) = [[B11 (4x4), B12 (4x1)], [B21 (1x4), B22 (1x1)]]
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
    
    # Ranks:
    # A1 B11: 4x4x4 -> Rank 64 (Schoolbook)
    # A1 B12: 4x4x1 -> Rank 4
    # A2 B21: 4x1x4 -> Rank 1
    # A2 B22: 4x1x1 -> Rank 1
    # Total Rank = 70
    
    R1 = 64
    R2 = 4
    R3 = 1
    R4 = 1
    R_total = R1 + R2 + R3 + R4
    
    U = np.zeros((n * m, R_total), dtype=np.float64)
    V = np.zeros((m * p, R_total), dtype=np.float64)
    W = np.zeros((p * n, R_total), dtype=np.float64)
    
    col = 0
    
    # Part 1: A1 B11 (4x4x4)
    # i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Part 2: A1 B12 (4x4x1)
    # i in 0..3, j in 0..3, k=4
    # Rank 4 decomposition
    for r in range(4):
        for i in range(4):
            # U row i*m + j. We need U[i*m+r, col+r] = 1
            U[i * m + r, col + r] = 1.0
            # V row j*p + k. We need V[r*p+4, col+r] = 1
            V[r * p + 4, col + r] = 1.0
            # W row k*n + i. We need W[4*n+i, col+r] = 1
            W[4 * n + i, col + r] = 1.0
    col += R2
    
    # Part 3: A2 B21 (4x1x4)
    # i in 0..3, j=4, k in 0..3
    # Rank 1 decomposition
    for i in range(4):
        for k in range(4):
            U[i * m + 4, col] = 1.0
            V[4 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
    # Note: The above loop sets multiple columns to the same values?
    # No, Rank 1 means one column.
    # But C has 16 entries.
    # C_{ki} = A_{i4} B_{4k}.
    # M_0 = A_{i4} B_{4k}.
    # This M_0 is the same for all i,k?
    # No, M_r depends on i,k?
    # In the formula C_{ik} = sum_r U_{ir} W_{kr} M_r.
    # M_r is a scalar computed from A and B.
    # For Rank 1:
    # M_0 = sum_j A_{ij} V_{jr} B_{jk}.
    # j=4. V[4*p+k, 0] = 1?
    # If V[4*p+k, 0] = 1 for all k, then M_0 = sum_k A_{i4} * 1 * B_{4k} = A_{i4} sum_k B_{4k}.
    # This is not A_{i4} B_{4k}.
    
    # Correction for Part 3:
    # We need C_{ki} = A_{i4} B_{4k}.
    # This is an outer product of vector A_col4 and vector B_row4.
    # The tensor for outer product $u v^T$ has rank 1?
    # $C_{ik} = u_i v_k$.
    # $C_{ik} = U_{ir} W_{kr} M_r$.
    # $M_r = \sum_j A_{ij} V_{jr} B_{jk}$.
    # Here $A$ is $4 \times 1$, $B$ is $1 \times 4$.
    # $j$ is fixed at 0 (local index) or 4 (global).
    # $M_0 = A_{i4} V_{4,0} B_{4k}$.
    # If $V_{4,0} = 1$, $M_0 = A_{i4} B_{4k}$.
    # Then $C_{ki} = U_{i4} W_{4k} A_{i4} B_{4k}$.
    # We need $U_{i4} W_{4k} = 1$ for all $i,k$.
    # So $U_{i4} = 1, W_{4k} = 1$.
    # This works with Rank 1.
    
    # So for Part 3:
    U[i * m + 4, col] = 1.0 # This is wrong, i is loop variable.
    
    # Reset Part 3
    # We need to set U, V, W for the single column col.
    # U rows: i*m + 4 for i in 0..3. All 1.
    # V rows: 4*p + k for k in 0..3. All 1.
    # W rows: k*n + i for k in 0..3, i in 0..3. All 1.
    
    for i in range(4):
        U[i * m + 4, col] = 1.0
    for k in range(4):
        V[4 * p + k, col] = 1.0
    for i in range(4):
        for k in range(4):
            W[k * n + i, col] = 1.0
    col += R3
    
    # Part 4: A2 B22 (4x1x1)
    # i in 0..3, j=4, k=4
    # C_{4i} = A_{i4} B_{44}.
    # Rank 1.
    # M_0 = A_{i4} V_{4,4} B_{44}.
    # V_{4,4} = 1.
    # C_{4i} = U_{i4} W_{4i} A_{i4} B_{44}.
    # U_{i4} = 1, W_{4i} = 1.
    
    for i in range(4):
        U[i * m + 4, col] = 1.0
    V[4 * p + 4, col] = 1.0
    for i in range(4):
        W[4 * n + i, col] = 1.0
    col += R4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
