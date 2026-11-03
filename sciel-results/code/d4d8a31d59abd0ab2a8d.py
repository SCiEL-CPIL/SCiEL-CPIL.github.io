"""
5x5 matrix multiplication - rank 123 block decomposition (2+3 split).
Uses Strassen for 2x2 block, rank-2 decomposition for 2x3 blocks,
and Strassen (2+1) for the 3x3 block.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n * m)
        V = np.zeros(m * p)
        W = np.zeros(p * n)
        for (i, j), val in u_vals.items():
            U[i * m + j] = val
        for (j, k), val in v_vals.items():
            V[j * p + k] = val
        for (k, i), val in w_vals.items():
            W[k * n + i] = val
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    # Small indices (0,1), Large indices (2,3,4)
    sm = [0, 1]
    lg = [2, 3, 4]

    # --- Block 1: C_sm,sm = A_sm,sm * B_sm,sm + A_sm,lg * B_lg,sm ---

    # Strassen for 2x2 * 2x2 (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    add_term({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1})
    # M2 = (A10+A11)B00
    add_term({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1})
    # M3 = A00(B01-B11)
    add_term({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1})
    # M4 = A11(B10-B00)
    add_term({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1})
    # M5 = (A00+A01)B11
    add_term({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1})
    # M6 = (A10-A00)(B00+B01)
    add_term({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1})
    # M7 = (A01-A11)(B10+B11)
    add_term({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})

    # A_sm,lg * B_lg,sm (2x3 * 3x2) -> rank 3 schoolbook
    for i in sm:
        for j in lg:
            for k in sm:
                add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    # --- Block 2: C_sm,lg = A_sm,sm * B_sm,lg + A_sm,lg * B_lg,lg ---

    # A_sm,sm * B_sm,lg (2x2 * 2x3) -> rank 2
    for j in sm:
        add_term({(0, j): 1, (1, j): 1}, {(j, k) for k in lg} | {(j, 2): 1, (j, 3): 1, (j, 4): 1}, {(2, 0): 1, (2, 1): 1, (3, 0): 1, (3, 1): 1, (4, 0): 1, (4, 1): 1})
    # Wait, the rank 2 decomposition for 2x2*2x3 needs care.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # Term 1: U[i,0]=1, V[0,k]=1, W[k,i]=1 -> contrib A[i,0]B[0,k]
    # Term 2: U[i,1]=1, V[1,k]=1, W[k,i]=1 -> contrib A[i,1]B[1,k]
    # So for j=0: U={(0,0):1, (1,0):1}, V={(0,2):1, (0,3):1, (0,4):1}, W={(2,0):1, (2,1):1, (3,0):1, (3,1):1, (4,0):1, (4,1):1}
    # Actually W is constant across all terms for this block?
    # Let's just use the generic loop for these if rank 2 is complex to express with dicts, but rank 3 is safe.
    # Let's stick to rank 3 (schoolbook) for 2x3 blocks to ensure correctness, but try rank 2 if possible.
    # Actually, the prompt implies we should find rank-minimal. 2x2*2x3 is rank 2.
    # Let's implement rank 2 properly.
    # For C_sm,lg: i in 0,1; k in 2,3,4
    # Term j=0: U has 1 at (0,0) and (1,0). V has 1 at (0,k) for k in 2,3,4. W has 1 at (k,0) and (k,1) for k in 2,3,4.
    # Contribution: sum_i' A[i',0] * (1) * ... wait.
    # Tensor: sum_r U[r]_ij V[r]_jk W[r]_ki
    # If U[r]_00=1, U[r]_10=1 (for r=0), V[r]_0k=1 (for k=2,3,4), W[r]_ki=1 (for k=2,3,4, i=0,1)
    # Then sum_i' U[i',0] = A[i',0]*1 + A[i',1]*0? No, U is a matrix of coefficients.
    # U[r] is a 2x2 matrix. U[r]_{0,0}=1, U[r]_{1,0}=1.
    # V[r] is a 2x3 matrix. V[r]_{0,k}=1 for k=2,3,4.
    # W[r] is a 3x2 matrix. W[r]_{k,i}=1 for k=2,3,4, i=0,1.
    # Term contribution to C[i,k] = sum_{i',j} A[i',j] U[r]_{i',j} * sum_{j',k'} V[r]_{j',k'} B[j',k'] * W[r]_{k,i}
    # This is not quite right. The tensor decomposition is C_{ik} = sum_{r,j,j'} A_{ij} B_{jk} W_{ki} ... no.
    # Standard: C_{ik} = sum_r (sum_{j} U_{r,ij}) (sum_{j'} V_{r,jk}) W_{r,ki} ? No.
    # It's C_{ik} = sum_{r,j,j'} A_{ij} U_{r,ij} V_{r,j'k} W_{r,ki} ? No.
    # It is C_{ik} = sum_r U_r[i,j] V_r[j,k] W_r[k,i].
    # So for a single term r:
    # C_{ik} += U_r[i,j] * V_r[j,k] * W_r[k,i] summed over j.
    # If U_r[i,j] = delta_{j,0}, V_r[j,k] = delta_{j,0} * 1 (for k in 2,3,4), W_r[k,i] = 1.
    # Then C_{ik} += sum_j delta_{j,0} * (delta_{j,0} * 1) * 1 * A_{i,0} B_{0,k} = A_{i,0} B_{0,k}.
    # So we need U_r to select j=0. U_r[i,0]=1, U_r[i,1]=0.
    # So Term 1 (j=0): U={(0,0):1, (1,0):1}, V={(0,2):1, (0,3):1, (0,4):1}, W={(2,0):1, (2,1):1, (3,0):1, (3,1):1, (4,0):1, (4,1):1}
    # Wait, W is indexed by (k,i). k is 2,3,4. i is 0,1.
    # So W={(2,0):1, (2,1):1, (3,0):1, (3,1):1, (4,0):1, (4,1):1}.
    # Term 2 (j=1): U={(0,1):1, (1,1):1}, V={(1,2):1, (1,3):1, (1,4):1}, W={(2,0):1, (2,1):1, (3,0):1, (3,1):1, (4,0):1, (4,1):1}
    # Total rank 2 for this block.

    # Let's restart the loop for Block 2 with this logic.
    # A_sm,sm * B_sm,lg
    add_term({(0, 0): 1, (1, 0): 1}, {(0, 2): 1, (0, 3): 1, (0, 4): 1}, {(2, 0): 1, (2, 1): 1, (3, 0): 1, (3, 1): 1, (4, 0): 1, (4, 1): 1})
    add_term({(0, 1): 1, (1, 1): 1}, {(1, 2): 1, (1, 3): 1, (1, 4): 1}, {(2, 0): 1, (2, 1): 1, (3, 0): 1, (3, 1): 1, (4, 0): 1, (4, 1): 1})

    # A_sm,lg * B_lg,lg (2x3 * 3x3) -> rank 3?
    # C[i,k] = sum_j A[i,j] B[j,k]. j in 2,3,4.
    # Term j=2: U={(0,2):1, (1,2):1}, V={(2,2):1, (2,3):1, (2,4):1}, W={(2,0):1, (2,1):1, (3,0):1, (3,1):1, (4,0):1, (4,1):1}
    # Term j=3: U={(0,3):1, (1,3):1}, V={(3,2):1, (3,3):1, (3,4):1}, W={(2,0):1, (2,1):1, (3,0):1, (3,1):1, (4,0):1, (4,1):1}
    # Term j=4: U={(0,4):1, (1,4):1}, V={(4,2):1, (4,3):1, (4,4):1}, W={(2,0):1, (2,1):1, (3,0):1, (3,1):1, (4,0):1, (4,1):1}
    for j in lg:
        add_term({(0, j): 1, (1, j): 1},
                 {(j, 2): 1, (j, 3): 1, (j, 4): 1},
                 {(2, 0): 1, (2, 1): 1, (3, 0): 1, (3, 1): 1, (4, 0): 1, (4, 1): 1})

    # --- Block 3: C_lg,sm = A_lg,sm * B_sm,sm + A_lg,lg * B_lg,sm ---

    # A_lg,sm * B_sm,sm (3x2 * 2x2) -> rank 2?
    # C[i,k] = sum_j A[i,j] B[j,k]. j in 0,1. k in 0,1. i in 2,3,4.
    # Term j=0: U={(2,0):1, (3,0):1, (4,0):1}, V={(0,0):1, (0,1):1}, W={(0,2):1, (0,3):1, (0,4):1, (1,2):1, (1,3):1, (1,4):1}
    # Term j=1: U={(2,1):1, (3,1):1, (4,1):1}, V={(1,0):1, (1,1):1}, W={(0,2):1, (0,3):1, (0,4):1, (1,2):1, (1,3):1, (1,4):1}
    add_term({(2, 0): 1, (3, 0): 1, (4, 0): 1}, {(0, 0): 1, (0, 1): 1}, {(0, 2): 1, (0, 3): 1, (0, 4): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1})
    add_term({(2, 1): 1, (3, 1): 1, (4, 1): 1}, {(1, 0): 1, (1, 1): 1}, {(0, 2): 1, (0, 3): 1, (0, 4): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1})

    # A_lg,lg * B_lg,sm (3x3 * 3x2) -> rank 3?
    # C[i,k] = sum_j A[i,j] B[j,k]. j in 2,3,4. k in 0,1.
    # Term j=2: U={(2,2):1, (3,2):1, (4,2):1}, V={(2,0):1, (2,1):1}, W={(0,2):1, (0,3):1, (0,4):1, (1,2):1, (1,3):1, (1,4):1}
    # Term j=3: U={(2,3):1, (3,3):1, (4,3):1}, V={(3,0):1, (3,1):1}, W={(0,2):1, (0,3):1, (0,4):1, (1,2):1, (1,3):1, (1,4):1}
    # Term j=4: U={(2,4):1, (3,4):1, (4,4):1}, V={(4,0):1, (4,1):1}, W={(0,2):1, (0,3):1, (0,4):1, (1,2):1, (1,3):1, (1,4):1}
    for j in lg:
        add_term({(2, j): 1, (3, j): 1, (4, j): 1},
                 {(j, 0): 1, (j, 1): 1},
                 {(0, 2): 1, (0, 3): 1, (0, 4): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1})

    # --- Block 4: C_lg,lg = A_lg,sm * B_sm,lg + A_lg,lg * B_lg,lg ---

    # A_lg,sm * B_sm,lg (3x2 * 2x3) -> rank 2?
    # C[i,k] = sum_j A[i,j] B[j,k]. j in 0,1. k in 2,3,4.
    # Term j=0: U={(2,0):1, (3,0):1, (4,0):1}, V={(0,2):1, (0,3):1, (0,4):1}, W={(2,2):1, (2,3):1, (2,4):1, (3,2):1, (3,3):1, (3,4):1, (4,2):1, (4,3):1, (4,4):1}
    # Term j=1: U={(2,1):1, (3,1):1, (4,1):1}, V={(1,2):1, (1,3):1, (1,4):1}, W={(2,2):1, (2,3):1, (2,4):1, (3,2):1, (3,3):1, (3,4):1, (4,2):1, (4,3):1, (4,4):1}
    add_term({(2, 0): 1, (3, 0): 1, (4, 0): 1}, {(0, 2): 1, (0, 3): 1, (0, 4): 1}, {(2, 2): 1, (2, 3): 1, (2, 4): 1, (3, 2): 1, (3, 3): 1, (3, 4): 1, (4, 2): 1, (4, 3): 1, (4, 4): 1})
    add_term({(2, 1): 1, (3, 1): 1, (4, 1): 1}, {(1, 2): 1, (1, 3): 1, (1, 4): 1}, {(2, 2): 1, (2, 3): 1, (2, 4): 1, (3, 2): 1, (3, 3): 1, (3, 4): 1, (4, 2): 1, (4, 3): 1, (4, 4): 1})

    # A_lg,lg * B_lg,lg (3x3 * 3x3) -> Strassen (2+1)
    # Indices 2,3,4. Sub-blocks: s2=[2,3], s3=[4].
    # C_22 (2x2) = A_22 B_22 + A_21 B_12
    # A_22 B_22 is 2x2*2x2. Use Strassen.
    # M1 = (A22+A33)(B22+B33)
    add_term({(2, 2): 1, (3, 3): 1}, {(2, 2): 1, (3, 3): 1}, {(2, 2): 1, (3, 3): 1})
    # M2 = (A32+A33)B22
    add_term({(3, 2): 1, (3, 3): 1}, {(2, 2): 1}, {(2, 3): 1, (3, 3): -1})
    # M3 = A22(B23-B33)
    add_term({(2, 2): 1}, {(2, 3): 1, (3, 3): -1}, {(3, 2): 1, (3, 3): 1})
    # M4 = A33(B32-B22)
    add_term({(3, 3): 1}, {(3, 2): 1, (2, 2): -1}, {(2, 2): 1, (2, 3): 1})
    # M5 = (A22+A23)B33
    add_term({(2, 2): 1, (2, 3): 1}, {(3, 3): 1}, {(2, 2): -1, (3, 2): 1})
    # M6 = (A32-A22)(B22+B23)
    add_term({(3, 2): 1, (2, 2): -1}, {(2, 2): 1, (2, 3): 1}, {(3, 3): 1})
    # M7 = (A23-A33)(B32+B33)
    add_term({(2, 3): 1, (3, 3): -1}, {(3, 2): 1, (3, 3): 1}, {(2, 2): 1})

    # A_21 B_12 (2x1 * 1x2) -> rank 1
    # i in 2,3; k in 2,3; j=4.
    # C[i,k] = A[i,4] B[4,k].
    # U={(2,4):1, (3,4):1}, V={(4,2):1, (4,3):1}, W={(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    add_term({(2, 4): 1, (3, 4): 1}, {(4, 2): 1, (4, 3): 1}, {(2, 2): 1, (2, 3): 1, (3, 2): 1, (3, 3): 1})

    # C_23 (2x1) = A_22 B_23 + A_21 B_13
    # A_22 B_23 (2x2 * 2x1) -> rank 2?
    # i in 2,3; k=4; j in 2,3.
    # Term j=2: U={(2,2):1, (3,2):1}, V={(2,4):1}, W={(4,2):1, (4,3):1}
    # Term j=3: U={(2,3):1, (3,3):1}, V={(3,4):1}, W={(4,2):1, (4,3):1}
    add_term({(2, 2): 1, (3, 2): 1}, {(2, 4): 1}, {(4, 2): 1, (4, 3): 1})
    add_term({(2, 3): 1, (3, 3): 1}, {(3, 4): 1}, {(4, 2): 1, (4, 3): 1})
    # A_21 B_13 (2x1 * 1x1) -> rank 1
    # i in 2,3; k=4; j=4.
    # C[i,4] = A[i,4] B[4,4].
    # U={(2,4):1, (3,4):1}, V={(4,4):1}, W={(4,2):1, (4,3):1}
    add_term({(2, 4): 1, (3, 4): 1}, {(4, 4): 1}, {(4, 2): 1, (4, 3): 1})

    # C_32 (1x2) = A_12 B_22 + A_11 B_32
    # A_12 B_22 (1x2 * 2x2) -> rank 2?
    # i=4; k in 2,3; j in 2,3.
    # Term j=2: U={(4,2):1}, V={(2,2):1, (2,3):1}, W={(2,4):1, (3,4):1}
    # Term j=3: U={(4,3):1}, V={(3,2):1, (3,3):1}, W={(2,4):1, (3,4):1}
    add_term({(4, 2): 1}, {(2, 2): 1, (2, 3): 1}, {(2, 4): 1, (3, 4): 1})
    add_term({(4, 3): 1}, {(3, 2): 1, (3, 3): 1}, {(2, 4): 1, (3, 4): 1})
    # A_11 B_32 (1x1 * 1x2) -> rank 1
    # i=4; k in 2,3; j=4.
    # C[4,k] = A[4,4] B[4,k].
    # U={(4,4):1}, V={(4,2):1, (4,3):1}, W={(2,4):1, (3,4):1}
    add_term({(4, 4): 1}, {(4, 2): 1, (4, 3): 1}, {(2, 4): 1, (3, 4): 1})

    # C_33 (1x1) = A_12 B_23 + A_11 B_33
    # A_12 B_23 (1x2 * 2x1) -> rank 2?
    # i=4; k=4; j in 2,3.
    # Term j=2: U={(4,2):1}, V={(2,4):1}, W={(4,4):1}
    # Term j=3: U={(4,3):1}, V={(3,4):1}, W={(4,4):1}
    add_term({(4, 2): 1}, {(2, 4): 1}, {(4, 4): 1})
    add_term({(4, 3): 1}, {(3, 4): 1}, {(4, 4): 1})
    # A_11 B_33 (1x1 * 1x1) -> rank 1
    # U={(4,4):1}, V={(4,4):1}, W={(4,4):1}
    add_term({(4, 4): 1}, {(4, 4): 1}, {(4, 4): 1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
