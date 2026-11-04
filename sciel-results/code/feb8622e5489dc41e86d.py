"""Combined: block partitioning (current) + explicit Strassen construction (inspiration 3)."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Trivial schoolbook solution (rank 90) as fallback
    def trivial_solution():
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

    # ============================================================
    # STRATEGY: Block partitioning
    #   Block 1: 1x1x1 (i=0, j=0, k=0) -> rank 1
    #   Block 2: 2x4x5 (i in {1,2}, j in {1..4}, k in {1..5}) -> rank 17
    #   Total rank = 18
    # ============================================================
    #
    # The 2x4x5 decomposition is built from Strassen on 2x2x2 blocks:
    #   2x4x5 = (2x2x2)_{k=0,1} + (2x2x2)_{k=2,3} + (2x2x1)_{k=4}
    #         = 7 + 7 + 3 = 17
    #
    # Strassen for 2x2x2:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    # ============================================================

    def build_2x4x5_decomposition():
        """
        Returns U (8x17), V (20x17), W (10x17) for 2x4x5 multiplication.
        Local indices: i' in {0,1}, j' in {0,1,2,3}, k' in {0,1,2,3,4}
        U indexed by (i', j') -> i'*4 + j'
        V indexed by (j', k') -> j'*5 + k'
        W indexed by (k', i') -> k'*2 + i'
        """
        n_sub, m_sub, p_sub = 2, 4, 5
        U = np.zeros((n_sub * m_sub, 17))
        V = np.zeros((m_sub * p_sub, 17))
        W = np.zeros((p_sub * n_sub, 17))

        def add_strassen_block(j_offset, k_offset, rank_offset):
            """
            Add a Strassen 2x2x2 block for:
              i' in {0,1}, j' in {j_offset, j_offset+1}, k' in {k_offset, k_offset+1}
            7 rank terms starting at rank_offset.
            """
            # Strassen terms:
            # Term 0: (A11+A22)(B11+B22)
            #   U: i'=0,j'=j_offset +1; i'=1,j'=j_offset+1 +1
            #   V: j'=j_offset,k'=k_offset +1; j'=j_offset+1,k'=k_offset+1 +1
            #   W: k'=k_offset,i'=0 +1; k'=k_offset+1,i'=1 +1
            U[0 * m_sub + (j_offset + 0), rank_offset + 0] += 1
            U[1 * m_sub + (j_offset + 1), rank_offset + 0] += 1
            V[(j_offset + 0) * p_sub + (k_offset + 0), rank_offset + 0] += 1
            V[(j_offset + 1) * p_sub + (k_offset + 1), rank_offset + 0] += 1
            W[(k_offset + 0) * n_sub + 0, rank_offset + 0] += 1
            W[(k_offset + 1) * n_sub + 1, rank_offset + 0] += 1

            # Term 1: (A21+A22)B11
            #   U: i'=1,j'=j_offset +1; i'=1,j'=j_offset+1 +1
            #   V: j'=j_offset,k'=k_offset +1
            #   W: k'=k_offset,i'=1 +1
            U[1 * m_sub + (j_offset + 0), rank_offset + 1] += 1
            U[1 * m_sub + (j_offset + 1), rank_offset + 1] += 1
            V[(j_offset + 0) * p_sub + (k_offset + 0), rank_offset + 1] += 1
            W[(k_offset + 0) * n_sub + 1, rank_offset + 1] += 1

            # Term 2: A11(B21-B22)
            #   U: i'=0,j'=j_offset +1
            #   V: j'=j_offset+1,k'=k_offset +1; j'=j_offset+1,k'=k_offset+1 -1
            #   W: k'=k_offset,i'=0 +1
            U[0 * m_sub + (j_offset + 0), rank_offset + 2] += 1
            V[(j_offset + 1) * p_sub + (k_offset + 0), rank_offset + 2] += 1
            V[(j_offset + 1) * p_sub + (k_offset + 1), rank_offset + 2] -= 1
            W[(k_offset + 0) * n_sub + 0, rank_offset + 2] += 1

            # Term 3: A22(B12-B11)
            #   U: i'=1,j'=j_offset+1 +1
            #   V: j'=j_offset,k'=k_offset+1 +1; j'=j_offset,k'=k_offset -1
            #   W: k'=k_offset+1,i'=1 +1
            U[1 * m_sub + (j_offset + 1), rank_offset + 3] += 1
            V[(j_offset + 0) * p_sub + (k_offset + 1), rank_offset + 3] += 1
            V[(j_offset + 0) * p_sub + (k_offset + 0), rank_offset + 3] -= 1
            W[(k_offset + 1) * n_sub + 1, rank_offset + 3] += 1

            # Term 4: (A11+A12)B22
            #   U: i'=0,j'=j_offset +1; i'=0,j'=j_offset+1 +1
            #   V: j'=j_offset+1,k'=k_offset+1 +1
            #   W: k'=k_offset+1,i'=0 +1
            U[0 * m_sub + (j_offset + 0), rank_offset + 4] += 1
            U[0 * m_sub + (j_offset + 1), rank_offset + 4] += 1
            V[(j_offset + 1) * p_sub + (k_offset + 1), rank_offset + 4] += 1
            W[(k_offset + 1) * n_sub + 0, rank_offset + 4] += 1

            # Term 5: (A21-A11)(B11+B12)
            #   U: i'=1,j'=j_offset +1; i'=0,j'=j_offset -1
            #   V: j'=j_offset,k'=k_offset +1; j'=j_offset,k'=k_offset+1 +1
            #   W: k'=k_offset,i'=1 +1; k'=k_offset+1,i'=1 +1
            U[1 * m_sub + (j_offset + 0), rank_offset + 5] += 1
            U[0 * m_sub + (j_offset + 0), rank_offset + 5] -= 1
            V[(j_offset + 0) * p_sub + (k_offset + 0), rank_offset + 5] += 1
            V[(j_offset + 0) * p_sub + (k_offset + 1), rank_offset + 5] += 1
            W[(k_offset + 0) * n_sub + 1, rank_offset + 5] += 1
            W[(k_offset + 1) * n_sub + 1, rank_offset + 5] += 1

            # Term 6: (A12-A22)(B21+B22)
            #   U: i'=0,j'=j_offset+1 +1; i'=1,j'=j_offset+1 -1
            #   V: j'=j_offset+1,k'=k_offset +1; j'=j_offset+1,k'=k_offset+1 +1
            #   W: k'=k_offset,i'=0 +1; k'=k_offset+1,i'=0 +1
            U[0 * m_sub + (j_offset + 1), rank_offset + 6] += 1
            U[1 * m_sub + (j_offset + 1), rank_offset + 6] -= 1
            V[(j_offset + 1) * p_sub + (k_offset + 0), rank_offset + 6] += 1
            V[(j_offset + 1) * p_sub + (k_offset + 1), rank_offset + 6] += 1
            W[(k_offset + 0) * n_sub + 0, rank_offset + 6] += 1
            W[(k_offset + 1) * n_sub + 0, rank_offset + 6] += 1

        # Block A: j' in {0,1}, k' in {0,1} -> Strassen, rank 0-6
        add_strassen_block(j_offset=0, k_offset=0, rank_offset=0)

        # Block B: j' in {2,3}, k' in {2,3} -> Strassen, rank 7-13
        add_strassen_block(j_offset=2, k_offset=2, rank_offset=7)

        # Block C: j' in {0,1}, k' in {4} -> schoolbook 2x2x1, rank 14-16
        # C[i', k'=4] = sum_{j'=0,1} A[i',j'] * B[j',4]
        # Term 14: i'=0, j'=0, k'=4
        U[0 * m_sub + 0, 14] += 1
        V[0 * p_sub + 4, 14] += 1
        W[4 * n_sub + 0, 14] += 1
        # Term 15: i'=1, j'=0, k'=4
        U[1 * m_sub + 0, 15] += 1
        V[0 * p_sub + 4, 15] += 1
        W[4 * n_sub + 1, 15] += 1
        # Term 16: i'=0, j'=1, k'=4  AND  i'=1, j'=1, k'=4
        # Actually for 2x2x1 schoolbook: C[i',4] = A[i',0]*B[0,4] + A[i',1]*B[1,4]
        # So we need: for each i' in {0,1}, j' in {0,1}: one term
        # That's 4 terms, but we only have 3 slots (rank 14,15,16).
        # Wait, 2x2x1 schoolbook needs 2*2*1 = 4 terms.
        # Let me recount: rank 14,15,16,17 would be 4 terms, but we only have 17 total.
        # 7 + 7 + 4 = 18, not 17.
        # Hmm, let me reconsider.
        # Actually 2x2x1: C[i',k] = sum_j A[i',j]*B[j,k]. For fixed k, this is 2x2x1.
        # Schoolbook rank for 2x2x1 is 4.
        # So total = 7 + 7 + 4 = 18.
        # But we want 17. Let me use a different partition.
        # Alternative: 2x4x5 = (2x2x2) + (2x2x2) + (2x2x1)
        # 2x2x1 schoolbook = 4, so total = 7+7+4 = 18.
        # Or: (2x2x3) + (2x2x2) = ?
        # 2x2x3 schoolbook = 12, 2x2x2 Strassen = 7, total = 19. Worse.
        # Or: (2x3x2) + (2x1x2) + ... this is getting complicated.
        # Let me just use 18 for the sub-problem: total = 1 + 18 = 19.
        # Actually, let me try a different approach:
        # 2x4x5: partition j into {0,1} and {2,3}, k into {0,1,2,3} and {4}
        # Block 1: 2x2x4 (j in {0,1}, k in {0,1,2,3})
        # Block 2: 2x2x1 (j in {2,3}, k in {4})  -- wait, j needs to be contiguous for the matrix structure
        # Actually the matrix multiplication tensor has a specific structure.
        # Let me reconsider: the tensor is T[i,j,k] = 1 for all i,j,k.
        # The factorization must satisfy: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k.
        # For the sub-problem 2x4x5, the tensor is the same structure.
        # So I can partition however I want as long as the indices match up.
        # Let me use: j in {0,1} with k in {0,1,2,3} (2x2x4), and j in {2,3} with k in {0,1,2,3,4} (2x2x5).
        # 2x2x4: schoolbook = 16, Strassen-based? 2x2x4 = (2x2x2)+(2x2x2) = 7+7 = 14
        # 2x2x5: schoolbook = 20, or (2x2x2)+(2x2x2)+(2x2x1) = 7+7+4 = 18
        # Total = 14 + 18 = 32. Worse than before.
        # 
        # Let me go back to the original partition and accept rank 18 for the sub-problem.
        # Total = 1 + 18 = 19.
        # Actually, let me try: 2x4x5 = (2x2x2)_{j=0,1;k=0,1} + (2x2x2)_{j=2,3;k=0,1} + (2x4x3)_{j=0..3;k=2,3,4}
        # 2x2x2 Strassen = 7, another 2x2x2 Strassen = 7, 2x4x3 schoolbook = 24. Total = 38. Bad.
        # 
        # Best simple partition: (2x2x2) + (2x2x2) + (2x2x1) with schoolbook for the last = 7+7+4 = 18.
        # Total rank = 1 + 18 = 19.
        # 
        # Let me just do that. It's deterministic and correct.

        # Redo Block C: 2x2x1 schoolbook for j' in {0,1}, k'=4
        # C[i',4] = A[i',0]*B[0,4] + A[i',1]*B[1,4]
        # 4 terms: (i',j') for i' in {0,1}, j' in {0,1}
        # Term 14: i'=0, j'=0
        U[0 * m_sub + 0, 14] = 1
        V[0 * p_sub + 4, 14] = 1
        W[4 * n_sub + 0, 14] = 1
        # Term 15: i'=1, j'=0
        U[1 * m_sub + 0, 15] = 1
        V[0 * p_sub + 4, 15] = 1
        W[4 * n_sub + 1, 15] = 1
        # Term 16: i'=0, j'=1
        U[0 * m_sub + 1, 16] = 1
        V[1 * p_sub + 4, 16] = 1
        W[4 * n_sub + 0, 16] = 1
        # We need a 4th term for i'=1, j'=1, but we only have 17 columns (0-16).
        # So I need 18 columns for the sub-problem. Let me adjust.

        # Actually, let me just extend to 18 columns for the sub-problem.
        # I'll return U,V,W with 18 columns.
        # Let me rebuild with 18 columns.
        U = np.zeros((n_sub * m_sub, 18))
        V = np.zeros((m_sub * p_sub, 18))
        W = np.zeros((p_sub * n_sub, 18))

        # Block A: j' in {0,1}, k' in {0,1} -> Strassen, rank 0-6
        add_strassen_block(j_offset=0, k_offset=0, rank_offset=0)

        # Block B: j' in {2,3}, k' in {2,3} -> Strassen, rank 7-13
        add_strassen_block(j_offset=2, k_offset=2, rank_offset=7)

        # Block C: j' in {0,1}, k'=4 -> schoolbook 2x2x1, rank 14-17
        # Term 14: i'=0, j'=0, k'=4
        U[0 * m_sub + 0, 14] = 1
        V[0 * p_sub + 4, 14] = 1
        W[4 * n_sub + 0, 14] = 1
        # Term 15: i'=1, j'=0, k'=4
        U[1 * m_sub + 0, 15] = 1
        V[0 * p_sub + 4, 15] = 1
        W[4 * n_sub + 1, 15] = 1
        # Term 16: i'=0, j'=1, k'=4
        U[0 * m_sub + 1, 16] = 1
        V[1 * p_sub + 4, 16] = 1
        W[4 * n_sub + 0, 16] = 1
        # Term 17: i'=1, j'=1, k'=4
        U[1 * m_sub + 1, 17] = 1
        V[1 * p_sub + 4, 17] = 1
        W[4 * n_sub + 1, 17] = 1

        return U, V, W  # shapes (8,18), (20,18), (10,18)

    # Verify the sub-decomposition
    U_sub, V_sub, W_sub = build_2x4x5_decomposition()
    n_sub, m_sub, p_sub = 2, 4, 5
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0

    T_approx_sub = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
    err_sub = np.sum((T_approx_sub - T_sub) ** 2)

    if err_sub < 1e-9:
        # Build full solution with rank = 1 + 18 = 19
        R_full = 19
        U_full = np.zeros((n * m, R_full))
        V_full = np.zeros((m * p, R_full))
        W_full = np.zeros((p * n, R_full))

        # Block 1: 1x1x1 (i=0, j=0, k=0)
        U_full[0 * m + 0, 0] = 1
        V_full[0 * p + 0, 0] = 1
        W_full[0 * n + 0, 0] = 1

        # Block 2: 2x4x5 (i in {1,2}, j in {1..4}, k in {1..5})
        # Local i' in {0,1} -> Global i = i' + 1
        # Local j' in {0,1,2,3} -> Global j = j' + 1
        # Local k' in {0,1,2,3,4} -> Global k = k' + 1
        for r in range(18):
            # U_sub
            for i_prime in range(2):
                for j_prime in range(4):
                    local_u_idx = i_prime * m_sub + j_prime
                    global_i = i_prime + 1
                    global_j = j_prime + 1
                    global_u_idx = global_i * m + global_j
                    U_full[global_u_idx, r + 1] = U_sub[local_u_idx, r]

            # V_sub
            for j_prime in range(4):
                for k_prime in range(5):
                    local_v_idx = j_prime * p_sub + k_prime
                    global_j = j_prime + 1
                    global_k = k_prime + 1
                    global_v_idx = global_j * p + global_k
                    V_full[global_v_idx, r + 1] = V_sub[local_v_idx, r]

            # W_sub
            for k_prime in range(5):
                for i_prime in range(2):
                    local_w_idx = k_prime * n_sub + i_prime
                    global_k = k_prime + 1
                    global_i = i_prime + 1
                    global_w_idx = global_k * n + global_i
                    W_full[global_w_idx, r + 1] = W_sub[local_w_idx, r]

        # Verify full solution
        T_full_target = np.zeros((n * m, m * p, p * n))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    T_full_target[a * m + b, b * p + c, c * n + a] = 1.0

        T_full_approx = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full)
        error_full = np.sum((T_full_approx - T_full_target) ** 2)

        if error_full < 1e-9:
            # Snap to half-integers
            U_snap = np.round(U_full * 2) / 2
            V_snap = np.round(V_full * 2) / 2
            W_snap = np.round(W_full * 2) / 2

            T_full_approx_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            error_full_snap = np.sum((T_full_approx_snap - T_full_target) ** 2)

            if error_full_snap < 1e-9:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }

    # Fallback to trivial
    return trivial_solution()
# EVOLVE-BLOCK-END
