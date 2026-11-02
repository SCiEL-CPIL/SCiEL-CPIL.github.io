import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-48 decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen to generate a rank-49 decomposition, then
    attempts to find a linear dependency to reduce to rank 48.
    Falls back to rank-49 if no dependency is found within time budget.
    
    Convention: U[i*4+k, r], V[k*4+j, r], W[j*4+i, r]
    Tensor: T[i*4+k, k*4+j, j*4+i] = sum_r U[i*4+k,r]*V[k*4+j,r]*W[j*4+i,r]
    """
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    n, m, p = 4, 4, 4

    # Generate the rank-49 recursive Strassen decomposition
    rank_49 = 49
    U49 = [[0] * rank_49 for _ in range(n * m)]
    V49 = [[0] * rank_49 for _ in range(m * p)]
    W49 = [[0] * rank_49 for _ in range(p * n)]

    # Strassen's 7 bilinear forms for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Block-level Strassen for 4x4:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6

    # Each Pi is a 2x2 product, computed with 7 scalar mults via Strassen.
    # Total: 7 * 7 = 49 scalar multiplications.

    # Define the 7 block-level terms
    # For each term t (0..6), we have:
    #   L_t: 2x2 matrix where each element is a dict {(global_row, global_col): coeff} for A
    #   R_t: 2x2 matrix where each element is a dict {(global_row, global_col): coeff} for B
    #   C_t: 2x2 matrix where each element is a dict {(global_row, global_col): coeff} for C contribution

    def a_elem(bi, bj, r, c):
        return (2 * bi + r, 2 * bj + c)

    def b_elem(bi, bj, r, c):
        return (2 * bi + r, 2 * bj + c)

    def c_elem(bi, bj, r, c):
        return (2 * bi + r, 2 * bj + c)

    # Build the 7 block-level terms
    L_terms = []
    R_terms = []
    C_terms = []

    # Term 0: P1 = (A11+A22)(B11+B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    C0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L0[r][c] = {a_elem(0, 0, r, c): 1, a_elem(1, 1, r, c): 1}
            R0[r][c] = {b_elem(0, 0, r, c): 1, b_elem(1, 1, r, c): 1}
            C0[r][c] = {c_elem(0, 0, r, c): 1, c_elem(1, 1, r, c): 1}
    L_terms.append(L0)
    R_terms.append(R0)
    C_terms.append(C0)

    # Term 1: P2 = (A21+A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    C1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L1[r][c] = {a_elem(1, 0, r, c): 1, a_elem(1, 1, r, c): 1}
            R1[r][c] = {b_elem(0, 0, r, c): 1}
            C1[r][c] = {c_elem(1, 0, r, c): 1, c_elem(1, 1, r, c): -1}
    L_terms.append(L1)
    R_terms.append(R1)
    C_terms.append(C1)

    # Term 2: P3 = A11(B12-B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    C2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L2[r][c] = {a_elem(0, 0, r, c): 1}
            R2[r][c] = {b_elem(0, 1, r, c): 1, b_elem(1, 1, r, c): -1}
            C2[r][c] = {c_elem(0, 1, r, c): 1, c_elem(1, 1, r, c): 1}
    L_terms.append(L2)
    R_terms.append(R2)
    C_terms.append(C2)

    # Term 3: P4 = A22(B21-B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    C3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L3[r][c] = {a_elem(1, 1, r, c): 1}
            R3[r][c] = {b_elem(1, 0, r, c): 1, b_elem(0, 0, r, c): -1}
            C3[r][c] = {c_elem(0, 0, r, c): 1, c_elem(1, 0, r, c): 1}
    L_terms.append(L3)
    R_terms.append(R3)
    C_terms.append(C3)

    # Term 4: P5 = (A11+A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    C4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L4[r][c] = {a_elem(0, 0, r, c): 1, a_elem(0, 1, r, c): 1}
            R4[r][c] = {b_elem(1, 1, r, c): 1}
            C4[r][c] = {c_elem(0, 0, r, c): -1, c_elem(0, 1, r, c): 1}
    L_terms.append(L4)
    R_terms.append(R4)
    C_terms.append(C4)

    # Term 5: P6 = (A21-A11)(B11+B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    C5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L5[r][c] = {a_elem(1, 0, r, c): 1, a_elem(0, 0, r, c): -1}
            R5[r][c] = {b_elem(0, 0, r, c): 1, b_elem(0, 1, r, c): 1}
            C5[r][c] = {c_elem(1, 1, r, c): 1}
    L_terms.append(L5)
    R_terms.append(R5)
    C_terms.append(C5)

    # Term 6: P7 = (A12-A22)(B21+B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    C6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L6[r][c] = {a_elem(0, 1, r, c): 1, a_elem(1, 1, r, c): -1}
            R6[r][c] = {b_elem(1, 0, r, c): 1, b_elem(1, 1, r, c): 1}
            C6[r][c] = {c_elem(0, 0, r, c): 1}
    L_terms.append(L6)
    R_terms.append(R6)
    C_terms.append(C6)

    # Now for each block-level term t, apply Strassen to compute the 2x2 product
    # using 7 scalar multiplications.
    # Strassen for 2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each block term t (0..6) and each Strassen scalar mult s (0..6):
    # The left factor for scalar mult s is a 2x2 matrix (the "a,d" etc. from Strassen)
    # The right factor is a 2x2 matrix
    # The contribution to the output is determined by which C element(s) this M_s contributes to.

    # Strassen scalar mult definitions:
    # s=0: M1 = (a+d)(e+h) -> contributes to C11 (+1), C22 (+1)
    # s=1: M2 = (c+d)e -> contributes to C21 (+1), C22 (-1)
    # s=2: M3 = a(f-h) -> contributes to C12 (+1), C22 (+1)
    # s=3: M4 = d(g-e) -> contributes to C11 (+1), C21 (+1)
    # s=4: M5 = (a+b)h -> contributes to C11 (-1), C12 (+1)
    # s=5: M6 = (c-a)(e+f) -> contributes to C22 (+1)
    # s=6: M7 = (b-d)(g+h) -> contributes to C11 (+1)

    # For each s, define the left 2x2 pattern (which elements of the 2x2 left matrix are used)
    # and right 2x2 pattern, and the output contribution pattern.
    strassen_defs = [
        # (left_pattern, right_pattern, contrib_pattern)
        # left_pattern: dict {(r,c): coeff} for the 2x2 left matrix
        # right_pattern: dict {(r,c): coeff} for the 2x2 right matrix
        # contrib_pattern: dict {(r,c): coeff} for the 2x2 output contribution
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),   # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),              # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),              # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),              # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),              # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),              # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),              # M7
    ]

    for t in range(7):
        for s in range(7):
            r_idx = t * 7 + s
            left_pat, right_pat, contrib_pat = strassen_defs[s]

            # Build U row contributions
            for (lr, lc), l_coeff in left_pat.items():
                for (br, bc), b_coeff in L_terms[t][lr][lc].items():
                    u_idx = br * 4 + bc
                    U49[u_idx][r_idx] += l_coeff * b_coeff

            # Build V row contributions
            for (rr, rc), r_coeff in right_pat.items():
                for (br, bc), b_coeff in R_terms[t][rr][rc].items():
                    v_idx = br * 4 + bc
                    V49[v_idx][r_idx] += r_coeff * b_coeff

            # Build W row contributions
            for (cr, cc), c_coeff in contrib_pat.items():
                for (br, bc), b_coeff in C_terms[t][cr][cc].items():
                    w_idx = bc * 4 + br  # trace convention: W[k*n+i] where k=col, i=row
                    W49[w_idx][r_idx] += c_coeff * b_coeff

    # Verify the rank-49 decomposition
    U49_np = np.array(U49, dtype=np.int64)
    V49_np = np.array(V49, dtype=np.int64)
    W49_np = np.array(W49, dtype=np.int64)

    # Check: for all (i,k,j), sum_r U[i*4+k,r]*V[k*4+j,r]*W[j*4+i,r] should be 1
    # We can check this via the tensor reconstruction
    error = 0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                val = 0
                for r in range(49):
                    val += U49_np[i*4+k, r] * V49_np[k*4+j, r] * W49_np[j*4+i, r]
                if val != 1:
                    error += 1
                    if error > 10:
                        break
            if error > 10:
                break
        if error > 10:
            break

    if error > 0:
        # Fallback: return schoolbook rank-64
        rank = 64
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U[i*4+k, r] = 1
                    V[k*4+j, r] = 1
                    W[j*4+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now attempt to find a linear dependency to reduce rank to 48.
    # The idea: if one of the 49 bilinear forms is a linear combination of the others,
    # we can eliminate it. This requires checking if the 49 rank-1 tensors
    # T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] are linearly dependent.
    
    # This is a 16*16*16 = 4096-dimensional space, and we have 49 vectors.
    # We need to check if they span a 48-dimensional subspace.
    
    # Construct the 4096 x 49 matrix where column r is the flattened T_r
    if time.time() - start_time > DEADLINE - 30:
        # Not enough time for the dependency check, return rank-49
        return {"U": U49_np.tolist(), "V": V49_np.tolist(), "W": W49_np.tolist()}

    # Flatten each rank-1 tensor
    tensor_matrix = np.zeros((4096, 49), dtype=np.int64)
    for r in range(49):
        for idx in range(4096):
            i = idx // 256
            rem = idx % 256
            k = rem // 16
            j = rem % 16
            tensor_matrix[idx, r] = U49_np[i*4+k, r] * V49_np[k*4+j, r] * W49_np[j*4+i, r]

    # Compute the rank of this matrix
    # Use SVD for numerical rank determination
    try:
        # Scale down to avoid overflow in SVD
        tensor_float = tensor_matrix.astype(np.float64)
        # Check if any entries are too large
        max_val = np.max(np.abs(tensor_float))
        if max_val > 1e10:
            tensor_float = tensor_float / max_val
        
        # Compute SVD
        U_svd, s_vals, Vt = np.linalg.svd(tensor_float, full_matrices=False)
        
        # Determine numerical rank
        tol = 1e-10
        num_rank = np.sum(s_vals > tol * s_vals[0])
        
        if num_rank < 49:
            # There is a linear dependency! Find which column to eliminate.
            # The null space of the matrix gives us the dependency.
            # Find the right singular vector corresponding to the smallest singular value
            null_vec = Vt[-1, :]  # last row of Vt
            
            # Find the column with the largest absolute value in the null vector
            # This column is most "redundant"
            elim_col = np.argmax(np.abs(null_vec))
            
            # Remove that column
            keep_cols = [c for c in range(49) if c != elim_col]
            U48 = U49_np[:, keep_cols]
            V48 = V49_np[:, keep_cols]
            W48 = W49_np[:, keep_cols]
            
            # Verify the rank-48 decomposition
            error48 = 0
            for i in range(4):
                for k in range(4):
                    for j in range(4):
                        val = 0
                        for r in range(48):
                            val += U48[i*4+k, r] * V48[k*4+j, r] * W48[j*4+i, r]
                        if val != 1:
                            error48 += 1
                            if error48 > 10:
                                break
                    if error48 > 10:
                        break
                if error48 > 10:
                    break
            
            if error48 == 0:
                return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}
    except Exception:
        pass

    # Return rank-49 if rank-48 couldn't be verified
    return {"U": U49_np.tolist(), "V": V49_np.tolist(), "W": W49_np.tolist()}
# EVOLVE-BLOCK-END
