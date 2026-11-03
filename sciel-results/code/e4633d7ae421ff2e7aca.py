"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 140  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        """
        Return Strassen decomposition of 2x2 * 2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices:
          U: row = i*2+j (i,j in {0,1})
          V: row = j*2+k (j,k in {0,1})
          W: row = k*2+i (k,i in {0,1})
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Strassen's 7 products
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # A indices: A00=0, A01=1, A10=2, A11=3
        # B indices: B00=0, B01=1, B10=2, B11=3
        # C indices (k*2+i): C00(k=0,i=0)=0, C01(k=1,i=0)=2, C10(k=0,i=1)=1, C11(k=1,i=1)=3
        
        # t=0: M1
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0
        
        # t=1: M2
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0; W[3, 1] = -1.0
        
        # t=2: M3
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0
        
        # t=3: M4
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0
        
        # t=4: M5
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0
        
        # t=5: M6
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0
        
        # t=6: M7
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0

        return U, V, W

    U_s, V_s, W_s = strassen_2x2_decomp()
    
    full_rank = 28
    U_full = np.zeros((n * m, full_rank))  # 8 x 28
    V_full = np.zeros((m * p, full_rank))  # 16 x 28
    W_full = np.zeros((p * n, full_rank))  # 8 x 28

    def add_block(offset, i_start, j_start_A, j_start_B, k_start):
        """
        Add a Strassen block.
        offset: column offset in full_rank
        i_start: start index for i in A (0 or 1, but A is 2x4 so i is always 0,1? No, A is 2x4. i is row 0,1. j is col 0..3)
        Actually A is 2x4. Rows 0,1. Cols 0..3.
        So i is always 0 or 1.
        j_start_A: start col for A block (0 or 2)
        j_start_B: start row for B block (0 or 2)
        k_start: start col for B block / result col (0 or 2)
        """
        for t in range(7):
            col = offset + t
            # U: A[i, j] -> global i*4 + j
            for li in range(2):
                for lj in range(2):
                    global_a = li * 4 + (j_start_A + lj)
                    U_full[global_a, col] += U_s[li * 2 + lj, t]
            
            # V: B[j, k] -> global j*4 + k
            for lj in range(2):
                for lk in range(2):
                    global_b = (j_start_B + lj) * 4 + (k_start + lk)
                    V_full[global_b, col] += V_s[lj * 2 + lk, t]
                    
            # W: C[k, i] -> global k*2 + i
            for lk in range(2):
                for li in range(2):
                    global_c = (k_start + lk) * 2 + li
                    W_full[global_c, col] += W_s[lk * 2 + li, t]

    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    add_block(0, 0, 0, 0, 0)
    
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    add_block(7, 0, 2, 2, 0)
    
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    add_block(14, 0, 0, 0, 2)
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    add_block(21, 0, 2, 2, 2)

    # Start with rank-28 Strassen decomposition
    best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
    best_rank = 28

    # Try to reduce rank by finding linear dependencies / zeroing columns
    # Strategy: for each column, check if it can be expressed as a combination of others
    # If so, remove it and adjust
    def try_reduce_rank(U, V, W, r, max_iter=50, tol=1e-6):
        """Try to reduce rank by finding redundant columns."""
        U, V, W = U.copy(), V.copy(), W.copy()
        r_current = r
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            found = False
            for t in range(r_current):
                # Check if column t is in span of other columns
                # Use the tensor reconstruction: if we can set column t to zero
                # and compensate with other columns, we've reduced rank
                # Simpler: check if the "slice" for column t is linearly dependent
                # on other slices
                # Build the 3-tensor slice for column t
                slice_t = np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
                # Check if slice_t is in span of other slices
                # This is expensive; use a heuristic: try to express as combo
                # Use SVD on the matricized tensor
                # For now, use a simpler check: if all entries of slice_t are ~0
                if np.max(np.abs(slice_t)) < tol:
                    r_current -= 1
                    if t < r_current:
                        U[:, t] = U[:, r_current]
                        V[:, t] = V[:, r_current]
                        W[:, t] = W[:, r_current]
                    else:
                        U[:, t] = 0
                        V[:, t] = 0
                        W[:, t] = 0
                    found = True
                    break
            if not found:
                break
        return U[:, :r_current], V[:, :r_current], W[:, :r_current], r_current

    # Try rank reduction on the initial decomposition
    U_red, V_red, W_red, r_red = try_reduce_rank(U_full, V_full, W_full, 28)
    if r_red < best_rank and verify(U_red, V_red, W_red, tol=1e-8):
        best_U, best_V, best_W, best_rank = U_red, V_red, W_red, r_red

    # Try a different approach: combine blocks 1&2 (both contribute to C[:, 0:2])
    # and blocks 3&4 (both contribute to C[:, 2:4]) using shared Strassen structure
    # This requires a 2x4 x 2x2 and 2x4 x 2x2 decomposition
    # For 2x4 * 2x2: A is 2x4, B is 2x2, C is 2x2
    # We can split A into A_left (2x2) and A_right (2x2)
    # C = A_left * B_left + A_right * B_right where B is 2x2
    # Actually B is 2x2, so B_left = B[:, 0:1] and B_right = B[:, 1:2]? No, that's not right.
    # Let me think again. A is 2x4, B is 2x2. C = A @ B is 2x2.
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [[b00, b01], [b10, b11]]
    # C = A1 @ B + A2 @ B? No, that's not how matrix multiplication works.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = A[i,0] B[0,k] + A[i,1] B[1,k] + A[i,2] B[2,k] + A[i,3] B[3,k]
    # But B is 2x2, so j only goes 0,1. So A must be 2x2 for this to work.
    # I'm confusing myself. Let me restart the structural analysis.

    # The correct decomposition:
    # A is 2x4, B is 4x4, C is 2x4
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # Split j into {0,1} and {2,3}:
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    # Let A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # Let B_top = B[0:2, :] (2x4), B_bottom = B[2:4, :] (2x4)
    # C = A_left @ B_top + A_right @ B_bottom
    # Each term is 2x2 * 2x4 = 2x4
    # So we need a 2x2 * 2x4 algorithm, used twice, and sum the results.
    # For 2x2 * 2x4: A is 2x2, B is 2x4, C is 2x4
    # C[i,k] = A[i,0] B[0,k] + A[i,1] B[1,k] for k=0..3
    # This is two 2x2 * 2x2 multiplications (for k=0,1 and k=2,3)
    # So 2x2 * 2x4 has rank at most 14 (two Strassen 2x2x2 blocks)
    # Total: 2 * 14 = 28, same as before.
    # But we can potentially share structure between the two 2x2 * 2x4 calls
    # since they use different A matrices but the same B structure pattern.

    # Let me try a direct rank-24 construction using a known result or search
    # For now, return the best we have
    if verify(best_U, best_V, best_W, tol=1e-8):
        return to_lists(best_U, best_V, best_W, best_rank)

    # Fallback: schoolbook rank 32
    schoolbook_rank = n * m * p
    U = np.zeros((n*m, schoolbook_rank)); V = np.zeros((m*p, schoolbook_rank)); W = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
    return to_lists(U, V, W, schoolbook_rank)
# EVOLVE-BLOCK-END
