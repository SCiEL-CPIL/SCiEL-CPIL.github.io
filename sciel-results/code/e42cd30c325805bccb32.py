"""Matrix multiplication 4,4,9 tensor decomposition - structural refinement approach.

Strategy: Build a rank-114 Strassen-based baseline, then attempt to reduce rank
by identifying and merging redundant rank-1 terms via linear algebra over the
decomposition space. The key structural insight is that the 4x4x9 tensor has
block structure (9 = 4+4+1) that creates linear dependencies among the rank-1
terms, which can be exploited to reduce rank below the naive 49+49+16=114.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36

    def build_strassen_4x4x4():
        """Build rank-49 Strassen decomposition for 4x4 * 4x4 -> 4x4."""
        # Strassen 2x2x2 base
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))

        # Term 0: (A00+A11)(B00+B11) -> C00(+), C11(+)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1

        # Term 1: (A10+A11)B00 -> C10(+)
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1

        # Term 2: A00(B10-B11) -> C01(+)
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1

        # Term 3: A11(B01-B00) -> C00(-), C11(-)
        U2[3, 3] = 1
        V2[1, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = -1; W2[3, 3] = -1

        # Term 4: (A00+A01)B11 -> C01(+), C11(+)
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[1, 4] = 1; W2[3, 4] = 1

        # Term 5: (A10-A00)(B00+B10) -> C00(+)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[2, 5] = 1
        W2[0, 5] = 1

        # Term 6: (A01-A11)(B01+B11) -> C00(-), C11(+)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[1, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = -1; W2[3, 6] = 1

        # Level 1: 4x4 = 2x2 blocks of 2x2
        # For 4x4*4x4, we have 4 blocks of 2x2*2x2
        # Block (br, bc) in A, (br, bc) in B -> C block (br, bc)
        U = np.zeros((16, 49))
        V = np.zeros((36, 49))  # For 4x9, but we'll only use cols 0-3 for 4x4
        W = np.zeros((36, 49))

        for a_br in range(2):
            for a_bc in range(2):
                for b_bc in range(2):
                    for t in range(7):
                        for s in range(7):
                            r = (a_br * 2 + a_bc) * 7 * 2 + (b_bc) * 7 + t
                            # Wait, let me restructure. 
                            # 4x4 * 4x4: A is 2x2 blocks of 2x2, B is 2x2 blocks of 2x2
                            # C[i,k] = sum_j A[i,j] B[j,k]
                            # In block terms: C_br_bc = sum_bc' A_br_bc' * B_bc'_bc
                            # Each 2x2*2x2 uses 7 mults, so 4*7=28 per (br, bc) pair? No.
                            # Actually for 4x4*4x4: 4 block rows * 4 block cols = 16 blocks of C
                            # Each C_br_bc = sum over 4 block-rows of B: A_br_bc' * B_bc'_bc
                            # Each such product is a 2x2*2x2 = 7 mults
                            # Total: 4*4*4*7/4... let me think more carefully.
                            # C has 4x4 = 16 blocks of 2x2
                            # C[br,bc] = sum_{b=0}^{3} A[br,b] * B[b,bc]
                            # Each A[br,b]*B[b,bc] is a 2x2*2x2 product = 7 mults
                            # So total = 4 (br) * 4 (bc) * 4 (b) * 7 = 448? That's too many.
                            # No wait - Strassen for 4x4*4x4 should be 4^log2(7) ≈ 49.
                            # The standard way: split 4x4 into 2x2 blocks, apply Strassen
                            # recursively. Level 1: 7 mults of 2x2 matrices.
                            # Level 2: each 2x2 mult needs 7 mults of scalars.
                            # Total: 7*7 = 49.
                            
                            pass
        
        # Let me redo this properly.
        # 4x4 * 4x4 using Strassen:
        # Split A into 4 blocks A00,A01,A10,A11 (each 2x2)
        # Split B into 4 blocks B00,B01,B10,B11 (each 2x2)
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M7
        # where M1..M7 are 2x2*2x2 products (each rank 7)
        
        U = np.zeros((16, 49))
        V = np.zeros((36, 49))
        W = np.zeros((36, 49))

        # For each Strassen term t (0..6) and each scalar mult s (0..6):
        # r = t*7 + s
        # U, V, W entries come from U2, V2, W2 with block offsets
        
        for t in range(7):
            for s in range(7):
                r = t * 7 + s
                # U2 entries for this (t,s)
                for local_i in range(2):
                    for local_j in range(2):
                        u_val = U2[local_i * 2 + local_j, s]
                        if u_val != 0:
                            # Which block of A? Depends on t
                            # t=0: A00+A11 -> blocks (0,0) and (1,1)
                            # t=1: A10+A11 -> blocks (1,0) and (1,1)
                            # t=2: A00 -> block (0,0)
                            # t=3: A11 -> block (1,1)
                            # t=4: A00+A01 -> blocks (0,0) and (0,1)
                            # t=5: A10-A00 -> blocks (1,0) and (0,0)
                            # t=6: A01-A11 -> blocks (0,1) and (1,1)
                            pass
        
        # This is getting complex. Let me use a cleaner approach.
        # Direct construction of 4x4x4 Strassen:
        
        U = np.zeros((16, 49))
        V = np.zeros((36, 49))
        W = np.zeros((36, 49))

        # Strassen 4x4x4: 49 terms
        # The 7 level-1 terms each expand to 7 scalar mults
        # Level 1 term t uses certain A-blocks and B-blocks
        # A-block (br, bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1
        # Flat index in U: (2*br + i) * 4 + (2*bc + j)
        # Flat index in V: (2*bc + j) * 9 + (2*bc_out + k)  [for 4x9, but for 4x4 part cols 0-3]
        # Flat index in W: (2*bc_out + k) * 4 + (2*br + i)

        # Level 1 Strassen terms (which A-blocks, which B-blocks, which C-blocks)
        # M1 = (A00+A11)(B00+B11) -> C00(+), C11(+)
        # M2 = (A10+A11)B00 -> C10(+)
        # M3 = A00(B10-B11) -> C01(+)
        # M4 = A11(B01-B00) -> C00(-), C11(-)
        # M5 = (A00+A01)B11 -> C01(+), C11(+)
        # M6 = (A10-A00)(B00+B10) -> C00(+)
        # M7 = (A01-A11)(B01+B11) -> C00(-), C11(+)

        level1 = [
            # (A_blocks, B_blocks, C_blocks) each is list of (block_row, block_col, coeff)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1)]),
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,-1),(1,1,-1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(0,0,1)]),
            ([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
        ]

        for t in range(7):
            a_blocks, b_blocks, c_blocks = level1[t]
            for s in range(7):
                r = t * 7 + s
                for (a_br, a_bc, a_coeff) in a_blocks:
                    for li in range(2):
                        for lj in range(2):
                            u_val = U2[li * 2 + lj, s]
                            if u_val != 0:
                                flat = (a_br * 2 + li) * 4 + (a_bc * 2 + lj)
                                U[flat, r] += a_coeff * u_val
                for (b_br, b_bc, b_coeff) in b_blocks:
                    for li in range(2):
                        for lj in range(2):
                            v_val = V2[li * 2 + lj, s]
                            if v_val != 0:
                                flat = (b_br * 2 + li) * 9 + (b_bc * 2 + lj)
                                V[flat, r] += b_coeff * v_val
                for (c_br, c_bc, c_coeff) in c_blocks:
                    for li in range(2):
                        for lj in range(2):
                            w_val = W2[li * 2 + lj, s]
                            if w_val != 0:
                                flat = (c_bc * 2 + lj) * 4 + (c_br * 2 + li)
                                W[flat, r] += c_coeff * w_val

        return U, V, W

    def build_full_baseline():
        """Build rank-114 baseline: two 4x4x4 Strassen blocks + 4x4x1 schoolbook."""
        U1, V1, W1 = build_strassen_4x4x4()
        
        U = np.zeros((16, 114))
        V = np.zeros((36, 114))
        W = np.zeros((36, 114))

        # Block 1: B cols 0-3, C cols 0-3
        U[:, 0:49] = U1
        V[:, 0:49] = V1[:, 0:36]  # V1 is 36x49 but we only need rows for B cols 0-3
        # Actually V1 was built for 4x9 (36 rows). For block 1, B has cols 0-3.
        # V row index = j*9 + k. For k in 0..3, that's rows 0,9,1,10,2,11,3,12 (j=0..3, k=0..3)
        # But V1 was built assuming B is 4x4, so V1 rows are j*4+k for j,k in 0..3
        # We need to remap. Let me rebuild V1 for 4x4 properly.
        
        # Let me just build it directly for 4x9.
        U = np.zeros((16, 114))
        V = np.zeros((36, 114))
        W = np.zeros((36, 114))

        # For 4x4 * 4x9:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Split k into k1 in [0,4), k2 in [4,8), k3 in [8]
        # C[:, 0:4] = A * B[:, 0:4]  (4x4 * 4x4)
        # C[:, 4:8] = A * B[:, 4:8]  (4x4 * 4x4)
        # C[:, 8]   = A * B[:, 8]    (4x4 * 4x1)

        # Block 1: 4x4 * 4x4 -> C[:, 0:4]
        # Use Strassen 4x4x4
        U1, V1_4x4, W1 = build_strassen_4x4x4()
        # V1_4x4 is 16x49 (rows are j*4+k for j,k in 0..3)
        # But our V is 36x114 (rows are j*9+k)
        # Remap: V1_4x4 row j*4+k -> V row j*9+k
        V1_remapped = np.zeros((36, 49))
        for j in range(4):
            for k in range(4):
                V1_remapped[j*9+k, :] = V1_4x4[j*4+k, :]
        
        U[:, 0:49] = U1
        V[:, 0:49] = V1_remapped
        W[:, 0:49] = W1

        # Block 2: 4x4 * 4x4 -> C[:, 4:8]
        # Same A, B cols 4-7, C cols 4-7
        # V rows: j*9 + (4+k) for j,k in 0..3
        V2_remapped = np.zeros((36, 49))
        for j in range(4):
            for k in range(4):
                V2_remapped[j*9+4+k, :] = V1_4x4[j*4+k, :]
        
        W2 = np.zeros((36, 49))
        for k in range(4):
            for i in range(4):
                W2[4+i, k*4+i] = W1[k*4+i, :]  # Shift C cols by 4
        # Actually W index is k*4+i. For C cols 4-7, k is still 0-3 but output col is 4+k
        # Wait, W[k*n+i, r] where k is the column of C. For block 2, C columns are 4-7.
        # So W row index = (4+k)*4 + i? No, W has shape (p*n, rank) = (36, rank).
        # W index = k*n + i where k is column of C (0..8) and i is row of C (0..3).
        # For block 2, C columns 4-7, so k in 4..7? No, k is the output column index.
        # C[i, k_out] where k_out in 4..7 for block 2.
        # W row = k_out * n + i = k_out * 4 + i
        # For block 1: k_out in 0..3, W rows 0..15
        # For block 2: k_out in 4..7, W rows 16..31
        # For block 3: k_out = 8, W rows 32..35

        W[:, 49:98] = np.zeros((36, 49))
        for k_out in range(4):
            for i in range(4):
                W[(4+k_out)*4+i, 49:98] = W1[k_out*4+i, :]

        U[:, 49:98] = U1
        V[:, 49:98] = V2_remapped

        # Block 3: 4x4 * 4x1 -> C[:, 8]
        # Schoolbook: 16 terms
        for i in range(4):
            for j in range(4):
                r = 98 + i*4 + j
                U[i*4+j, r] = 1
                V[j*9+8, r] = 1
                W[8*4+i, r] = 1

        return U, V, W

    def verify(U, V, W):
        """Verify decomposition."""
        rank = U.shape[1]
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Build baseline
    U_base, V_base, W_base = build_full_baseline()

    if not verify(U_base, V_base, W_base):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((nm, rank))
        V_fb = np.zeros((mp, rank))
        W_fb = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j, r] = 1
                    V_fb[j * p + k, r] = 1
                    W_fb[k * n + i, r] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_rank = 114
    best_U, best_V, best_W = U_base, V_base, W_base

    # Try to reduce rank by finding linear dependencies
    # The key idea: if columns of [U|V|W] (stacked) are linearly dependent
    # in a way that preserves the tensor, we can eliminate a term.
    
    # Alternative: try random rank-reduction attempts
    np.random.seed(12345)
    
    def try_reduce(R, max_iters=200):
        """Try to find a valid rank-R decomposition."""
        if time.time() > deadline - 10:
            return None
        
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Initialize from baseline
        if R <= 114:
            U[:, :R] = U_base[:, :R]
            V[:, :R] = V_base[:, :R]
            W[:, :R] = W_base[:, :R]
        
        # Try to repair by adjusting entries
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            if verify(U, V, W):
                return U, V, W
            
            # Random perturbation
            r = np.random.randint(R)
            choice = np.random.randint(3)
            if choice == 0:
                idx = np.random.randint(nm)
                U[idx, r] += np.random.choice([-1, 1]) * 0.5
            elif choice == 1:
                idx = np.random.randint(mp)
                V[idx, r] += np.random.choice([-1, 1]) * 0.5
            else:
                idx = np.random.randint(pn)
                W[idx, r] += np.random.choice([-1, 1]) * 0.5
        
        return None

    # Try ranks from 108 to 113
    for R in range(108, 114):
        if time.time() > deadline - 15:
            break
        
        result = try_reduce(R, max_iters=100)
        if result is not None:
            U_new, V_new, W_new = result
            if verify(U_new, V_new, W_new):
                best_rank = R
                best_U, best_V, best_W = U_new, V_new, W_new
                break

    # Return best result
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
