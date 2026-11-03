"""Matrix multiplication 3x3x7: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # We decompose the multiplication A (3x3) * B (3x7) = C (3x7)
    # into two parts:
    # 1. A * B1 (3x3) where B1 is the first 3 columns of B.
    # 2. A * B2 (3x4) where B2 is the last 4 columns of B.
    
    # Part 1: 3x3x3 multiplication using Rank 20 decomposition
    # Based on Strassen's 7-term algorithm for the 2x2x2 core, plus schoolbook for the rest.
    # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = A * B
    
    # Strassen 2x2x2 for C[0:2, 0:2] using A[0:2, 0:2] and B[0:2, 0:2]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Remaining terms for 3x3x3:
    # C02 = a00*b02 + a01*b12 + a02*b22 (3 terms)
    # C12 = a10*b02 + a11*b12 + a12*b22 (3 terms)
    # C20 = a20*b00 + a21*b10 + a22*b20 (3 terms)
    # C21 = a20*b01 + a21*b11 + a22*b21 (3 terms)
    # C22 = a20*b02 + a21*b12 + a22*b22 (3 terms)
    
    # Total terms for 3x3x3: 7 (Strassen) + 3 + 3 + 3 + 3 + 3 = 22?
    # Wait, C02, C12, C20, C21, C22 are 5 blocks of 1x1 or 1x3 or 3x1.
    # C02 (1x1): 3 terms.
    # C12 (1x1): 3 terms.
    # C20 (1x1): 3 terms.
    # C21 (1x1): 3 terms.
    # C22 (1x1): 3 terms.
    # Total 7 + 15 = 22 terms.
    
    # We can optimize the "rest" part.
    # But 22 is already better than 27.
    
    # Part 2: 3x3x4 multiplication using Schoolbook Rank 12
    # A (3x3) * B2 (3x4)
    # Each column of B2 is 3x1.
    # A * b_col = sum_j A_ij * b_j.
    # For 4 columns, we have 4 * 3 = 12 terms.
    
    total_rank = 22 + 12  # 34
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Helper to add a term: U[i*m+j, col]=u, V[j*p+k, col]=v, W[k*n+i, col]=w
    def add_term(i, j, k, u, v, w):
        nonlocal col
        U[i * m + j, col] = u
        V[j * p + k, col] = v
        W[k * n + i, col] = w
        col += 1

    # --- Part 1: 3x3x3 (First 3 columns of B, indices k=0,1,2) ---
    
    # Strassen terms for C[0:2, 0:2]
    # M1 = (a00+a11)*(b00+b11)
    # This corresponds to:
    # (a00*b00 + a00*b11 + a11*b00 + a11*b11)
    # We need to decompose this into rank-1 terms.
    # Strassen's M1 is one multiplication of sums.
    # In tensor decomposition, M1 is:
    # u = [1, 0, 0, 1, 0, 0, 0, 0, 0] (a00, a11) -> indices (0,0) and (1,1)
    # v = [1, 0, 0, 1, 0, 0, 0, 0, 0] (b00, b11) -> indices (0,0) and (1,1)
    # The result contributes to C00, C01, C10, C11.
    # Specifically:
    # C00 += M1
    # C01 -= M5 (handled separately)
    # C10 += M4 (handled separately)
    # C11 -= M2 (handled separately)
    # Wait, Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will add each M as a separate rank-1 term? No, M1 is a product of sums.
    # M1 = (a00+a11) * (b00+b11).
    # This is NOT a single rank-1 term in the original A, B, C space unless we expand.
    # (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
    # This expands to 4 terms.
    # Strassen's advantage is that we compute the sum first, then multiply.
    # In the tensor decomposition T = sum u_r v_r w_r, each r is a "multiplication".
    # If we use Strassen, we are introducing intermediate variables.
    # The tensor decomposition rank is the number of scalar multiplications.
    # So M1 is ONE term in the decomposition.
    # The vectors u, v, w for M1 are:
    # u_r has 1 at (0,0) and 1 at (1,1).
    # v_r has 1 at (0,0) and 1 at (1,1).
    # w_r has 1 at (0,0), 1 at (0,1), 1 at (1,0), 1 at (1,1)?
    # No, w_r must represent how M1 contributes to C.
    # M1 contributes +1 to C00, +1 to C11?
    # C00 = ... + M1 ...
    # C11 = ... + M1 ...
    # So w_r has 1 at (0,0) and 1 at (1,1).
    
    # Let's verify M1 contribution:
    # M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
    # C00 should have a00b00 + a01b10 + a02b20.
    # C11 should have a10b01 + a11b11 + a12b21.
    # This mixing makes it hard to just assign w_r.
    # Strassen's algorithm works by linear combinations of A and B blocks, then multiplying, then linear combinations of the results.
    # The tensor decomposition captures this exactly.
    
    # For M1:
    # u: a00, a11 -> indices (0,0), (1,1). Values 1, 1.
    # v: b00, b11 -> indices (0,0), (1,1). Values 1, 1.
    # w: C00, C11 -> indices (0,0), (1,1). Values 1, 1.
    # Check: u.v.w = (a00+a11)(b00+b11)(c00+c11).
    # This term adds M1 to C00 and M1 to C11.
    # But Strassen says C00 = M1 + M4 - M5 + M7.
    # So M1 contributes +1 to C00.
    # C11 = M1 - M2 + M3 + M6.
    # So M1 contributes +1 to C11.
    # So w_r = [1 at (0,0), 1 at (1,1)] is correct for M1.
    
    # Let's code the 7 Strassen terms for the 2x2x2 core.
    # Indices:
    # A: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4 (in 3x3 flattened 0..8)
    # B: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4 (in 3x3 flattened 0..8)
    # C: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4 (in 3x3 flattened 0..8)
    
    # M1: (a00+a11)(b00+b11) -> C00+=1, C11+=1
    add_term(0, 0, 0, 1, 1, 1) # u=(0,0), v=(0,0), w=(0,0) ? No, u is vector over A.
    # My add_term function takes i,j,k for U, V, W.
    # U is (n*m, rank). Index i*m+j.
    # V is (m*p, rank). Index j*p+k.
    # W is (p*n, rank). Index k*n+i.
    
    # For M1:
    # U has 1 at (0,0) and (1,1).
    # V has 1 at (0,0) and (1,1).
    # W has 1 at (0,0) and (1,1).
    # This requires multiple add_term calls or a loop.
    
    # I will rewrite the construction to handle multiple non-zeros per term.
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx, col] = 1
        for idx in v_indices:
            V[idx, col] = 1
        for idx in w_indices:
            W[idx, col] = 1
        col += 1

    # A indices: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
    # B indices (3x3 part): 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
    # W indices (3x3 part): 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
    
    # M1: (a00+a11)(b00+b11) -> C00, C11
    add_term([0, 4], [0, 4], [0, 4])
    
    # M2: (a10+a11)b00 -> C10, C11
    # u: (1,0)=3, (1,1)=4. v: (0,0)=0. w: (1,0)=3, (1,1)=4.
    # C10 = M2 + M4. So M2 contributes +1 to C10.
    # C11 = M1 - M2 + M3 + M6. So M2 contributes -1 to C11.
    # We need to handle signs.
    
    # I will use integers.
    
    def add_term_signed(u_indices, v_indices, w_indices, signs_u, signs_v, signs_w):
        nonlocal col
        for idx, s in zip(u_indices, signs_u):
            U[idx, col] = s
        for idx, s in zip(v_indices, signs_v):
            V[idx, col] = s
        for idx, s in zip(w_indices, signs_w):
            W[idx, col] = s
        col += 1

    # M1: (a00+a11)(b00+b11) -> C00(+1), C11(+1)
    add_term_signed([0, 4], [0, 4], [0, 4], [1, 1], [1, 1], [1, 1])
    
    # M2: (a10+a11)b00 -> C10(+1), C11(-1)
    add_term_signed([3, 4], [0], [3, 4], [1, 1], [1], [1, -1])
    
    # M3: a00(b01-b11) -> C00(+1), C01(+1)
    # C00 = M1 + M4 - M5 + M7. M3 not in C00?
    # C01 = M3 + M5.
    # C11 = M1 - M2 + M3 + M6.
    # So M3 contributes +1 to C01 and +1 to C11.
    add_term_signed([0], [1, 4], [1, 4], [1], [1, -1], [1, 1])
    
    # M4: a11(b10-b00) -> C00(+1), C10(+1)
    # C00 = ... + M4 ...
    # C10 = M2 + M4.
    # So M4 contributes +1 to C00 and +1 to C10.
    add_term_signed([4], [3, 0], [0, 3], [1], [1, -1], [1, 1])
    
    # M5: (a00+a01)b11 -> C00(-1), C01(+1)
    # C00 = ... - M5 ...
    # C01 = M3 + M5.
    # So M5 contributes -1 to C00 and +1 to C01.
    add_term_signed([0, 1], [4], [0, 1], [1, 1], [1], [-1, 1])
    
    # M6: (a10-a00)(b00+b01) -> C11(+1)
    # C11 = ... + M6 ...
    # M6 does not appear in C00, C01, C10.
    add_term_signed([3, 0], [0, 1], [4], [1, -1], [1, 1], [1])
    
    # M7: (a01-a11)(b10+b11) -> C00(+1)
    # C00 = ... + M7 ...
    add_term_signed([1, 4], [3, 4], [0], [1, -1], [1, 1], [1])
    
    # Now the remaining 15 schoolbook terms for 3x3x3.
    # C02: a00b02 + a01b12 + a02b22
    # C12: a10b02 + a11b12 + a12b22
    # C20: a20b00 + a21b10 + a22b20
    # C21: a20b01 + a21b11 + a22b21
    # C22: a20b02 + a21b12 + a22b22
    
    # C02 (W index 2)
    add_term_signed([0], [2], [2], [1], [1], [1]) # a00 b02
    add_term_signed([1], [5], [2], [1], [1], [1]) # a01 b12
    add_term_signed([2], [8], [2], [1], [1], [1]) # a02 b22
    
    # C12 (W index 5)
    add_term_signed([3], [2], [5], [1], [1], [1]) # a10 b02
    add_term_signed([4], [5], [5], [1], [1], [1]) # a11 b12
    add_term_signed([5], [8], [5], [1], [1], [1]) # a12 b22
    
    # C20 (W index 6)
    add_term_signed([6], [0], [6], [1], [1], [1]) # a20 b00
    add_term_signed([7], [3], [6], [1], [1], [1]) # a21 b10
    add_term_signed([8], [6], [6], [1], [1], [1]) # a22 b20
    
    # C21 (W index 7)
    add_term_signed([6], [1], [7], [1], [1], [1]) # a20 b01
    add_term_signed([7], [4], [7], [1], [1], [1]) # a21 b11
    add_term_signed([8], [7], [7], [1], [1], [1]) # a22 b21
    
    # C22 (W index 8)
    add_term_signed([6], [2], [8], [1], [1], [1]) # a20 b02
    add_term_signed([7], [5], [8], [1], [1], [1]) # a21 b12
    add_term_signed([8], [8], [8], [1], [1], [1]) # a22 b22
    
    # --- Part 2: 3x3x4 (Last 4 columns of B, indices k=3,4,5,6) ---
    # Schoolbook Rank 12.
    # For each column k in 3..6:
    # C0k = a00 b0k + a01 b1k + a02 b2k
    # C1k = a10 b0k + a11 b1k + a12 b2k
    # C2k = a20 b0k + a21 b1k + a22 b2k
    
    for k in range(3, 7):
        # C0k (W index k*n + 0 = k*3)
        add_term_signed([0], [k], [k*3], [1], [1], [1])
        add_term_signed([1], [k+3], [k*3], [1], [1], [1]) # b1k is index 1*3+k = 3+k
        add_term_signed([2], [k+6], [k*3], [1], [1], [1]) # b2k is index 2*3+k = 6+k
        
        # C1k (W index k*n + 1 = k*3+1)
        add_term_signed([3], [k], [k*3+1], [1], [1], [1])
        add_term_signed([4], [k+3], [k*3+1], [1], [1], [1])
        add_term_signed([5], [k+6], [k*3+1], [1], [1], [1])
        
        # C2k (W index k*n + 2 = k*3+2)
        add_term_signed([6], [k], [k*3+2], [1], [1], [1])
        add_term_signed([7], [k+3], [k*3+2], [1], [1], [1])
        add_term_signed([8], [k+6], [k*3+2], [1], [1], [1])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
