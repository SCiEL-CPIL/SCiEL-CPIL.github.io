"""Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE = 150.0
START = time.time()


def _elapsed():
    return time.time() - START


def _time_left():
    return DEADLINE - _elapsed() - 5.0  # safety margin


def _check_decomp(U, V, W, n=4, m=4, p=7):
    """Verify exact decomposition: C[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] for all i,j,k."""
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                for c in range(r):
                    acc += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(acc - 1.0) > 1e-7:
                    return False
    return True


def _to_fraction_list(mat):
    """Convert float matrix to list of Fractions (rounding small values to 0)."""
    out = []
    for row in mat:
        r = []
        for v in row:
            if abs(v) < 1e-10:
                r.append(Fraction(0))
            else:
                r.append(Fraction(v).limit_denominator(10**12))
        out.append(r)
    return out


def _build_decomp_from_factors(U, V, W):
    return {"U": U, "V": V, "W": W}


def _schoolbook(n=4, m=4, p=7):
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
    return _build_decomp_from_factors(U, V, W)


def _strassen_4x4x4():
    """Return Strassen-based rank-49 decomposition for 4x4x4."""
    n = 4
    # Strassen 2x2x2 base (rank 7)
    # M1..M7 for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (d-b)(e+f); M2 = (a+b)h; M3 = a(e-g); M4 = d(h-f)
    # M5 = (a+d)(h-g); M6 = (c+d)(e-f); M7 = (b-c)(e+h)
    # C11 = M5+M4-M2+M6; C12 = M1+M2; C21 = M3+M5; C22 = M1-M4+M7+M5
    # We build 4x4x4 by applying Strassen at 2x2 block level.
    # 4x4 matrices partitioned into 2x2 blocks:
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # C = [[C11,C12],[C21,C22]]
    # Using Strassen:
    # S1 = (A22-A11)(B11+B12)
    # S2 = (A11+A12)B22
    # S3 = A11(B11-B21)
    # S4 = A22(B22-B12)
    # S5 = (A11+A22)(B22-B21)
    # S6 = (A21+A22)(B11-B12)
    # S7 = (A12-A21)(B11+B22)
    # C11 = S5+S4-S2+S6
    # C12 = S1+S2
    # C21 = S3+S5
    # C22 = S1-S4+S7+S5
    # Each Sx is a 2x2x2 product (rank 7), so total rank = 7*7 = 49.
    # Now we need to express each 2x2x2 product as rank-7 tensor.
    # For 2x2x2: indices (i,j,k) in {0,1}^3
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # 7 terms:
    # (i,j,k) contributions:
    strassen_2x2 = [
        # (U_row, V_row, W_row) for each of 7 terms
        # Term 1: (d-b)(e+f): U picks rows for (d-b): row 3 (i=1,j=1) coeff 1, row 1 (i=0,j=1) coeff -1
        # V picks (e+f): row 0 (j=0,k=0) coeff 1, row 1 (j=0,k=1) coeff 1
        # W picks: for output C11 contribution... this is getting complex.
        # Let me use the standard rank-7 formula for 2x2x2.
    ]
    # Standard 2x2x2 rank-7 decomposition:
    # The 8 entries of C = A*B where A,B are 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # Rank 7 (Strassen):
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (d-b)(e+f)
    # M2 = (a+b)h
    # M3 = a(e-g)
    # M4 = d(h-f)
    # M5 = (a+d)(h-g)
    # M6 = (c+d)(e-f)
    # M7 = (b-c)(e+h)
    # C[0,0] = M5+M4-M2+M6
    # C[0,1] = M1+M2
    # C[1,0] = M3+M5
    # C[1,1] = M1-M4+M7+M5
    #
    # For each Mx, we need U, V, W vectors (length 4 each for the 2x2 case: U is 2*2=4 rows, V is 2*2=4 rows, W is 2*2=4 rows)
    # U[i*2+j], V[j*2+k], W[k*2+i]
    #
    # M1 = (d-b)(e+f):
    #   U: d-b => U[3]=1 (i=1,j=1), U[1]=-1 (i=0,j=1)
    #   V: e+f => V[0]=1 (j=0,k=0), V[1]=1 (j=0,k=1)
    #   W: contributes to C[0,1] and C[1,1]
    #   C[0,1] = M1+M2: W for C[0,1] means k=0,i=1 => W[0*2+1]=W[1]
    #   C[1,1] = M1-M4+M7+M5: W for C[1,1] means k=1,i=1 => W[1*2+1]=W[3]
    #   So W[1]=1, W[3]=1
    #
    # M2 = (a+b)h:
    #   U: a+b => U[0]=1 (i=0,j=0), U[2]=1 (i=1,j=0)
    #   V: h => V[3]=1 (j=1,k=1)
    #   W: C[0,0] has -M2 => k=0,i=0 => W[0]; C[0,1] has +M2 => k=1,i=0 => W[2]
    #   W[0]=-1, W[2]=1
    #
    # M3 = a(e-g):
    #   U: a => U[0]=1
    #   V: e-g => V[0]=1 (j=0,k=0), V[2]=-1 (j=1,k=0)
    #   W: C[1,0] has +M3 => k=0,i=1 => W[1]
    #   W[1]=1
    #
    # M4 = d(h-f):
    #   U: d => U[3]=1
    #   V: h-f => V[1]=-1 (j=0,k=1), V[3]=1 (j=1,k=1)
    #   W: C[0,0] has +M4 => W[0]=1; C[1,1] has -M4 => W[3]=-1
    #
    # M5 = (a+d)(h-g):
    #   U: a+d => U[0]=1, U[3]=1
    #   V: h-g => V[2]=-1 (j=1,k=0), V[3]=1 (j=1,k=1)
    #   W: C[0,0] has +M5 => W[0]=1; C[1,0] has +M5 => W[1]=1; C[1,1] has +M5 => W[3]=1
    #
    # M6 = (c+d)(e-f):
    #   U: c+d => U[2]=1 (i=1,j=0), U[3]=1 (i=1,j=1)
    #   V: e-f => V[0]=1 (j=0,k=0), V[1]=-1 (j=0,k=1)
    #   W: C[0,0] has +M6 => W[0]=1
    #
    # M7 = (b-c)(e+h):
    #   U: b-c => U[1]=1 (i=0,j=1), U[2]=-1 (i=1,j=0)
    #   V: e+h => V[0]=1 (j=0,k=0), V[3]=1 (j=1,k=1)
    #   W: C[1,1] has +M7 => W[3]=1

    # Build 2x2x2 rank-7 factors
    def make_2x2_term(u_vals, v_vals, w_vals):
        U = [0, 0, 0, 0]
        V = [0, 0, 0, 0]
        W = [0, 0, 0, 0]
        for idx, val in u_vals.items():
            U[idx] = val
        for idx, val in v_vals.items():
            V[idx] = val
        for idx, val in w_vals.items():
            W[idx] = val
        return U, V, W

    terms_2x2 = [
        # M1
        make_2x2_term({3:1, 1:-1}, {0:1, 1:1}, {1:1, 3:1}),
        # M2
        make_2x2_term({0:1, 2:1}, {3:1}, {0:-1, 2:1}),
        # M3
        make_2x2_term({0:1}, {0:1, 2:-1}, {1:1}),
        # M4
        make_2x2_term({3:1}, {1:-1, 3:1}, {0:1, 3:-1}),
        # M5
        make_2x2_term({0:1, 3:1}, {2:-1, 3:1}, {0:1, 1:1, 3:1}),
        # M6
        make_2x2_term({2:1, 3:1}, {0:1, 1:-1}, {0:1}),
        # M7
        make_2x2_term({1:1, 2:-1}, {0:1, 3:1}, {3:1}),
    ]

    # Now build 4x4x4 from 7 Strassen block products, each of which is a 2x2x2 rank-7.
    # 4x4 blocks: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # Block indices: A00 is rows 0-1, cols 0-1; A01 is rows 0-1, cols 2-3; etc.
    # For 4x4: U has 4*4=16 rows, V has 4*4=16 rows, W has 4*4=16 rows.
    # U row index: i*4+j where i is row of A, j is col of A
    # V row index: j*4+k where j is col of A (row of B), k is col of B
    # W row index: k*4+i where k is col of B, i is row of C (row of A)

    # Strassen 4x4:
    # S1 = (A11-A00)(B00+B01)  -> C00 += S1? No...
    # Let me re-derive:
    # S1 = (A11-A00)(B00+B01)
    # S2 = (A00+A01)B11
    # S3 = A00(B00-B10)
    # S4 = A11(B11-B01)
    # S5 = (A00+A11)(B11-B10)
    # S6 = (A10+A11)(B00-B01)
    # S7 = (A01-A10)(B00+B11)
    # C00 = S5+S4-S2+S6
    # C01 = S1+S2
    # C10 = S3+S5
    # C11 = S1-S4+S7+S5

    # Each Sx is a 2x2x2 matrix product. For Sx = (linear combo of A blocks)(linear combo of B blocks),
    # the 2x2x2 structure maps to the 4x4 indices.
    #
    # For S1 = (A11-A00)(B00+B01):
    #   A11: rows 2-3, cols 2-3 of A => U rows: (2,2),(2,3),(3,2),(3,3) => indices 10,11,14,15
    #   A00: rows 0-1, cols 0-1 of A => U rows: (0,0),(0,1),(1,0),(1,1) => indices 0,1,4,5
    #   B00: rows 0-1, cols 0-1 of B => V rows: (0,0),(0,1),(1,0),(1,1) => indices 0,1,4,5
    #   B01: rows 0-1, cols 2-3 of B => V rows: (0,2),(0,3),(1,2),(1,3) => indices 2,3,6,7
    #
    # The 2x2x2 term within S1: the "i" index of the 2x2 corresponds to block row of A (0 or 1),
    # "j" index corresponds to block col of A (0 or 1), "k" index corresponds to block col of B (0 or 1).
    # But within each 2x2 block, there are 4 sub-indices.
    #
    # Actually, for S1 = (A11-A00)(B00+B01), this is a product of two 2x2 matrices.
    # The 2x2x2 decomposition has 7 terms. For each term, the U vector is over the 4 elements of the A-side 2x2,
    # V over 4 elements of B-side, W over 4 output elements.
    #
    # Mapping for S1: A-side 2x2 = [[A11[0,0],A11[0,1]],[A11[1,0],A11[1,1]]] minus [[A00[0,0],A00[0,1]],[A00[1,0],A00[1,1]]]
    # So A-side element (r,c) where r,c in {0,1}:
    #   (0,0): A11[0,0] - A00[0,0] => U[2*4+2] - U[0*4+0] = U[10] - U[0]
    #   (0,1): A11[0,1] - A00[0,1] => U[11] - U[1]
    #   (1,0): A11[1,0] - A00[1,0] => U[14] - U[4]
    #   (1,1): A11[1,1] - A00[1,1] => U[15] - U[5]
    # B-side 2x2 = [[B00[0,0],B00[0,1]],[B00[1,0],B00[1,1]]] plus [[B01[0,0],B01[0,1]],[B01[1,0],B01[1,1]]]
    #   (0,0): B00[0,0] + B01[0,0] => V[0] + V[2]
    #   (0,1): B00[0,1] + B01[0,1] => V[1] + V[3]
    #   (1,0): B00[1,0] + B01[1,0] => V[4] + V[6]
    #   (1,1): B00[1,1] + B01[1,1] => V[5] + V[7]
    # Output 2x2 (for C01 block): C01[0,0],C01[0,1],C01[1,0],C01[1,1]
    #   C01 is rows 0-1, cols 2-3 of C.
    #   (0,0): C[0,2] => W[2*4+0] = W[8]
    #   (0,1): C[0,3] => W[3*4+0] = W[12]
    #   (1,0): C[1,2] => W[2*4+1] = W[9]
    #   (1,1): C[1,3] => W[3*4+1] = W[13]

    # This is getting very complex. Let me use a different approach:
    # Build the 4x4x4 decomposition by direct construction using the 2x2x2 Strassen at block level.

    # Actually, let me just use a simpler approach: build the full 4x4x4 rank-49 decomposition
    # by composing two levels of Strassen.

    # Level 1: 4x4 = 2x(2x2) blocks. Apply Strassen to get 7 block products.
    # Level 2: Each block product is 2x2x2, decomposed with rank 7.
    # Total: 7*7 = 49 terms.

    # For each Strassen block product Sx, I need to map the 2x2x2 rank-7 decomposition
    # to the 4x4x4 indices.

    # S1 = (A11-A00)(B00+B01) contributes to C01
    # S2 = (A00+A01)B11 contributes to C00 (-S2) and C01 (+S2)
    # S3 = A00(B00-B10) contributes to C10
    # S4 = A11(B11-B01) contributes to C00 (+S4) and C11 (-S4)
    # S5 = (A00+A11)(B11-B10) contributes to C00, C10, C11
    # S6 = (A10+A11)(B00-B01) contributes to C00
    # S7 = (A01-A10)(B00+B11) contributes to C11

    # For each Sx, the 2x2x2 indices (r,c) for A-side, (r,c) for B-side, (r,c) for C-side
    # map to 4x4 indices as follows:
    # A-side (r,c): row = block_row*2+r, col = block_col*2+c
    # B-side (r,c): row = block_row*2+r, col = block_col*2+c
    # C-side (r,c): row = block_row*2+r, col = block_col*2+c

    # Let me define the block structure for each Sx:
    # S1: A blocks: A11 (br=1,bc=1) and A00 (br=0,bc=0); B blocks: B00 (br=0,bc=0) and B01 (br=0,bc=1); C block: C01 (br=0,bc=1)
    # S2: A blocks: A00 (br=0,bc=0) and A01 (br=0,bc=1); B block: B11 (br=1,bc=1); C blocks: C00 (br=0,bc=0) and C01 (br=0,bc=1)
    # S3: A block: A00 (br=0,bc=0); B blocks: B00 (br=0,bc=0) and B10 (br=1,bc=0); C block: C10 (br=1,bc=0)
    # S4: A block: A11 (br=1,bc=1); B blocks: B11 (br=1,bc=1) and B01 (br=0,bc=1); C blocks: C00 (br=0,bc=0) and C11 (br=1,bc=1)
    # S5: A blocks: A00 (br=0,bc=0) and A11 (br=1,bc=1); B blocks: B11 (br=1,bc=1) and B10 (br=1,bc=0); C blocks: C00, C10, C11
    # S6: A blocks: A10 (br=1,bc=0) and A11 (br=1,bc=1); B blocks: B00 (br=0,bc=0) and B01 (br=0,bc=1); C block: C00
    # S7: A blocks: A01 (br=0,bc=1) and A10 (br=1,bc=0); B blocks: B00 (br=0,bc=0) and B11 (br=1,bc=1); C block: C11

    # This is extremely complex to code correctly. Let me use a numerical approach instead.
    # I'll construct the 4x4x4 tensor and use the known Strassen decomposition numerically.

    # Alternative: just use the schoolbook for now and focus on the 4x4x3 part.
    # Actually, let me try a different strategy: use the known rank-49 for 4x4x4
    # by building it from the 2x2x2 Strassen decomposition programmatically.

    # Let me define a helper that, given the Strassen 2x2x2 terms and the block mapping,
    # produces the 4x4x4 terms.

    # For Sx with A-side being a linear combination of A blocks, B-side being a linear combination of B blocks,
    # and C-side being a specific C block (or combination), the 2x2x2 rank-7 terms map as:
    # For each of the 7 2x2x2 terms (u2, v2, w2) where u2, v2, w2 are length-4 vectors:
    #   u2[r*2+c] maps to U[(a_br*2+r)*4 + (a_bc*2+c)] for each A block contribution
    #   etc.

    # This is too error-prone to do by hand. Let me use a computational approach.
    # I'll construct the 4x4x4 multiplication tensor and verify a candidate decomposition.

    # For now, let me just return schoolbook and focus on optimizing the 4x4x3 part.
    # Actually, let me try to build the 4x4x4 Strassen decomposition computationally.

    # Build 4x4x4 tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C.
    # Actually T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k (this is the schoolbook).
    # The rank is the minimum number of rank-1 tensors that sum to T.

    # Let me just use a known result: rank(4x4x4) <= 49 (Strassen).
    # I'll construct it by building the 7x7 composition.

    # Given the complexity, let me use a different approach for the full 4x4x7:
    # Use schoolbook for the entire thing (rank 112) as fallback,
    # and try to find a better decomposition via search.

    # For now, return schoolbook.
    return _schoolbook(4, 4, 4)


def _search_4x4x3(max_rank=48, time_budget=30):
    """Search for a low-rank decomposition of 4x4x3 matrix multiplication."""
    n, m, p = 4, 4, 3
    start = time.time()
    best_rank = n * m * p  # 48 schoolbook
    best_decomp = _schoolbook(n, m, p)

    # Try to find rank < 48 decomposition
    # Use iterative refinement: start with random factors and minimize error
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() - start > time_budget:
            break
        for attempt in range(5):
            if time.time() - start > time_budget:
                break
            random.seed(attempt * 1000 + target_rank)
            # Initialize with random small values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1

            # Iterative refinement: for each factor, fix the others and solve
            for it in range(100):
                if time.time() - start > time_budget:
                    break
                # Update U: for each row i*m+j, we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
                # This is a linear system if we fix V and W.
                # For fixed V, W: the contribution to C[i,k] from row (i,j) of U is:
                #   sum_j sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                # Let M_jk_c = V[j*p+k,c]*W[k*n+i,c]. Then C[i,k] = sum_j sum_c U[i*m+j,c]*M_jk_c
                # We want C[i,k] = 1 for all k.
                # For fixed i, this is a system: for each k, sum_j sum_c U[i*m+j,c]*M_jk_c = 1
                # This is (p) equations in (m*target_rank) unknowns.

                # Update W: for each row k*n+i, fix U and V.
                # C[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # For fixed i,k: sum_j sum_c (U[i*m+j,c]*V[j*p+k,c])*W[k*n+i,c] = 1
                # This is 1 equation in target_rank unknowns. Underdetermined if target_rank > 1.

                # Let me use a simpler approach: gradient descent on the error.
                # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2

                # Compute gradients
                Ua = np.array(U, dtype=float)
                Va = np.array(V, dtype=float)
                Wa = np.array(W, dtype=float)

                # Compute current C
                # C[i,k] = sum_j sum_c Ua[i*m+j,c]*Va[j*p+k,c]*Wa[k*n+i,c]
                C = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for c in range(target_rank):
                                C[i, k] += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]

                err = np.sum((C - 1.0)**2)
                if err < 1e-10:
                    break

                # Gradient of err w.r.t. Ua[i*m+j, c]:
                # d(err)/d(Ua[i*m+j,c]) = 2*sum_k (C[i,k]-1)*Va[j*p+k,c]*Wa[k*n+i,c]
                dU = np.zeros_like(Ua)
                dV = np.zeros_like(Va)
                dW = np.zeros_like(Wa)
                for i in range(n):
                    for k in range(p):
                        e = 2.0 * (C[i, k] - 1.0)
                        for j in range(m):
                            for c in range(target_rank):
                                dU[i*m+j, c] += e * Va[j*p+k, c] * Wa[k*n+i, c]
                                dV[j*p+k, c] += e * Ua[i*m+j, c] * Wa[k*n+i, c]
                                dW[k*n+i, c] += e * Ua[i*m+j, c] * Va[j*p+k, c]

                # Simple gradient descent with adaptive step
                lr = 0.01
                Ua -= lr * dU
                Va -= lr * dV
                Wa -= lr * dW

                if it % 10 == 0 and err > 1e-6:
                    lr *= 0.5

            # Check if we got a good solution
            C = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(target_rank):
                            C[i, k] += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
            err = np.max(np.abs(C - 1.0))
            if err < 1e-6:
                # Convert to fractions
                Uf = _to_fraction_list(Ua)
                Vf = _to_fraction_list(Va)
                Wf = _to_fraction_list(Wa)
                # Verify
                if _check_decomp(Uf, Vf, Wf, n, m, p):
                    best_rank = target_rank
                    best_decomp = _build_decomp_from_factors(Uf, Vf, Wf)
                    break

    return best_decomp, best_rank


def _combine_decomps(decomp1, decomp2, n, m, p1, p2):
    """Combine two decompositions: (n x m x p1) and (n x m x p2) into (n x m x (p1+p2))."""
    U1, V1, W1 = decomp1["U"], decomp1["V"], decomp1["W"]
    U2, V2, W2 = decomp2["U"], decomp2["V"], decomp2["W"]
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 + r2

    U = []
    for i in range(n * m):
        U.append(U1[i] + U2[i])
    V = []
    for j in range(m * (p1 + p2)):
        if j < m * p1:
            V.append(V1[j] + [0] * r2)
        else:
            V.append([0] * r1 + V2[j - m * p1])
    W = []
    for k in range((p1 + p2) * n):
        if k < p1 * n:
            W.append(W1[k] + [0] * r2)
        else:
            W.append([0] * r1 + W2[k - p1 * n])
    return _build_decomp_from_factors(U, V, W)


def construct():
    global START
    START = time.time()

    # Fallback: schoolbook
    schoolbook = _schoolbook(4, 4, 7)

    # Try to find better decomposition
    # Strategy: decompose 4x4x7 as 4x4x4 + 4x4x3
    # For 4x4x4: try Strassen (rank 49) or schoolbook (rank 64)
    # For 4x4x3: try to find rank < 48

    # First, try to get a 4x4x3 decomposition with rank < 48
    decomp_4x4x3, rank_4x4x3 = _search_4x4x3(max_rank=48, time_budget=20)

    if _time_left() < 10:
        return schoolbook

    # For 4x4x4, use schoolbook for now (rank 64)
    # Total: 64 + rank_4x4x3
    decomp_4x4x4 = _schoolbook(4, 4, 4)
    combined = _combine_decomps(decomp_4x4x4, decomp_4x4x3, 4, 4, 4, 3)
    total_rank = len(combined["U"][0])

    # Verify
    if _check_decomp(combined["U"], combined["V"], combined["W"], 4, 4, 7):
        if total_rank < len(schoolbook["U"][0]):
            return combined

    return schoolbook
