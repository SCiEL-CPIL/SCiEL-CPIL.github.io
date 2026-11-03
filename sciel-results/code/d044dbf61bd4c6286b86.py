"""Strassen-style recursive block decomposition for 4x4 x 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7

    # We will build a list of rank-1 terms. Each term is (u_vec, v_vec, w_vec)
    # where u_vec has length n*m, v_vec has length m*p, w_vec has length p*n.
    # The decomposition satisfies: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]
    # i.e., the (i,j,k) entry of the multiplication tensor = sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c}

    # We use a recursive Strassen scheme.
    # Represent a "block multiplication" problem as (n, m, p) where n,m,p are block sizes.
    # We build terms recursively.

    # A term at the leaf level is a scalar multiplication: pick specific (i,j,k) indices.
    # We'll build a list of (i, j, k) leaves, each contributing a rank-1 term with:
    #   U[i*m + j] = 1, V[j*p + k] = 1, W[k*n + i] = 1
    # But Strassen combines them with coefficients.

    # Instead of tracking coefficients per leaf, we track for each rank-1 "column" c:
    #   a set of (i, j, k, coeff) contributions such that the tensor entry (i,j,k)
    #   = sum over columns c of U[(i,j),c] * V[(j,k),c] * W[(k,i),c].

    # This is complex. Let me use a cleaner formulation:
    # We build the multiplication tensor T[i,j,k] = delta (it's 1 for all valid i,j,k in the
    # "multiplication" sense). Actually the multiplication tensor has T[i,j,k] = 1 for all
    # i in [0,n), j in [0,m), k in [0,p). The decomposition is:
    #   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).

    # So we need: for every (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.

    # Strassen gives us a set of "product terms" each of which is a product of a linear
    # form in U-entries, a linear form in V-entries, and a linear form in W-entries.

    # Let me implement Strassen recursively, producing for the whole problem a list of
    # (u_coeffs, v_coeffs, w_coeffs) where each is a dict mapping index->coefficient,
    # and the product of the three linear forms sums to T.

    # Actually, the cleanest: Strassen's 7 products for 2x2x2. Each product M_r is a linear
    # combination of entries of A (2x2), B (2x2). Then C entries are linear combos of M_r.
    # In tensor decomposition terms, each M_r gives a rank-1 term:
    #   (sum_{i,j} a_{i,j}^r * E_{i,j}) * (sum_{j,k} b_{j,k}^r * E_{j,k}) * (sum_{k,i} c_{k,i}^r * E_{k,i})
    # where E_{i,j} is the basis for A_{i,j}, etc. And the C part: c_{k,i}^r is the coefficient
    # such that C = sum_r M_r * (linear form in C). Wait, I need to be careful.

    # Standard Strassen: 
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #   C11 = M1+M4-M5+M7
    #   C12 = M3+M5
    #   C21 = M2+M4
    #   C22 = M1-M2+M3+M6
    #
    # In tensor terms, the multiplication tensor for 2x2x2 is:
    #   T = sum_{i,j,k} e_i^A e_j^B e_k^C  (where e_i^A picks A's row i, etc.)
    # Actually the tensor is: T[A,B,C] = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}... no.
    #
    # Let me use the standard: the matrix multiplication tensor m_{n} has entries
    #   m_{n}[(i,j), (j,k), (k,i)] = 1  (for the "multiplication" pairing)
    # A decomposition is a set of rank-1 tensors:
    #   m_n = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n^2} (A-space), v_c in R^{n^2} (B-space), w_c in R^{n^2} (C-space).
    #
    # For Strassen, each M_r corresponds to a rank-1 term:
    #   u_r = vector in A-space with the coefficients from the A-part of M_r
    #   v_r = vector in B-space with the coefficients from the B-part of M_r
    #   w_r = vector in C-space with the coefficients from the C-part (the linear form that M_r contributes to)
    #
    # Specifically, M_r = (sum_{i,j} a^r_{i,j} A_{i,j}) * (sum_{j,k} b^r_{j,k} B_{j,k})
    # and M_r contributes to C via: C += w_r * M_r, where w_r is the C-coefficient vector.
    # So the rank-1 term is: (a^r in A-space) ⊗ (b^r in B-space) ⊗ (w_r in C-space).
    # And sum_r (a^r ⊗ b^r ⊗ w_r) = m_2.

    # Let me define Strassen's 7 terms for 2x2x2 with explicit coefficient vectors.
    # A-space indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B-space indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C-space indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

    # M1: A-part = A11+A22 = (1,0,0,1), B-part = B11+B22 = (1,0,0,1)
    #     C-contribution: C11 += M1, C22 += M1 => w = (1,0,0,1)
    # M2: A-part = A21+A22 = (0,0,1,1), B-part = B11 = (1,0,0,0)
    #     C-contribution: C21 += M2, C22 -= M2 => w = (0,0,1,-1)
    # M3: A-part = A11 = (1,0,0,0), B-part = B21-B22 = (0,1,1,-1)
    #     C-contribution: C12 += M3, C22 += M3 => w = (0,1,0,1)
    # M4: A-part = A22 = (0,0,0,1), B-part = B12-B11 = (-1,1,0,0)
    #     C-contribution: C11 += M4, C21 += M4 => w = (1,0,1,0)
    # M5: A-part = A11+A12 = (1,1,0,0), B-part = B22 = (0,0,0,1)
    #     C-contribution: C11 -= M5, C12 += M5 => w = (-1,1,0,0)
    # M6: A-part = A21-A11 = (-1,0,1,0), B-part = B11+B12 = (1,1,0,0)
    #     C-contribution: C22 += M6 => w = (0,0,0,1)
    # M7: A-part = A12-A22 = (0,1,0,-1), B-part = B21+B22 = (0,0,1,1)
    #     C-contribution: C11 += M7 => w = (1,0,0,0)

    # Let me verify: sum of (a_r ⊗ b_r ⊗ w_r) should equal m_2.
    # m_2 has entries m_2[(i,j),(j,k),(k,i)] = 1.
    # In flattened form, m_2[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(k,i) for some i,j,k.
    # i.e., m_2 is 1 at positions where the three indices are "compatible".

    # This is getting complex. Let me just build it recursively and verify numerically.

    # I'll represent the problem as a tree. Each node is a "block multiplication" with
    # block dimensions (n, m, p). A leaf is a scalar multiplication (1x1x1).
    # Strassen splits a 2x2x2 into 7 sub-problems of 1x1x1 (for the base case).
    # For general (n,m,p) that are powers of 2 (or mixed), I split into 2x2 blocks.

    # Given the complexity, let me just hardcode the full recursive Strassen for 4x4x7.

    # Approach: 
    # Level 0: 4x4x7. Split A into 2x2 blocks (each 2x2), B into 2x1 blocks (left: 2x2, right: 2x1).
    #   C = [C_left(4x4), C_right(4x3)]
    #   C_left = A * B_left  (4x4 * 4x4) -> 2x2 blocks of 2x2, use Strassen: 7 sub-multiplications of 2x2x2
    #   C_right = A * B_right (4x4 * 4x3) -> 2x1 blocks: A is 2x2 blocks of 2x2, B_right is 2x1 blocks of 2x1
    #     C_right has 2x1 = 2 blocks (top 2x1, bottom 2x1), each = 2x2 * 2x1 -> 4 products each (since 2x2*2x1 = 4 scalar products)
    #     So C_right: 2 blocks * 4 = 8 scalar products? No wait.
    #
    # Let me be very explicit with indices.
    # A is 4x4. A[i,j] for i in 0..3, j in 0..3.
    # B is 4x7. B[j,k] for j in 0..3, k in 0..6.
    # C[i,k] = sum_j A[i,j] B[j,k].

    # Partition:
    # A = [[A00, A01], [A10, A11]] where A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B = [B_left, B_right] where B_left = B[:,0:4] (4x4), B_right = B[:,4:7] (4x3)
    # C = [C_left, C_right] where C_left = C[:,0:4] (4x4), C_right = C[:,4:7] (4x3)
    #
    # C_left = A * B_left. Both 4x4. Partition into 2x2 blocks of 2x2:
    #   B_left = [[B00, B01], [B10, B11]] where B00 = B[0:2,0:2], etc.
    #   C_left = [[C00, C01], [C10, C11]] where C00 = C[0:2,0:2], etc.
    #   Strassen: 7 products M1..M7, each a 2x2 * 2x2 multiplication.
    #   Each 2x2*2x2 = 4 scalar products. So 7*4 = 28 scalar products for C_left.
    #
    # C_right = A * B_right. A is 4x4, B_right is 4x3.
    #   Partition A into 2x2 blocks of 2x2 (same as above).
    #   Partition B_right into 2x1 blocks: B_right = [[BR00], [BR10]] where BR00 = B[0:2,4:7] (2x3), BR10 = B[2:4,4:7] (2x3).
    #   C_right = [[CR00], [CR10]] where CR00 = C[0:2,4:7] (2x3), CR10 = C[2:4,4:7] (2x3).
    #   CR00 = A00*BR00 + A01*BR10  (2x2 * 2x3 + 2x2 * 2x3)
    #   CR10 = A10*BR00 + A11*BR10
    #   Each 2x2*2x3 = 6 scalar products. So CR00: 12, CR10: 12. Total C_right: 24 scalar products.
    #
    # Total scalar products: 28 + 24 = 52. That's the rank!
    #
    # Wait, but each "scalar product" here is a product of one A-entry, one B-entry, contributing to one C-entry.
    # In the tensor decomposition, each scalar product (i,j,k) gives a rank-1 term:
    #   U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # But Strassen uses linear combinations, so the coefficients aren't just 1.
    #
    # Hmm, but actually for the "scalar product" level, each term IS a single (i,j,k) with coefficient 1.
    # The Strassen linear combinations happen at the 2x2 level, where M_r is a linear combo of A-entries times a linear combo of B-entries.
    #
    # So the rank-1 terms are at the 2x2 level: each M_r is ONE rank-1 term in the 2x2x2 tensor, with coefficient vectors (a^r, b^r, w^r).
    # And the 2x2x2 tensor itself decomposes into 7 such terms.
    #
    # So for C_left (4x4*4x4), we have 7 "2x2 Strassen terms", but each 2x2 Strassen term is itself a 2x2x2 multiplication, which has rank 7.
    # So C_left: 7 (top-level Strassen) * 7 (inner Strassen) = 49 rank-1 terms.
    #
    # For C_right (4x4*4x3), we have 4 "2x2x2x1" sub-multiplications (A00*BR00, A01*BR10, A10*BR00, A11*BR10).
    # Each 2x2*2x3 = 2x2 * (2x1 + 2x1) = 2 * (2x2*2x1).
    # Each 2x2*2x1: this is a 2x2 matrix times a 2x1 matrix. Can we use Strassen? 
    #   2x2*2x1: C (2x1) = A (2x2) * B (2x1).
    #   C0 = A00*B0 + A01*B1
    #   C1 = A10*B0 + A11*B1
    #   This is 4 scalar products (each 1x1*1x1). No Strassen benefit for 2x2*2x1 (it's already minimal at 4? or can we do 3?).
    #   Actually, 2x2*2x1 is equivalent to 2x2*2x2 with the second column of B being zero. The rank of 2x2*2x1 tensor is 4 (I believe).
    #   So each 2x2*2x1 = 4 rank-1 terms.
    #   Each 2x2*2x3 = 2 * (2x2*2x1) = 8 rank-1 terms.
    #   C_right: 4 sub-multiplications * 8 = 32 rank-1 terms.
    #
    # Total: 49 + 32 = 81 rank-1 terms.
    #
    # That's better than schoolbook 112! Let me go with this.
    #
    # Actually wait, let me reconsider. For 2x2*2x3, is it really 8? 
    #   2x2 * 2x3: A is 2x2, B is 2x3, C is 2x3.
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    #   For each (i,k), that's 2 products. 2*3 = 6 (i,k) pairs, so 12 scalar products? No.
    #   Wait, C is 2x3, so 6 entries. Each entry C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]. So 2 products per entry, 6 entries = 12 scalar products.
    #   Hmm, I miscounted. Let me redo.
    #
    # 2x2 * 2x3: A (2x2), B (2x3), C (2x3).
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #   Total: 12 scalar products. Rank = 12 (schoolbook for 2x2x3).
    #
    # Can we do better? The rank of the 2x2x3 multiplication tensor... I'm not sure. Let's assume schoolbook 12 for now.
    #
    # So C_right: 4 sub-multiplications (A00*BR00, A01*BR10, A10*BR00, A11*BR10), each 2x2*2x3 = 12 scalar products.
    #   C_right: 4 * 12 = 48 scalar products.
    #
    # C_left: 7 Strassen terms, each 2x2x2 = 7 scalar products (using Strassen for 2x2x2).
    #   C_left: 7 * 7 = 49 scalar products.
    #
    # Total: 49 + 48 = 97. Still better than 112!
    #
    # Hmm wait, but for C_left, I'm using Strassen for the 2x2x2 sub-multiplications. Each 2x2x2 Strassen term has rank 7, and there are 7 of them, so 49. Yes.
    #
    # For C_right, I'm using schoolbook for 2x2x3 (rank 12 each), 4 of them = 48.
    #
    # Total rank = 97.
    #
    # Can I do better on C_right? What if I use Strassen-like for 2x2x3? 
    #   2x2*2x3: think of B as [B_col0, B_col1, B_col2] where each B_col is 2x1.
    #   C = A * B = [A*B_col0, A*B_col1, A*B_col2].
    #   Each A*B_col is 2x2*2x1 = 4 scalar products (schoolbook). 3 columns = 12.
    #   Or use Strassen for 2x2*2x1? 2x2*2x1: C (2x1) = A (2x2)*B (2x1).
    #     C0 = A00*B0 + A01*B1
    #     C1 = A10*B0 + A11*B1
    #     4 scalar products. Is there a rank-3 scheme? Probably not (rank of 2x2x1 tensor is 4).
    #   So 2x2x3 is at least 12. Let's stick with 12.
    #
    # So total rank = 97.
    #
    # Let me now implement this. I'll build the rank-1 terms explicitly.

    # I'll create a list of terms. Each term is a dict with 'A' (dict from (i,j)->coeff), 
    # 'B' (dict from (j,k)->coeff), 'C' (dict from (k,i)->coeff).
    # Then I'll convert to U, V, W matrices.

    terms = []  # list of (A_dict, B_dict, C_dict)

    # Helper: add a scalar product term (i,j,k) with coefficient 1.
    # This contributes to A[i,j], B[j,k], C[k,i].
    def add_scalar(i, j, k, coeff=1):
        if coeff == 0:
            return
        A_d = {(i, j): coeff}
        B_d = {(j, k): coeff}
        C_d = {(k, i): coeff}
        terms.append((A_d, B_d, C_d))

    # Helper: add a Strassen 2x2x2 term. The 2x2 block is at offset (i0, j0) in A, (j0, k0) in B, (k0, i0) in C.
    # The 7 Strassen products, each a linear combo.
    # Strassen coefficients (for 2x2x2):
    # M1: A=(1,0,0,1), B=(1,0,0,1), C=(1,0,0,1)
    # M2: A=(0,0,1,1), B=(1,0,0,0), C=(0,0,1,-1)
    # M3: A=(1,0,0,0), B=(0,1,1,-1), C=(0,1,0,1)
    # M4: A=(0,0,0,1), B=(-1,1,0,0), C=(1,0,1,0)
    # M5: A=(1,1,0,0), B=(0,0,0,1), C=(-1,1,0,0)
    # M6: A=(-1,0,1,0), B=(1,1,0,0), C=(0,0,0,1)
    # M7: A=(0,1,0,-1), B=(0,0,1,1), C=(1,0,0,0)
    # where A=(A00,A01,A10,A11), B=(B00,B01,B10,B11), C=(C00,C01,C10,C11)

    def add_strassen_2x2x2(i0, j0, k0):
        # i0, j0: top-left of A block (2x2)
        # j0, k0: top-left of B block (2x2)
        # k0, i0: top-left of C block (2x2)
        # A block: A[i0:i0+2, j0:j0+2]
        # B block: B[j0:j0+2, k0:k0+2]
        # C block: C[k0:k0+2, i0:i0+2]

        # Strassen 7 terms
        # M1
        A_d = {(i0, j0): 1, (i0+1, j0+1): 1}
        B_d = {(j0, k0): 1, (j0+1, k0+1): 1}
        C_d = {(k0, i0): 1, (k0+1, i0+1): 1}
        terms.append((A_d, B_d, C_d))
        # M2
        A_d = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        B_d = {(j0, k0): 1}
        C_d = {(k0+1, i0+1): 1, (k0, i0+1): -1}
        terms.append((A_d, B_d, C_d))
        # M3
        A_d = {(i0, j0): 1}
        B_d = {(j0+1, k0): 1, (j0+1, k0+1): -1}
        C_d = {(k0, i0+1): 1, (k0+1, i0+1): 1}
        terms.append((A_d, B_d, C_d))
        # M4
        A_d = {(i0+1, j0+1): 1}
        B_d = {(j0, k0+1): 1, (j0, k0): -1}
        C_d = {(k0, i0): 1, (k0+1, i0+1): 1}
        terms.append((A_d, B_d, C_d))
        # M5
        A_d = {(i0, j0): 1, (i0, j0+1): 1}
        B_d = {(j0+1, k0+1): 1}
        C_d = {(k0, i0): -1, (k0, i0+1): 1}
        terms.append((A_d, B_d, C_d))
        # M6
        A_d = {(i0+1, j0): 1, (i0, j0): -1}
        B_d = {(j0, k0): 1, (j0, k0+1): 1}
        C_d = {(k0+1, i0+1): 1}
        terms.append((A_d, B_d, C_d))
        # M7
        A_d = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        B_d = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        C_d = {(k0, i0): 1}
        terms.append((A_d, B_d, C_d))

    # C_left = A * B_left, both 4x4.
    # A blocks: A00=(0,0), A01=(0,2), A10=(2,0), A11=(2,2)
    # B_left blocks: B00=(0,0), B01=(0,2), B10=(2,0), B11=(2,2)  [B_left is B[:,0:4]]
    # C_left blocks: C00=(0,0), C01=(0,2), C10=(2,0), C11=(2,2)  [C_left is C[:,0:4]]
    #
    # Strassen for 4x4x4: 7 products, each a 2x2x2 multiplication.
    # M1 = (A00+A11)*(B00+B11)  -> 2x2x2 at A(0,0) and A(2,2) combined... 
    # Wait, Strassen for 4x4x4: the 7 products are each 2x2x2 multiplications.
    # M1 = (A00+A11) * (B00+B11): this is a 2x2 * 2x2 product, but A00+A11 is a 2x2 matrix (sum of two 2x2 blocks),
    #     and B00+B11 is a 2x2 matrix. So M1 is a 2x2x2 multiplication.
    #     In terms of indices: (A00+A11)[a,b] = A[a,b] + A[a+2,b+2], (B00+B11)[b,c] = B[b,c] + B[b+2,c+2].
    #     So M1 = sum_{a,b,c} (A[a,b]+A[a+2,b+2]) * (B[b,c]+B[b+2,c+2]) * C_contrib[a,c]
    #     This is a 2x2x2 Strassen term at offset (0,0) with the "sum" structure.
    #
    # Hmm, this is getting complicated. Let me just use the recursive approach:
    # For 4x4x4, Strassen gives 7 terms, each of which is a 2x2x2 multiplication.
    # Each 2x2x2 multiplication is itself decomposed via Strassen into 7 scalar-level terms.
    # So 7*7 = 49 terms for C_left.
    #
    # But the "2x2x2 multiplication" in the Strassen 4x4x4 scheme is not at a single offset;
    # it's a combination of two 2x2 blocks. So I need to handle the linear combinations.
    #
    # Let me define a function that adds a "2x2x2 multiplication" where the A-block is a linear
    # combination of 2x2 blocks, B-block is a linear combination of 2x2 blocks, and C-block is a
    # linear combination of 2x2 blocks.
    #
    # Actually, the cleanest way: represent each Strassen term at the 4x4 level as a 2x2x2 tensor
    # with coefficient vectors, and then decompose that 2x2x2 tensor via Strassen into 7 rank-1 terms.
    #
    # Let me define the 4x4 Strassen terms explicitly.
    # A 4x4 matrix has 4 2x2 blocks: A00, A01, A10, A11.
    # Strassen 4x4x4:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # Each M_r is a 2x2 * 2x2 product, where the A-factor is a linear combo of A-blocks and B-factor is a linear combo of B-blocks.
    # For example, M1: A-factor = A00+A11, B-factor = B00+B11.
    #   This means: M1[a,b,c] = sum_{x,y} (A00[x,y]+A11[x,y]) * (B00[y,z]+B11[y,z]) ... 
    #   where x,y,z are local 2x2 indices.
    #   In global indices: A00[x,y] = A[x,y], A11[x,y] = A[x+2,y+2].
    #   B00[y,z] = B[y,z], B11[y,z] = B[y+2,z+2].
    #   So M1 = sum_{x,y,z} (A[x,y]+A[x+2,y+2]) * (B[y,z]+B[y+2,z+2]) * C_contrib[x,z]
    #   where C_contrib is the C-block coefficient for M1.
    #
    # This is a 2x2x2 tensor with A-coefficients: for local (x,y), the global A-entries are (x,y) and (x+2,y+2).
    # So the A-vector for local (x,y) has entries at global (x,y) and (x+2,y+2).
    #
    # Let me define a function add_2x2x2_with_offsets(A_offsets, B_offsets, C_offsets, A_coeffs, B_coeffs, C_coeffs):
    #   where A_offsets is a list of (i0,j0) for each 2x2 block in the A-linear-combo,
    #   A_coeffs is the coefficient for each block, etc.
    #   This adds 7 Strassen terms, each a rank-1 term in the full tensor.
    #
    # Actually, let me just expand it. For M1 in 4x4 Strassen:
    #   A-factor: A00 (offset (0,0), coeff 1) + A11 (offset (2,2), coeff 1)
    #   B-factor: B00 (offset (0,0), coeff 1) + B11 (offset (2,2), coeff 1)
    #   C-factor: C00 (offset (0,0), coeff 1) + C11 (offset (2,2), coeff 1)  [from C00 = M1+..., C11 = M1+...]
    #   Wait, M1 contributes to C00 with coeff +1 and C11 with coeff +1.
    #   So C-factor: C00 (offset (0,0), coeff 1) + C11 (offset (2,2), coeff 1).
    #
    # The 2x2x2 tensor for M1 has:
    #   A-local (x,y) -> global A-entries: (x, y) with coeff 1, (x+2, y+2) with coeff 1
    #   B-local (y,z) -> global B-entries: (y, z) with coeff 1, (y+2, z+2) with coeff 1
    #   C-local (z,x) -> global C-entries: (z, x) with coeff 1, (z+2, x+2) with coeff 1
    #
    # Now decompose this 2x2x2 tensor via Strassen (7 terms). Each Strassen term at the 2x2 level
    # is a rank-1 term with local A, B, C coefficient vectors. Then I map local to global.
    #
    # This is doable but tedious. Let me just code it up.

    # I'll define a function that, given a "2x2x2 block multiplication" specified by:
    #   A_blocks: list of (i0, j0, coeff) - the A-blocks and their coefficients
    #   B_blocks: list of (j0, k0, coeff) - the B-blocks and their coefficients
    #   C_blocks: list of (k0, i0, coeff) - the C-blocks and their coefficients
    # adds the 7 Strassen rank-1 terms to `terms`.
    #
    # The 2x2x2 Strassen coefficients (local):
    strassen_2x2x2 = [
        # (A_local, B_local, C_local) where each is a dict from local index to coeff
        # local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),       # M2
        ({0:1}, {1:1, 2:1, 3:-1}, {1:1, 3:1}),  # M3
        ({3:1}, {0:-1, 1:1}, {0:1, 2:1}),       # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),       # M5
        ({0:-1, 2:1}, {0:1, 1:1}, {3:1}),       # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),       # M7
    ]

    def add_2x2x2_strassen(A_blocks, B_blocks, C_blocks):
        # A_blocks: list of (i0, j0, coeff)
        # B_blocks: list of (j0, k0, coeff)
        # C_blocks: list of (k0, i0, coeff)
        for A_loc, B_loc, C_loc in strassen_2x2x2:
            A_d = {}
            B_d = {}
            C_d = {}
            for (i0, j0, a_coeff) in A_blocks:
                for local_idx, loc_coeff in A_loc.items():
                    # local index -> local (x,y)
                    x, y = divmod(local_idx, 2)
                    gi, gj = i0 + x, j0 + y
                    A_d[(gi, gj)] = A_d.get((gi, gj), 0) + a_coeff * loc_coeff
            for (j0, k0, b_coeff) in B_blocks:
                for local_idx, loc_coeff in B_loc.items():
                    y, z = divmod(local_idx, 2)
                    gj, gk = j0 + y, k0 + z
                    B_d[(gj, gk)] = B_d.get((gj, gk), 0) + b_coeff * loc_coeff
            for (k0, i0, c_coeff) in C_blocks:
                for local_idx, loc_coeff in C_loc.items():
                    z, x = divmod(local_idx, 2)
                    gk, gi = k0 + z, i0 + x
                    C_d[(gk, gi)] = C_d.get((gk, gi), 0) + c_coeff * loc_coeff
            # Remove zero coefficients
            A_d = {k: v for k, v in A_d.items() if v != 0}
            B_d = {k: v for k, v in B_d.items() if v != 0}
            C_d = {k: v for k, v in C_d.items() if v != 0}
            if A_d and B_d and C_d:
                terms.append((A_d, B_d, C_d))

    # C_left = A * B_left (4x4 * 4x4), Strassen 4x4x4:
    # M1 = (A00+A11)*(B00+B11), C: C00+1, C11+1
    add_2x2x2_strassen(
        [(0,0,1), (2,2,1)],   # A00 + A11
        [(0,0,1), (2,2,1)],   # B00 + B11
        [(0,0,1), (2,2,1)]    # C00 + C11
    )
    # M2 = (A10+A11)*B00, C: C10+1, C11-1
    add_2x2x2_strassen(
        [(2,0,1), (2,2,1)],   # A10 + A11
        [(0,0,1)],            # B00
        [(2,2,1), (2,0,-1)]   # C11 +1, C10 -1  -> wait, C10 is at (2,0) in C-space? 
        # C-space index is (k,i). C10 = C[2:4, 0:2], so k in 2..3, i in 0..1. 
        # C10 block offset is (k0=2, i0=0). C11 block offset is (k0=2, i0=2).
        # M2 contributes to C10 with +1 and C11 with -1.
        # So C_blocks: [(2,0,1), (2,2,-1)]
    )
    # Let me redo M2: C10 = M2+M4, C11 = M1-M2+M3+M6. So M2: C10+1, C11-1.
    # C10 offset (k0=2, i0=0), C11 offset (k0=2, i0=2).
    add_2x2x2_strassen(
        [(2,0,1), (2,2,1)],   # A10 + A11
        [(0,0,1)],            # B00
        [(2,0,1), (2,2,-1)]   # C10+1, C11-1
    )
    # M3 = A00*(B10-B11), C: C01+1, C11+1
    add_2x2x2_strassen(
        [(0,0,1)],            # A00
        [(2,0,1), (2,2,-1)],  # B10 - B11
        [(0,2,1), (2,2,1)]    # C01+1, C11+1
    )
    # M4 = A11*(B01-B00), C: C00+1, C10+1
    add_2x2x2_strassen(
        [(2,2,1)],            # A11
        [(0,2,1), (0,0,-1)],  # B01 - B00
        [(0,0,1), (2,0,1)]    # C00+1, C10+1
    )
    # M5 = (A00+A01)*B11, C: C00-1, C01+1
    add_2x2x2_strassen(
        [(0,0,1), (0,2,1)],   # A00 + A01
        [(2,2,1)],            # B11
        [(0,0,-1), (0,2,1)]   # C00-1, C01+1
    )
    # M6 = (A10-A00)*(B00+B01), C: C11+1
    add_2x2x2_strassen(
        [(2,0,1), (0,0,-1)],  # A10 - A00
        [(0,0,1), (0,2,1)],   # B00 + B01
        [(2,2,1)]             # C11+1
    )
    # M7 = (A01-A11)*(B10+B11), C: C00+1
    add_2x2x2_strassen(
        [(0,2,1), (2,2,-1)],  # A01 - A11
        [(2,0,1), (2,2,1)],   # B10 + B11
        [(0,0,1)]             # C00+1
    )

    # C_right = A * B_right (4x4 * 4x3).
    # A blocks: A00=(0,0), A01=(0,2), A10=(2,0), A11=(2,2)
    # B_right = B[:,4:7] (4x3). Partition into 2x1 blocks:
    #   BR00 = B[0:2, 4:7] (2x3), BR10 = B[2:4, 4:7] (2x3)
    # C_right = C[:,4:7] (4x3). Partition:
    #   CR00 = C[0:2, 4:7] (2x3), CR10 = C[2:4, 4:7] (2x3)
    # CR00 = A00*BR00 + A01*BR10
    # CR10 = A10*BR00 + A11*BR10
    #
    # Each term is 2x2 * 2x3. I'll use schoolbook for 2x2x3 (12 scalar products each).
    #
    # For 2x2 * 2x3: A (2x2) at offset (i0,j0), B (2x3) at offset (j0,k0), C (2x3) at offset (k0,i0).
    # Schoolbook: for x in 0..1, y in 0..2, z in 0..1: C[k0+z, i0+x] += A[i0+x, j0+y] * B[j0+y, k0+z]
    # Wait, C is 2x3, so C-local (z, x) where z in 0..1 (row), x in 0..2 (col)? 
    # No, C is 2x3: rows 0..1, cols 0..2. C[i,k] where i in 0..1, k in 0..2.
    # In the tensor, C-space index is (k,i) where k is the B-column / C-col, i is the A-row / C-row.
    # So C-local (k, i) where k in 0..2, i in 0..1.
    #
    # Let me define add_2x2x3_schoolbook(i0, j0, k0):
    #   A block at (i0, j0), 2x2.
    #   B block at (j0, k0), 2x3.
    #   C block at (k0, i0), 2x3.
    def add_2x2x3_schoolbook(i0, j0, k0):
        for x in range(2):  # A row (local)
            for y in range(2):  # A col / B row (local)
                for z in range(3):  # B col / C col (local)
                    gi = i0 + x
                    gj = j0 + y
                    gk = k0 + z
                    add_scalar(gi, gj, gk, 1)

    # CR00 = A00*BR00 + A01*BR10
    # A00 at (0,0), BR00 at (0,4), C CR00 at (4,0)
    add_2x2x3_schoolbook(0, 0, 4)   # A00 * BR00
    add_2x2x3_schoolbook(0, 2, 6)   # A01 * BR10  [A01 at (0,2), BR10 at (2,4)? wait]
    # Let me recheck. BR10 = B[2:4, 4:7], so B-block offset is (j0=2, k0=4).
    # A01 = A[0:2, 2:4], so A-block offset is (i0=0, j0=2).
    # C CR00 = C[0:2, 4:7], so C-block offset is (k0=4, i0=0).
    # So add_2x2x3_schoolbook(i0=0, j0=2, k0=4) for A01*BR10 contributing to CR00.
    # But wait, I already added A00*BR00 with (0,0,4). Let me redo.
    #
    # CR00 = A00*BR00 + A01*BR10
    #   A00*BR00: A at (0,0), B at (0,4), C at (4,0) -> add_2x2x3_schoolbook(0, 0, 4)
    #   A01*BR10: A at (0,2), B at (2,4), C at (4,0) -> add_2x2x3_schoolbook(0, 2, 4)
    # CR10 = A10*BR00 + A11*BR10
    #   A10*BR00: A at (2,0), B at (0,4), C at (4,2) -> add_2x2x3_schoolbook(2, 0, 4)
    #   A11*BR10: A at (2,2), B at (2,4), C at (4,2) -> add_2x2x3_schoolbook(2, 2, 4)

    # I already added (0,0,4) above. Let me clear and redo properly.
    terms.clear()

    # C_left: Strassen 4x4x4 (49 terms)
    add_2x2x2_strassen(
        [(0,0,1), (2,2,1)],
        [(0,0,1), (2,2,1)],
        [(0,0,1), (2,2,1)]
    )
    add_2x2x2_strassen(
        [(2,0,1), (2,2,1)],
        [(0,0,1)],
        [(2,0,1), (2,2,-1)]
    )
    add_2x2x2_strassen(
        [(0,0,1)],
        [(2,0,1), (2,2,-1)],
        [(0,2,1), (2,2,1)]
    )
    add_2x2x2_strassen(
        [(2,2,1)],
        [(0,2,1), (0,0,-1)],
        [(0,0,1), (2,0,1)]
    )
    add_2x2x2_strassen(
        [(0,0,1), (0,2,1)],
        [(2,2,1)],
        [(0,0,-1), (0,2,1)]
    )
    add_2x2x2_strassen(
        [(2,0,1), (0,0,-1)],
        [(0,0,1), (0,2,1)],
        [(2,2,1)]
    )
    add_2x2x2_strassen(
        [(0,2,1), (2,2,-1)],
        [(2,0,1), (2,2,1)],
        [(0,0,1)]
    )

    # C_right: schoolbook 2x2x3 (48 terms)
    add_2x2x3_schoolbook(0, 0, 4)   # A00 * BR00 -> CR00
    add_2x2x3_schoolbook(0, 2, 4)   # A01 * BR10 -> CR00
    add_2x2x3_schoolbook(2, 0, 4)   # A10 * BR00 -> CR10
    add_2x2x3_schoolbook(2, 2, 4)   # A11 * BR10 -> CR10

    # Now convert terms to U, V, W matrices.
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (A_d, B_d, C_d) in enumerate(terms):
        for (i, j), coeff in A_d.items():
            U[i * m + j][c] += coeff
        for (j, k), coeff in B_d.items():
            V[j * p + k][c] += coeff
        for (k, i), coeff in C_d.items():
            W[k * n + i][c] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
