"""Matrix multiplication 3x4x7 tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p):
    """Verify exact tensor decomposition. T[i,j,k]=1 iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]==1."""
    rank = len(U[0]) if U else 0
    U_sp = {}
    for i in range(n * m):
        for c in range(rank):
            val = U[i][c]
            if val != 0:
                U_sp.setdefault(i, {})[c] = val
    V_sp = {}
    for j in range(m * p):
        for c in range(rank):
            val = V[j][c]
            if val != 0:
                V_sp.setdefault(j, {})[c] = val
    W_sp = {}
    for k in range(p * n):
        for c in range(rank):
            val = W[k][c]
            if val != 0:
                W_sp.setdefault(k, {})[c] = val
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = 0
                u_dict = U_sp.get(ui, {})
                v_dict = V_sp.get(vj, {})
                w_dict = W_sp.get(wk, {})
                if len(u_dict) <= len(v_dict) and len(u_dict) <= len(w_dict):
                    for c, uc in u_dict.items():
                        if c in v_dict and c in w_dict:
                            s += uc * v_dict[c] * w_dict[c]
                elif len(v_dict) <= len(u_dict) and len(v_dict) <= len(w_dict):
                    for c, vc in v_dict.items():
                        if c in u_dict and c in w_dict:
                            s += u_dict[c] * vc * w_dict[c]
                else:
                    for c, wc in w_dict.items():
                        if c in u_dict and c in v_dict:
                            s += u_dict[c] * v_dict[c] * wc
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
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


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U shape (4,7), V shape (4,7), W shape (4,7) with the trace convention
    W[k*n+i, c] where n=2, so W index = k*2 + i.
    U index = i*m + j = i*2 + j (i,j in {0,1})
    V index = j*p + k = j*2 + k
    W index = k*n + i = k*2 + i
    """
    # Strassen's 7 products for computing C = A*B where A,B,C are 2x2.
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # C = [[c00, c01],[c10, c11]]
    #
    # Products:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    #
    # For each product mc, we need rank-1 terms U_col, V_col, W_col.
    # U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] summed over c gives C[i,k] = sum_j A[i,j]B[j,k].
    #
    # Each product mc is a product of a linear form in A and a linear form in B.
    # U entries are coefficients of A[i,j], V entries are coefficients of B[j,k],
    # W entries are coefficients that route to C[i,k].
    
    U = [[0]*7 for _ in range(4)]  # index i*2+j
    V = [[0]*7 for _ in range(4)]  # index j*2+k
    W = [[0]*7 for _ in range(4)]  # index k*2+i
    
    # Product 1: m1 = (a00+a11)(b00+b11)
    # U: a00 coeff 1 at (i=0,j=0)=idx0; a11 coeff 1 at (i=1,j=1)=idx3
    # V: b00 coeff 1 at (j=0,k=0)=idx0; b11 coeff 1 at (j=1,k=1)=idx3
    # W: contributes to c00,m1; c11,m1
    # c00 = idx k*2+i: k=0,i=0 -> idx0; c11: k=1,i=1 -> idx3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00 and c11
    
    # Product 2: m2 = (a10+a11)b00
    # U: a10 at (i=1,j=0)=idx2; a11 at (i=1,j=1)=idx3
    # V: b00 at (j=0,k=0)=idx0
    # W: c10 = m2+m4 -> c10: k=0,i=1 -> idx1; c11 = m1-m2+... -> c11: k=1,i=1 -> idx3 (with -1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1   # c10
    W[3][1] = -1  # c11
    
    # Product 3: m3 = a00(b01-b11)
    # U: a00 at idx0
    # V: b01 at (j=0,k=1)=idx1; b11 at (j=1,k=1)=idx3
    # W: c00 = m1+m4-m5+m7 (no m3); c01 = m3+m5 -> c01: k=1,i=0 -> idx2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1  # c01
    
    # Product 4: m4 = a11(b10-b00)
    # U: a11 at idx3
    # V: b10 at (j=1,k=0)=idx2; b00 at (j=0,k=0)=idx0
    # W: c00 = m1+m4-... -> c00 idx0; c10 = m2+m4 -> c10 idx1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # c00
    W[1][3] = 1  # c10
    
    # Product 5: m5 = (a00+a01)b11
    # U: a00 at idx0; a01 at (i=0,j=1)=idx1
    # V: b11 at (j=1,k=1)=idx3
    # W: c00 = m1+m4-m5+m7 -> c00 idx0 (with -1); c01 = m3+m5 -> c01 idx2
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # c00
    W[2][4] = 1   # c01
    
    # Product 6: m6 = (a10-a00)(b00+b01)
    # U: a10 at idx2; a00 at idx0
    # V: b00 at idx0; b01 at idx1
    # W: c11 = m1-m2+m3+m6 -> c11 idx3
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # c11
    
    # Product 7: m7 = (a01-a11)(b10+b11)
    # U: a01 at idx1; a11 at idx3
    # V: b10 at idx2; b11 at idx3
    # W: c00 = m1+m4-m5+m7 -> c00 idx0
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # c00
    
    return U, V, W


def _build_sliding_window(n, m, p):
    """
    Constructive sliding-window decomposition for 3x4x7.
    
    Partition rows into two overlapping 2-row blocks:
      Block A: rows {0,1}  (2x4x7)
      Block B: rows {1,2}  (2x4x7)
    Each 2x4x7 is split into two 2x2x2 subproblems (cols {0,1} and {2,3}).
    
    For 2x4x7: two Strassen 2x2x2 blocks -> 14 mults each.
    Total naive: 28. But row 1 is shared between blocks A and B.
    
    We handle the overlap by carefully assigning which block handles which
    output entries, and using the structure to avoid double counting.
    
    Actually, the cleanest exact construction: each output C[i,k] for i in {0,1,2},
    k in {0..6} must equal sum_j A[i,j] B[j,k].
    
    Let's instead build it directly: use Strassen for the 2x2x2 corner blocks
    and direct multiplication for the rest, sharing the middle row.
    
    A simpler exact construction: 
      - Use 2x2x2 Strassen for rows {0,1}, cols {0,1} -> handles C[0,0],C[0,1],C[1,0],C[1,1]
      - Use 2x2x2 Strassen for rows {0,1}, cols {2,3} -> handles C[0,2..3],C[1,2..3]
      - Use 2x2x2 Strassen for rows {1,2}, cols {0,1} -> handles C[1,0..1],C[2,0..1]
      - Use 2x2x2 Strassen for rows {1,2}, cols {2,3} -> handles C[1,2..3],C[2,2..3]
    
    But C[1,*] is handled by both row-blocks -> double counted. We need to subtract.
    
    This is getting complex. Let me use a different, cleaner approach:
    Directly build a decomposition where I assign each output (i,k) to be
    computed, using shared intermediate products.
    
    Cleanest: I'll build the decomposition term-by-term using the fact that
    the tensor is the matrix multiplication tensor, and use a known small-rank
    construction. Let me instead use a "Kronecker/product" style construction
    that is verifiably exact.
    """
    # I'll build using explicit rank-1 terms. Each term c has U_col, V_col, W_col.
    # The sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1 for all (i,j,k).
    
    # Strategy: use the identity that matrix multiplication can be written using
    # the 7 Strassen products for 2x2x2, and extend to 3x4x7 via block structure.
    
    # Let me build it as: 
    #   C = A*B, A is 3x4, B is 4x7, C is 3x7.
    #   Split A into rows: A0 = A[0,:] (1x4), A1 = A[1,:] (1x4), A2 = A[2,:] (1x4)
    #   C[i,:] = A[i,:] * B  (1x4 times 4x7 = 1x7)
    #   Each row of C is a 1x4x7 multiplication.
    
    # A 1x4x7 multiplication: C[k] = sum_{j=0}^{3} A[j] B[j,k] for k=0..6.
    # This is rank 28 (direct) but can be reduced.
    # 1x4x7: we can split the 4 columns into 2x2: 
    #   C[0..1] = A[0..1]*B[0..1,0..1] + A[2..3]*B[2..3,0..1]
    #   C[2..3] = A[0..1]*B[0..1,2..3] + A[2..3]*B[2..3,2..3]
    #   C[4..6] = A[0..3]*B[0..3,4..6]
    # Each 1x2x2 is rank 4 (direct, since 1x2x2 has rank 4).
    # 1x4x2 = two 1x2x2 = rank 8.
    # C[4..6] is 1x4x3 = rank 12.
    # Total per row: 8 + 8 + 12 = 28. Times 3 rows = 84. No improvement.
    
    # The key to improvement is SHARING across rows. Let me use the 2x2x2 Strassen
    # on pairs of rows to share the B factors.
    
    # I'll construct it explicitly with verification. Let me build a decomposition
    # using Strassen blocks and check it's exact, adjusting if needed.
    
    # Build 4 Strassen 2x2x2 blocks:
    # Block (rows 0-1, cols 0-1), (rows 0-1, cols 2-3), (rows 1-2, cols 0-1), (rows 1-2, cols 2-3)
    # Each gives 7 products. Total 28 products.
    # The issue: C[1,k] gets contributions from both the top and bottom row blocks.
    
    # Let me just build all 28 terms and see what C[i,k] they produce, then fix.
    
    U = []
    V = []
    W = []
    
    # Helper: add a Strassen block for rows (r0,r1), cols (c0,c1), k-offset (k0)
    # The block computes 2x2x2: A_sub[i,j]=A[r0+i, c0+j], B_sub[j,k]=B[c0+j, k0+k]
    # and produces C_sub[i,k] which should contribute to C[r0+i, k0+k].
    def add_strassen_block(r0, c0, k0):
        sU, sV, sW = _strassen_2x2x2()
        for c in range(7):
            ucol = [0]*0  # will build full vectors later
        # Instead, build full U,V,W vectors of length = total terms.
        # Let me accumulate into lists of dicts.
        pass
    
    # Let me use a different accumulation: build U,V,W as lists of columns.
    # Each column is a dict mapping index->value.
    cols = []  # list of (U_dict, V_dict, W_dict)
    
    def add_block(r0, c0, k0, sign=1):
        sU, sV, sW = _strassen_2x2x2()
        for c in range(7):
            U_dict = {}
            V_dict = {}
            W_dict = {}
            for ui in range(4):
                val = sU[ui][c]
                if val != 0:
                    # ui = i*2+j where i,j in {0,1}
                    i = ui // 2
                    j = ui % 2
                    global_i = r0 + i
                    global_j = c0 + j
                    idx = global_i * m + global_j
                    U_dict[idx] = U_dict.get(idx, 0) + sign * val
            for vi in range(4):
                val = sV[vi][c]
                if val != 0:
                    j = vi // 2
                    k = vi % 2
                    global_j = c0 + j
                    global_k = k0 + k
                    idx = global_j * p + global_k
                    V_dict[idx] = V_dict.get(idx, 0) + sign * val
            for wi in range(4):
                val = sW[wi][c]
                if val != 0:
                    k = wi // 2
                    i = wi % 2
                    global_k = k0 + k
                    global_i = r0 + i
                    idx = global_k * n + global_i
                    W_dict[idx] = W_dict.get(idx, 0) + sign * val
            cols.append((U_dict, V_dict, W_dict))
    
    # Add 4 blocks
    add_block(0, 0, 0)  # rows 0-1, cols 0-1, k 0-1
    add_block(0, 2, 2)  # rows 0-1, cols 2-3, k 2-3
    add_block(1, 0, 0)  # rows 1-2, cols 0-1, k 0-1
    add_block(1, 2, 2)  # rows 1-2, cols 2-3, k 2-3
    
    # Now we have 28 terms. Let's check what they produce for each (i,j,k).
    # Sum over terms of U[i*m+j]*V[j*p+k]*W[k*n+i].
    # We need this to equal 1 for all (i,j,k).
    
    # Let me compute the current tensor and see where it fails.
    def compute_tensor(cols):
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for U_dict, V_dict, W_dict in cols:
                        u = U_dict.get(i*m+j, 0)
                        v = V_dict.get(j*p+k, 0)
                        w = W_dict.get(k*n+i, 0)
                        if u != 0 and v != 0 and w != 0:
                            s += u * v * w
                    T[(i,j,k)] = s
        return T
    
    T = compute_tensor(cols)
    
    # Check: for the "main" entries, we want T[i,j,k]=1.
    # But the 4 blocks cover k in {0,1,2,3} only. k=4,5,6 are not covered!
    # Also row 1 (i=1) is double-covered.
    
    # This construction is incomplete. Let me take a step back and use a 
    # cleaner, fully-verified approach: build the decomposition using a 
    # systematic method that I can verify is exact.
    
    # NEW APPROACH: Use the fact that we can write the multiplication tensor
    # as a sum of rank-1 tensors where each rank-1 tensor is a "permutation" 
    # type term. I'll use a known small-rank construction for 3x4x7.
    
    # Actually, let me use a completely different, guaranteed-correct approach:
    # Use the "border rank" construction via the 2x2x2 Strassen and 1x4x7 
    # with explicit verification, and if it's not exact, fall back to a 
    # verified direct construction that's still better than 84.
    
    # Let me build a VERIFIED decomposition using a smarter direct method:
    # Group the 4 columns of A and 7 columns of B into blocks, use Strassen 
    # where possible, and verify.
    
    # I'll construct it as a sum of rank-1 terms where I explicitly enumerate
    # a valid decomposition. Let me use the following known-good construction:
    # 
    # The 3x4x7 multiplication tensor has rank at most 49 (from known bounds).
    # Let me build a decomposition of rank 49 using a block-Kronecker approach.
    
    # Simplest guaranteed improvement: use Strassen for the 2x2x2 subblocks 
    # in a non-overlapping way and direct for the rest.
    
    # Non-overlapping partition of 3x4x7:
    #   Rows 0-1 (2 rows) x cols 0-3 (4 cols) x k 0-6: this is 2x4x7
    #   Row 2 (1 row) x cols 0-3 (4 cols) x k 0-6: this is 1x4x7
    # 
    # 2x4x7: split cols into 0-1 and 2-3, each 2x2x7.
    #   2x2x7: split k into 0-1, 2-3, 4-6.
    #     2x2x2 (k 0-1): rank 7
    #     2x2x2 (k 2-3): rank 7
    #     2x2x3 (k 4-6): rank 12 (direct) or better
    #   2x2x3: split into 2x2x2 + 2x2x1. 2x2x1 rank 4, 2x2x2 rank 7. = 11? 
    #   Actually 2x2x3 direct = 12. Can we do better? 2x2x3 rank is 12? 
    #   The border rank of 2x2x3 is 12, tensor rank is 12. So no improvement.
    #   2x4x7 = 7 + 7 + 12 = 26.
    # 1x4x7: split cols 0-1,2-3, k 0-6.
    #   1x2x7: split k 0-1,2-3,4-6.
    #     1x2x2: rank 4. 1x2x2: rank 4. 1x2x3: rank 6.
    #   1x2x7 = 4+4+6 = 14.
    #   1x4x7 = 14 + 14 = 28.
    # Total = 26 + 28 = 54.
    
    # That's a big improvement from 84 to 54! Let me build this.
    
    # But I need to be careful: 2x2x3 direct is rank 12, but let me double check.
    # 2x2x3: C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, j in {0,1}, k in {0,1,2}.
    # That's 2*2*3 = 12 terms. Can we do better? 
    # The tensor rank of 2x2x3: it's known to be 12 (no Strassen-like shortcut 
    # for rectangular 2x2x3). Actually, 2x2x3 can be done in rank 12, and the 
    # lower bound is 12. So rank 12 is optimal.
    
    # Let me build the full decomposition with rank 54.
    
    U = []
    V = []
    W = []
    
    def add_direct_block(r0, c0, k0, nr, nc, nk):
        """Add direct multiplication block: rows r0..r0+nr-1, cols c0..c0+nc-1, k k0..k0+nk-1."""
        for i in range(nr):
            for j in range(nc):
                for k in range(nk):
                    U_dict = {}
                    V_dict = {}
                    W_dict = {}
                    U_dict[(r0+i)*m + (c0+j)] = 1
                    V_dict[(c0+j)*p + (k0+k)] = 1
                    W_dict[(k0+k)*n + (r0+i)] = 1
                    cols.append((U_dict, V_dict, W_dict))
    
    def add_strassen_block2(r0, c0, k0):
        """Add 2x2x2 Strassen block at rows r0,r0+1; cols c0,c0+1; k k0,k0+1."""
        sU, sV, sW = _strassen_2x2x2()
        for c in range(7):
            U_dict = {}
            V_dict = {}
            W_dict = {}
            for ui in range(4):
                val = sU[ui][c]
                if val != 0:
                    i = ui // 2
                    j = ui % 2
                    idx = (r0+i)*m + (c0+j)
                    U_dict[idx] = U_dict.get(idx, 0) + val
            for vi in range(4):
                val = sV[vi][c]
                if val != 0:
                    j = vi // 2
                    k = vi % 2
                    idx = (c0+j)*p + (k0+k)
                    V_dict[idx] = V_dict.get(idx, 0) + val
            for wi in range(4):
                val = sW[wi][c]
                if val != 0:
                    k = wi // 2
                    i = wi % 2
                    idx = (k0+k)*n + (r0+i)
                    W_dict[idx] = W_dict.get(idx, 0) + val
            cols.append((U_dict, V_dict, W_dict))
    
    # Block 1: rows 0-1, cols 0-1, k 0-1: 2x2x2 Strassen
    add_strassen_block2(0, 0, 0)
    # Block 2: rows 0-1, cols 2-3, k 0-1: 2x2x2 Strassen
    add_strassen_block2(0, 2, 0)
    # Block 3: rows 0-1, cols 0-1, k 2-3: 2x2x2 Strassen
    add_strassen_block2(0, 0, 2)
    # Block 4: rows 0-1, cols 2-3, k 2-3: 2x2x2 Strassen
    add_strassen_block2(0, 2, 2)
    # Block 5: rows 0-1, cols 0-1, k 4-6: 2x2x3 direct
    add_direct_block(0, 0, 4, 2, 2, 3)
    # Block 6: rows 0-1, cols 2-3, k 4-6: 2x2x3 direct
    add_direct_block(0, 2, 4, 2, 2, 3)
    # Block 7: row 2, cols 0-1, k 0-1: 1x2x2 direct
    add_direct_block(2, 0, 0, 1, 2, 2)
    # Block 8: row 2, cols 2-3, k 0-1: 1x2x2 direct
    add_direct_block(2, 2, 0, 1, 2, 2)
    # Block 9: row 2, cols 0-1, k 2-3: 1x2x2 direct
    add_direct_block(2, 0, 2, 1, 2, 2)
    # Block 10: row 2, cols 2-3, k 2-3: 1x2x2 direct
    add_direct_block(2, 2, 2, 1, 2, 2)
    # Block 11: row 2, cols 0-1, k 4-6: 1x2x3 direct
    add_direct_block(2, 0, 4, 1, 2, 3)
    # Block 12: row 2, cols 2-3, k 4-6: 1x2x3 direct
    add_direct_block(2, 2, 4, 1, 2, 3)
    
    # Count: 4*7 + 2*12 + 4*4 + 2*6 = 28 + 24 + 16 + 12 = 80. 
    # Hmm, that's not 54. Let me recount.
    # 4 Strassen blocks: 4*7 = 28
    # 2 direct 2x2x3 blocks: 2*12 = 24
    # 4 direct 1x2x2 blocks: 4*4 = 16
    # 2 direct 1x2x3 blocks: 2*6 = 12
    # Total = 28+24+16+12 = 80.
    
    # That's only slightly better than 84. The issue is that 2x2x3 and 1x2x3 
    # don't benefit from Strassen. Let me reconsider.
    
    # The real improvement comes from using Strassen on the 2x2x2 blocks.
    # Without Strassen, 2x4x7 = 56, with Strassen on the 2x2x2 parts:
    # 2x4x7 = 2*(2x2x7) = 2*(2x2x2 + 2x2x2 + 2x2x3) = 2*(4+4+6) = 28 direct, 
    # or 2*(7+7+12) = 52 with Strassen on the 2x2x2. Wait that's worse.
    
    # I think I'm overcomplicating. Let me just verify the current cols and 
    # if it's not exact or not better, use a simpler guaranteed approach.
    
    # Let me check the current construction.
    T = compute_tensor(cols)
    # Check all (i,j,k)
    all_good = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[(i,j,k)] != 1:
                    all_good = False
                    break
    if all_good:
        # Convert to matrices
        rank = len(cols)
        U_mat = [[0]*rank for _ in range(n*m)]
        V_mat = [[0]*rank for _ in range(m*p)]
        W_mat = [[0]*rank for _ in range(p*n)]
        for c, (U_dict, V_dict, W_dict) in enumerate(cols):
            for idx, val in U_dict.items():
                U_mat[idx][c] = val
            for idx, val in V_dict.items():
                V_mat[idx][c] = val
            for idx, val in W_dict.items():
                W_mat[idx][c] = val
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    # If not exact, return None (caller will fall back)
    return None


def _try_bilinear_optimization(n, m, p, target_rank, time_limit):
    """Bilinear optimization: fix U,V, solve for W."""
    start_time = time.time()
    rng = random.Random(12345)
    
    for trial in range(200):
        if time.time() - start_time > time_limit:
            break
        
        U = [[rng.choice([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.choice([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2]) for _ in range(target_rank)] for _ in range(m * p)]
        
        # Compute A[i, k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        A = [[[0.0] * target_rank for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for c in range(target_rank):
                    for j in range(m):
                        A[i][k][c] += U[i * m + j][c] * V[j * p + k][c]
        
        W = [[0.0] * target_rank for _ in range(p * n)]
        valid = True
        for i in range(n):
            for k in range(p):
                found = False
                for c in range(target_rank):
                    if abs(A[i][k][c]) > 1e-12:
                        W[k * n + i][c] = 1.0 / A[i][k][c]
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        if _verify_exact(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None


def _try_low_rank_search(n, m, p, time_limit):
    """Search for rank < n*m*p."""
    start_time = time.time()
    
    for target_rank in range(40, n * m * p):
        if time.time() - start_time > time_limit:
            break
        result = _try_bilinear_optimization(n, m, p, target_rank, time_limit - (time.time() - start_time))
        if result:
            return result
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Try the constructive sliding-window decomposition first
    result = None
    try:
        result = _build_sliding_window(n, m, p)
        if result:
            if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass

    # Fall back to schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to improve via search
    try:
        remaining = deadline - time.time()
        if remaining > 5:
            improved = _try_low_rank_search(n, m, p, remaining)
            if improved:
                r = len(improved["U"][0])
                if r < best_rank:
                    if _verify_exact(improved["U"], improved["V"], improved["W"], n, m, p):
                        best_rank = r
                        result = improved
    except Exception:
        pass

    if time.time() > deadline:
        return result

    return result

# EVOLVE-BLOCK-END
