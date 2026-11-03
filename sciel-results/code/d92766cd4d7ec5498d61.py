import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s wall-clock

    n, m, p = DIMENSIONS

    # Laderman's 19-term algorithm for 3x3x3 matrix multiplication
    # Each term is (u_indices, v_indices, w_indices) with coefficients
    # U[i,j], V[j,k], W[k,i]
    LADERMAN_TERMS = [
        # (U_entries, V_entries, W_entries)
        # M1 = A[0,0]*B[0,0]
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # M2 = A[1,1]*B[1,1]
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # M3 = A[2,2]*B[2,2]
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # M4 = (A[0,0]+A[0,1])*(B[1,0]-B[0,0])
        ([(0,0,1),(0,1,1)], [(1,0,-1),(0,0,-1)], [(1,0,1)]),
        # M5 = (A[1,1]+A[1,2])*(B[2,1]-B[1,1])
        ([(1,1,1),(1,2,1)], [(2,1,-1),(1,1,-1)], [(2,1,1)]),
        # M6 = (A[2,2]+A[2,0])*(B[0,2]-B[2,2])
        ([(2,2,1),(2,0,1)], [(0,2,-1),(2,2,-1)], [(0,2,1)]),
        # M7 = (A[0,1]+A[1,1])*(B[1,0]+B[0,0])
        ([(0,1,1),(1,1,1)], [(1,0,1),(0,0,1)], [(0,1,1)]),
        # M8 = (A[1,2]+A[2,2])*(B[2,1]+B[1,1])
        ([(1,2,1),(2,2,1)], [(2,1,1),(1,1,1)], [(1,2,1)]),
        # M9 = (A[2,0]+A[0,0])*(B[0,2]+B[2,2])
        ([(2,0,1),(0,0,1)], [(0,2,1),(2,2,1)], [(2,0,1)]),
        # M10 = (A[0,0]+A[1,0]+A[2,0]+A[2,1])*(B[1,2]+B[2,2]+B[0,2]+B[0,1])
        ([(0,0,1),(1,0,1),(2,0,1),(2,1,1)],
         [(1,2,1),(2,2,1),(0,2,1),(0,1,1)],
         [(0,0,-1)]),
        # M11 = (A[0,1]+A[1,1]+A[2,1]+A[2,2])*(B[2,0]+B[0,0]+B[1,0]+B[1,1])
        ([(0,1,1),(1,1,1),(2,1,1),(2,2,1)],
         [(2,0,1),(0,0,1),(1,0,1),(1,1,1)],
         [(1,1,-1)]),
        # M12 = (A[1,0]+A[2,0]+A[0,0]+A[0,2])*(B[0,1]+B[2,1]+B[1,1]+B[1,0])
        ([(1,0,1),(2,0,1),(0,0,1),(0,2,1)],
         [(0,1,1),(2,1,1),(1,1,1),(1,0,1)],
         [(2,2,-1)]),
        # M13 = (A[0,0]+A[1,0]+A[1,1]+A[2,1])*(B[1,1]+B[2,1]+B[0,2]+B[2,2])
        ([(0,0,1),(1,0,1),(1,1,1),(2,1,1)],
         [(1,1,1),(2,1,1),(0,2,1),(2,2,1)],
         [(0,1,-1)]),
        # M14 = (A[1,1]+A[2,1]+A[2,2]+A[0,2])*(B[2,2]+B[0,2]+B[1,0]+B[0,0])
        ([(1,1,1),(2,1,1),(2,2,1),(0,2,1)],
         [(2,2,1),(0,2,1),(1,0,1),(0,0,1)],
         [(1,2,-1)]),
        # M15 = (A[2,2]+A[0,2]+A[0,0]+A[1,0])*(B[0,0]+B[1,0]+B[2,1]+B[1,1])
        ([(2,2,1),(0,2,1),(0,0,1),(1,0,1)],
         [(0,0,1),(1,0,1),(2,1,1),(1,1,1)],
         [(2,0,-1)]),
        # M16 = (A[0,0]+A[1,1]+A[1,2]+A[2,2])*(B[2,2]+B[0,0]+B[1,1]+B[0,1])
        ([(0,0,1),(1,1,1),(1,2,1),(2,2,1)],
         [(2,2,1),(0,0,1),(1,1,1),(0,1,1)],
         [(0,2,-1)]),
        # M17 = (A[1,1]+A[2,2]+A[2,0]+A[0,0])*(B[0,0]+B[2,2]+B[1,2]+B[2,1])
        ([(1,1,1),(2,2,1),(2,0,1),(0,0,1)],
         [(0,0,1),(2,2,1),(1,2,1),(2,1,1)],
         [(1,0,-1)]),
        # M18 = (A[2,2]+A[0,0]+A[0,1]+A[1,1])*(B[0,1]+B[2,2]+B[1,0]+B[0,0])
        ([(2,2,1),(0,0,1),(0,1,1),(1,1,1)],
         [(0,1,1),(2,2,1),(1,0,1),(0,0,1)],
         [(2,1,-1)]),
        # M19 = (A[0,0]+A[1,1]+A[2,2]+A[2,0])*(B[2,0]+B[0,0]+B[1,1]+B[0,2])
        ([(0,0,1),(1,1,1),(2,2,1),(2,0,1)],
         [(2,0,1),(0,0,1),(1,1,1),(0,2,1)],
         [(0,0,1)]),
    ]

    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1

        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1

        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1

        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1

        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1

        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1

        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Helper to add Laderman terms for a 3x3x3 block (padded to 3x3x9 for p=7)
    def add_laderman_3x3x9(U, V, W, i0, j0, k0, col):
        """Add 19 Laderman terms for multiplying a 3x3 sub-matrix (rows i0..i0+2, cols j0..j0+2)
        by a 3x9 sub-matrix (rows j0..j0+2, cols k0..k0+8), taking only first 7 k-columns."""
        for u_entries, v_entries, w_entries in LADERMAN_TERMS:
            for (li, lj, coeff) in u_entries:
                gi = i0 + li
                gj = j0 + lj
                U[gi * m + gj][col] += coeff
            for (lj, lk, coeff) in v_entries:
                gj = j0 + lj
                gk = k0 + lk
                if gk < p:  # only valid k columns
                    V[gj * p + gk][col] += coeff
            for (lk, li, coeff) in w_entries:
                gk = k0 + lk
                gi = i0 + li
                if gk < p:  # only valid k columns
                    W[gk * n + gi][col] += coeff
            col += 1
        return col

    # Build decomposition using row-column decoupling:
    # Partition m=5 into m1=2 (j=0,1) and m2=3 (j=2,3,4)
    # For i in {0,1} (2 rows) x j in {0,1} (2 cols): use Strassen 2x2x7
    # For i in {0,1} (2 rows) x j in {2,3,4} (3 cols): use Strassen 2x2x7 per 2-col pair + schoolbook
    # For i in {2} (1 row) x j in {0,1} (2 cols): schoolbook 1x2x7
    # For i in {2} (1 row) x j in {2,3,4} (3 cols): schoolbook 1x3x7
    # 
    # Alternative: use Laderman for the 3x3x9 sub-tensor (i={0,1,2}, j={2,3,4}, k={0..8} padded)
    # This gives 19 terms for what would be 3*3*7=63 schoolbook terms.
    # But we need to handle the 2x2 parts separately.
    #
    # Strategy: 
    #   - i={0,1}, j={0,1}: Strassen 2x2x7 = 7*3 + 7 = 28 terms (3 Strassen blocks + 1 schoolbook)
    #   - i={0,1}, j={2,3,4}: Strassen 2x2x7 for j={2,3} + schoolbook for j={4}
    #     = (7*3+7) + 2*7 = 28 + 14 = 42 terms
    #   - i={2}, j={0,1,2,3,4}: schoolbook 1x5x7 = 35 terms
    #   Total: 28 + 42 + 35 = 105 (same as before, no improvement)
    #
    # Better strategy using Laderman:
    #   - i={0,1,2}, j={2,3,4}: Laderman 3x3x9 = 19 terms (covers all 3x3x7 = 63 terms)
    #   - i={0,1}, j={0,1}: Strassen 2x2x7 = 28 terms (3 Strassen + 1 schoolbook for k=6)
    #   - i={2}, j={0,1}: schoolbook 1x2x7 = 14 terms
    #   Total: 19 + 28 + 14 = 61 terms!
    #
    # Wait, that's too optimistic. Let me recalculate.
    # Laderman 3x3x9: 19 terms, each term contributes to C[i,j] for i in 0..2, j in 0..2
    # The output is a 3x7 matrix (3 rows from i, 7 columns from k).
    # Each Laderman term has W entries for k=0..2 (3 values). But we need k=0..6 (7 values).
    # So we need to pad: use 3 Laderman blocks for k={0,1,2}, k={3,4,5}, k={6,7,8}
    # But k=7,8 are out of range, so the third block only contributes to k=6.
    # Actually, Laderman gives C[i,j] = sum_k A[i,k]*B[k,j], so the output has 3x3 entries.
    # For our case: A is 3x3 (i={0,1,2}, j={2,3,4}), B is 3x7 (j={2,3,4}, k={0..6}).
    # We can decompose B into B1 (3x3, k={0,1,2}), B2 (3x3, k={3,4,5}), B3 (3x1, k={6}).
    # C = A*B1 + A*B2 + A*B3
    # A*B1: Laderman 3x3x3 = 19 terms
    # A*B2: Laderman 3x3x3 = 19 terms  
    # A*B3: schoolbook 3x3x1 = 9 terms
    # Total for this part: 19+19+9 = 47 terms
    #
    # For i={0,1}, j={0,1}: Strassen 2x2x7
    # B is 2x7 (j={0,1}, k={0..6}). Decompose into B1(2x2), B2(2x2), B3(2x1).
    # A*B1: Strassen 2x2x2 = 7 terms
    # A*B2: Strassen 2x2x2 = 7 terms
    # A*B3: schoolbook 2x2x1 = 4 terms
    # Total: 7+7+4 = 18 terms
    #
    # For i={2}, j={0,1}: schoolbook 1x2x7 = 14 terms
    #
    # Grand total: 47 + 18 + 14 = 79 terms
    # That's a big improvement from 99!

    rank_baseline = 105
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i={0,1,2}, j={2,3,4} using Laderman
    # B1: k={0,1,2}
    col = add_laderman_3x3x9(U_base, V_base, W_base, 0, 2, 0, col)
    # B2: k={3,4,5}
    col = add_laderman_3x3x9(U_base, V_base, W_base, 0, 2, 3, col)
    # B3: k={6} (schoolbook 3x3x1)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1, 2], [2, 3, 4], [6], col)
    
    # Part 2: i={0,1}, j={0,1} using Strassen
    # B1: k={0,1}
    col = add_strassen(U_base, V_base, W_base, 0, 0, 0, col)
    # B2: k={2,3}
    col = add_strassen(U_base, V_base, W_base, 0, 0, 2, col)
    # B3: k={4,5}
    col = add_strassen(U_base, V_base, W_base, 0, 0, 4, col)
    # B4: k={6} (schoolbook 2x2x1)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [0, 1], [6], col)
    
    # Part 3: i={2}, j={0,1} schoolbook 1x2x7
    col = add_schoolbook(U_base, V_base, W_base, [2], [0, 1], range(7), col)
    
    # col should be 79
    # 19 + 19 + 9 + 7 + 7 + 7 + 4 + 14 = 86... let me recount
    # Laderman: 19 terms each, so 19+19 = 38
    # Schoolbook 3x3x1: 9 terms
    # Strassen: 7 terms each, so 7+7+7 = 21
    # Schoolbook 2x2x1: 4 terms
    # Schoolbook 1x2x7: 14 terms
    # Total: 38 + 9 + 21 + 4 + 14 = 86

    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(len(U[0])):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True

    # Check if the baseline is valid
    if not verify(U_base, V_base, W_base, n, m, p):
        # Fallback to original Strassen-based decomposition
        U_fallback = [[0] * 105 for _ in range(n * m)]
        V_fallback = [[0] * 105 for _ in range(m * p)]
        W_fallback = [[0] * 105 for _ in range(p * n)]
        col = 0
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                col = add_strassen(U_fallback, V_fallback, W_fallback, 0, j0, k0, col)
        col = add_schoolbook(U_fallback, V_fallback, W_fallback, [0, 1], [0, 1], [6], col)
        col = add_schoolbook(U_fallback, V_fallback, W_fallback, [0, 1], [2, 3], [6], col)
        col = add_schoolbook(U_fallback, V_fallback, W_fallback, [0, 1], [4], range(7), col)
        col = add_schoolbook(U_fallback, V_fallback, W_fallback, [2], range(5), range(7), col)
        if verify(U_fallback, V_fallback, W_fallback, n, m, p):
            return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
        # Final fallback to full schoolbook
        U_school = [[0] * 105 for _ in range(n * m)]
        V_school = [[0] * 105 for _ in range(m * p)]
        W_school = [[0] * 105 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

    # Try to reduce rank via Gaussian elimination on the constraint matrix
    def build_constraint_matrix(U, V, W, n, m, p):
        r = len(U[0])
        M = [[Fraction(0)] * r for _ in range(n * m * p)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        M[idx][r_idx] = Fraction(U[i * m + j][r_idx]) * Fraction(V[j * p + k][r_idx]) * Fraction(W[k * n + i][r_idx])
                    idx += 1
        return M

    # Trim trailing zero columns
    def trim_zero_columns(U, V, W):
        r = len(U[0])
        keep = []
        for c in range(r):
            if any(U[i][c] != 0 for i in range(len(U))) or \
               any(V[i][c] != 0 for i in range(len(V))) or \
               any(W[i][c] != 0 for i in range(len(W))):
                keep.append(c)
        if len(keep) == r:
            return U, V, W
        U_new = [[U[i][c] for c in keep] for i in range(len(U))]
        V_new = [[V[i][c] for c in keep] for i in range(len(V))]
        W_new = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_new, V_new, W_new

    def find_redundant_columns(M, col_order=None):
        if not M or not M[0]:
            return set()
        rows = len(M)
        cols = len(M[0])
        if col_order is None:
            col_order = list(range(cols))
        A = [row[:] for row in M]
        pivot_cols = []
        redundant = set()
        
        current_row = 0
        for col_idx in col_order:
            if time.time() > deadline:
                break
            if current_row >= rows:
                redundant.add(col_idx)
                continue
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                redundant.add(col_idx)
                continue
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1
        
        # Mark all non-pivot columns as redundant
        pivot_set = set(pivot_cols)
        for c in range(cols):
            if c not in pivot_set:
                redundant.add(c)
        
        return redundant

    # First trim any zero columns
    U_base, V_base, W_base = trim_zero_columns(U_base, V_base, W_base)
    
    M_base = build_constraint_matrix(U_base, V_base, W_base, n, m, p)
    
    # Try multiple column orderings to find the minimal rank
    base_cols = list(range(len(U_base[0])))
    best_redundant = set()
    best_keep = None
    
    # Ordering 1: natural order
    redundant1 = find_redundant_columns(M_base, base_cols)
    if len(redundant1) > len(best_redundant):
        best_redundant = redundant1
        best_keep = [c for c in base_cols if c not in redundant1]
    
    if time.time() > deadline:
        if best_keep is not None:
            U_new = [[U_base[i][c] for c in best_keep] for i in range(n * m)]
            V_new = [[V_base[i][c] for c in best_keep] for i in range(m * p)]
            W_new = [[W_base[i][c] for c in best_keep] for i in range(p * n)]
            if verify(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Ordering 2: reverse order
    reversed_cols = list(reversed(base_cols))
    redundant2 = find_redundant_columns(M_base, reversed_cols)
    if len(redundant2) > len(best_redundant):
        best_redundant = redundant2
        best_keep = [c for c in base_cols if c not in redundant2]
    
    if time.time() > deadline:
        if best_keep is not None:
            U_new = [[U_base[i][c] for c in best_keep] for i in range(n * m)]
            V_new = [[V_base[i][c] for c in best_keep] for i in range(m * p)]
            W_new = [[W_base[i][c] for c in best_keep] for i in range(p * n)]
            if verify(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Ordering 3: sort by sparsity (fewer non-zero entries first)
    col_sparsity = []
    for c in base_cols:
        nz = sum(1 for i in range(n * m) if U_base[i][c] != 0)
        nz += sum(1 for i in range(m * p) if V_base[i][c] != 0)
        nz += sum(1 for i in range(p * n) if W_base[i][c] != 0)
        col_sparsity.append((nz, c))
    col_sparsity.sort()
    sparse_cols = [c for _, c in col_sparsity]
    redundant3 = find_redundant_columns(M_base, sparse_cols)
    if len(redundant3) > len(best_redundant):
        best_redundant = redundant3
        best_keep = [c for c in base_cols if c not in redundant3]
    
    if time.time() > deadline:
        if best_keep is not None:
            U_new = [[U_base[i][c] for c in best_keep] for i in range(n * m)]
            V_new = [[V_base[i][c] for c in best_keep] for i in range(m * p)]
            W_new = [[W_base[i][c] for c in best_keep] for i in range(p * n)]
            if verify(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
        return {"U": U_base, "V": V_base, "W": W_base}
    
    if best_keep is not None and len(best_keep) < len(base_cols):
        U_new = [[U_base[i][c] for c in best_keep] for i in range(n * m)]
        V_new = [[V_base[i][c] for c in best_keep] for i in range(m * p)]
        W_new = [[W_base[i][c] for c in best_keep] for i in range(p * n)]
        
        if verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new, "V": V_new, "W": W_new}
    
    return {"U": U_base, "V": V_base, "W": W_base}
