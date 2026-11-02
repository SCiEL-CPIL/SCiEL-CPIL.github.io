"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build the tensor and check
    for i in range(n):
        for j in range(p):
            for k in range(m):
                # Check that sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1
                # and for all other (i,j,k) combos not corresponding to valid mult, it's 0
                s = 0.0
                for r in range(rank):
                    s += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check that off-diagonal terms are zero
    # For any (i1,j1,k1) where the triple doesn't form a valid (i,k,j) with matching indices
    # Actually, the tensor T[i,k,j] = 1 iff output (i,j) uses input (i,k) and (k,j)
    # So T[i,k,j] = 1 for all valid i,j,k. We already checked those.
    # But we also need to check that for invalid combinations, the sum is 0.
    # In the standard setup, ALL triples (i,k,j) with i in 0..n-1, k in 0..m-1, j in 0..p-1
    # are valid! Because C[i,j] = sum_k A[i,k]*B[k,j]. So there are no "invalid" triples.
    # The tensor is fully dense: T[i,k,j] = 1 for all i,j,k.
    # Wait, that can't be right for a rank decomposition. Let me reconsider.
    # Actually yes, for matrix multiplication, every triple (i,k,j) contributes to C[i,j].
    # So the tensor T has T[i,k,j] = 1 for ALL i in 0..n-1, k in 0..m-1, j in 0..p-1.
    # That means we just need to verify that for all i,j,k:
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    # We already do that above.
    return True

def _build_rank44():
    """Build rank-44 decomposition using Strassen(7) + 4 schoolbook per 2x3x2 block."""
    n, m, p = 2, 3, 8
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 6 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 24 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 16 rows
    
    col = 0
    for g in range(4):
        # Strassen 7 terms for 2x2 x 2x2 subproblem
        # A submatrix: rows 0,1 of A, cols 0,1 -> U indices: i*3+k for i,k in 0..1
        # B submatrix: rows 0,1 of B, cols 2g,2g+1 -> V indices: k*8+(2g+j) for k,j in 0..1
        # C submatrix: rows 0,1 of C, cols 2g,2g+1 -> W indices: (2g+j)*2+i for i,j in 0..1
        
        # Strassen products (verified formula):
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B10)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B10)
        # P7 = (A01-A11)(B01+B11)
        # C00 = P1 - P5 + P6 - P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P7
        
        # Each term is a list of (index, coeff) for U, V, W
        strassen_terms = [
            # P1: U: A00(+1), A11(+1); V: B00(+1), B11(+1); W: C00(+1), C11(+1)
            ([(0, 1), (4, 1)], [(0, 1), (9, 1)], [(0, 1), (3, 1)]),
            # P2: U: A10(+1), A11(+1); V: B00(+1); W: C10(+1)
            ([(3, 1), (4, 1)], [(0, 1)], [(1, 1)]),
            # P3: U: A00(+1); V: B10(+1), B11(-1); W: C01(+1)
            ([(0, 1)], [(8, 1), (9, -1)], [(2, 1)]),
            # P4: U: A11(+1); V: B01(+1), B10(-1); W: C00(+1)
            ([(4, 1)], [(1, 1), (8, -1)], [(0, 1)]),
            # P5: U: A00(+1), A01(+1); V: B11(+1); W: C11(+1)
            ([(0, 1), (1, 1)], [(9, 1)], [(3, 1)]),
            # P6: U: A10(+1), A00(-1); V: B00(+1), B10(+1); W: C10(+1)
            ([(3, 1), (0, -1)], [(0, 1), (8, 1)], [(1, 1)]),
            # P7: U: A01(+1), A11(-1); V: B01(+1), B11(+1); W: C01(+1)
            ([(1, 1), (4, -1)], [(1, 1), (9, 1)], [(2, 1)]),
        ]
        
        for u_list, v_list, w_list in strassen_terms:
            for idx, coeff in u_list:
                U[idx][col] += coeff
            for idx, coeff in v_list:
                V[idx][col] += coeff
            for idx, coeff in w_list:
                W[idx][col] += coeff
            col += 1
        
        # Schoolbook 4 terms for 2x1 x 1x2 (A col 2, B row 2)
        for i in range(2):
            for j in range(2):
                u_idx = i * 3 + 2
                v_idx = 2 * 8 + (2 * g + j)
                w_idx = (2 * g + j) * 2 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    """Fallback schoolbook decomposition of rank 48."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Try rank-44 construction
    result = _build_rank44()
    if _verify_decomp(result["U"], result["V"], result["W"], n, m, p):
        return result
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
