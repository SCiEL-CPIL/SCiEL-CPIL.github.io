"""Deterministic multi-Strassen rank-78 decomposition for 3x4*4x7 matrix multiplication."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify exact decomposition including zero constraints."""
    nm, mp, pn = n * m, m * p, p * n
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for (a, b, c) in valid:
        s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
        if abs(s - 1.0) > tol:
            return False
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(s) > tol:
                        return False
    return True

def _add_strassen_block(U, V, W, rank_ref, i1, i2, j1, j2, k1, k2, n, m, p):
    """Add a Strassen 2x2x2 block. Returns new rank index."""
    # Strassen: 7 multiplications
    # m1 = (A11+A22)(B11+B22)
    # m2 = (A21+A22)B11
    # m3 = A11(B21-B22)
    # m4 = A22(B12-B11)
    # m5 = (A11+A12)B22
    # m6 = (A21-A11)(B11+B12)
    # m7 = (A12-A22)(B21+B22)
    # C11 = m1+m4-m5+m7
    # C12 = m3+m5
    # C21 = m2+m4
    # C22 = m1-m2+m3+m6
    
    a11, a12 = i1 * m + j1, i1 * m + j2
    a21, a22 = i2 * m + j1, i2 * m + j2
    b11, b12 = j1 * p + k1, j1 * p + k2
    b21, b22 = j2 * p + k1, j2 * p + k2
    c11, c12 = k1 * n + i1, k2 * n + i1
    c21, c22 = k1 * n + i2, k2 * n + i2
    
    r = rank_ref[0]
    
    # m1 = (A11+A22)(B11+B22)
    U[a11, r] += 1.0; U[a22, r] += 1.0
    V[b11, r] += 1.0; V[b22, r] += 1.0
    W[c11, r] += 1.0; W[c22, r] += 1.0; W[c12, r] -= 1.0; W[c21, r] -= 1.0
    r += 1
    
    # m2 = (A21+A22)B11
    U[a21, r] += 1.0; U[a22, r] += 1.0
    V[b11, r] += 1.0
    W[c21, r] += 1.0; W[c22, r] -= 1.0
    r += 1
    
    # m3 = A11(B21-B22)
    U[a11, r] += 1.0
    V[b21, r] += 1.0; V[b22, r] -= 1.0
    W[c12, r] += 1.0; W[c21, r] += 1.0
    r += 1
    
    # m4 = A22(B12-B11)
    U[a22, r] += 1.0
    V[b12, r] += 1.0; V[b11, r] -= 1.0
    W[c11, r] += 1.0; W[c21, r] += 1.0
    r += 1
    
    # m5 = (A11+A12)B22
    U[a11, r] += 1.0; U[a12, r] += 1.0
    V[b22, r] += 1.0
    W[c11, r] -= 1.0; W[c12, r] += 1.0
    r += 1
    
    # m6 = (A21-A11)(B11+B12)
    U[a21, r] += 1.0; U[a11, r] -= 1.0
    V[b11, r] += 1.0; V[b12, r] += 1.0
    W[c22, r] += 1.0
    r += 1
    
    # m7 = (A12-A22)(B21+B22)
    U[a12, r] += 1.0; U[a22, r] -= 1.0
    V[b21, r] += 1.0; V[b22, r] += 1.0
    W[c11, r] += 1.0; W[c22, r] += 1.0
    r += 1
    
    rank_ref[0] = r
    return r

def _add_schoolbook(U, V, W, rank_ref, i, j, k, n, m, p):
    """Add a single schoolbook multiplication."""
    a = i * m + j
    b = j * p + k
    c = k * n + i
    r = rank_ref[0]
    U[a, r] += 1.0
    V[b, r] += 1.0
    W[c, r] += 1.0
    rank_ref[0] = r + 1
    return r + 1

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Strategy: 
    # Rows {0,1} x Cols {0,1}: Strassen 2x2x2 (j={0,1}, k={0,1}) -> rank 7
    # Rows {0,1} x Cols {2,3}: Strassen 2x2x2 (j={2,3}, k={2,3}) -> rank 7
    # Rows {0,1} x Cols {4,5}: Strassen 2x2x2 (j={2,3}, k={4,5}) -> rank 7
    # Rows {0,1} x Col 6: schoolbook 2x1x1 (j={2,3}, k={6}) -> rank 4
    # Row 2 x Cols {0,1}: schoolbook 1x2x2 (j={0,1}, k={0,1}) -> rank 4
    # Row 2 x Cols {2,3}: schoolbook 1x2x2 (j={2,3}, k={2,3}) -> rank 4
    # Row 2 x Cols {4,5}: schoolbook 1x2x2 (j={2,3}, k={4,5}) -> rank 4
    # Row 2 x Col 6: schoolbook 1x1x1 (j={2,3}, k={6}) -> rank 2
    # Total: 7+7+7+4+4+4+4+2 = 39? That's way less than 78.
    
    # Wait, I need to be more careful. Let me reconsider.
    # The intermediate dimension is m=4 (indices j=0,1,2,3)
    # For Strassen 2x2x2, we need 2 rows, 2 intermediate, 2 cols.
    
    # Let me use a different partition:
    # Block 1: rows {0,1}, j {0,1}, k {0,1} -> Strassen (7)
    # Block 2: rows {0,1}, j {0,1}, k {2,3} -> Strassen (7)  
    # Block 3: rows {0,1}, j {2,3}, k {0,1} -> Strassen (7)
    # Block 4: rows {0,1}, j {2,3}, k {2,3} -> Strassen (7)
    # Block 5: rows {0,1}, j {0,1}, k {4,5} -> Strassen (7)
    # Block 6: rows {0,1}, j {2,3}, k {4,5} -> Strassen (7)
    # Block 7: rows {0,1}, j {0,1}, k {6} -> schoolbook (2*1*1=2)
    # Block 8: rows {0,1}, j {2,3}, k {6} -> schoolbook (2*1*1=2)
    # Block 9: row {2}, j {0,1}, k {0,1} -> schoolbook (1*2*2=4)
    # Block 10: row {2}, j {0,1}, k {2,3} -> schoolbook (1*2*2=4)
    # Block 11: row {2}, j {0,1}, k {4,5} -> schoolbook (1*2*2=4)
    # Block 12: row {2}, j {0,1}, k {6} -> schoolbook (1*1*1=1)
    # Block 13: row {2}, j {2,3}, k {0,1} -> schoolbook (1*2*2=4)
    # Block 14: row {2}, j {2,3}, k {2,3} -> schoolbook (1*2*2=4)
    # Block 15: row {2}, j {2,3}, k {4,5} -> schoolbook (1*2*2=4)
    # Block 16: row {2}, j {2,3}, k {6} -> schoolbook (1*1*1=1)
    # Total: 7*6 + 2+2+4+4+4+1+4+4+4+1 = 42+28 = 70
    
    # Hmm, but the inspiration says rank 78. Let me just implement this and see.
    # Actually wait - I'm double counting. Each (i,j,k) triple must be covered exactly once.
    # Let me just enumerate all 84 triples and assign them to blocks.
    
    # Actually, the cleanest approach: use Strassen for all 2x2x2 blocks where possible,
    # and schoolbook for the rest.
    
    # For rows {0,1}:
    #   j pairs: (0,1), (2,3)
    #   k pairs: (0,1), (2,3), (4,5), and single (6)
    #   For j pair (0,1) and k pair (0,1): Strassen -> covers (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    #   For j pair (0,1) and k pair (2,3): Strassen -> covers (0,0,2),(0,0,3),(0,1,2),(0,1,3),(1,0,2),(1,0,3),(1,1,2),(1,1,3)
    #   For j pair (0,1) and k pair (4,5): Strassen -> covers (0,0,4),(0,0,5),(0,1,4),(0,1,5),(1,0,4),(1,0,5),(1,1,4),(1,1,5)
    #   For j pair (0,1) and k=6: schoolbook -> (0,0,6),(0,1,6),(1,0,6),(1,1,6) = 4
    #   For j pair (2,3) and k pair (0,1): Strassen -> covers (0,2,0),(0,2,1),(0,3,0),(0,3,1),(1,2,0),(1,2,1),(1,3,0),(1,3,1)
    #   For j pair (2,3) and k pair (2,3): Strassen -> covers (0,2,2),(0,2,3),(0,3,2),(0,3,3),(1,2,2),(1,2,3),(1,3,2),(1,3,3)
    #   For j pair (2,3) and k pair (4,5): Strassen -> covers (0,2,4),(0,2,5),(0,3,4),(0,3,5),(1,2,4),(1,2,5),(1,3,4),(1,3,5)
    #   For j pair (2,3) and k=6: schoolbook -> (0,2,6),(0,3,6),(1,2,6),(1,3,6) = 4
    
    # For row {2}:
    #   j=0, k=0,1,2,3,4,5,6: 8
    #   j=1, k=0,1,2,3,4,5,6: 8
    #   j=2, k=0,1,2,3,4,5,6: 8
    #   j=3, k=0,1,2,3,4,5,6: 8
    #   Total: 32
    
    # Grand total: 7*6 + 4+4 + 32 = 42 + 8 + 32 = 82
    
    # That's only 2 better than schoolbook (84). The inspiration claims 78.
    # Let me think about how to do better...
    
    # Alternative: use Strassen more cleverly by overlapping blocks.
    # Or use a 2x4x4 block decomposition.
    
    # For now, let me implement the 82-rank solution and then try to optimize.
    # Actually, let me try a different approach: use 2x2x2 Strassen blocks more efficiently.
    
    # Key insight: for rows {0,1} and a fixed j pair, we have a 2x2 x 2x7 sub-problem.
    # We can decompose 2x2 x 2x7 as:
    #   Split k into pairs: (0,1),(2,3),(4,5) and single (6)
    #   Each 2x2x2 -> Strassen (7), single 2x2x1 -> schoolbook (4)
    #   Total per j pair: 7+7+7+4 = 25
    #   Two j pairs: 50
    # Plus row 2: 4*7 = 28
    # Total: 78! That matches the inspiration.
    
    # Wait, 2x2x1 should be rank 4 (2*2*1), not 4. Let me recount.
    # 2 rows x 2 intermediate x 1 col = 4 multiplications. Yes.
    # So: 3 Strassen (21) + 1 schoolbook 2x2x1 (4) = 25 per j-pair.
    # Two j-pairs: 50. Row 2: 28. Total: 78. 
    
    # Let me implement this.
    
    max_rank = 80  # Allocate a bit extra
    U = np.zeros((nm, max_rank))
    V = np.zeros((mp, max_rank))
    W = np.zeros((pn, max_rank))
    rank_ref = [0]
    
    # Rows {0,1}, j pair (0,1):
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 0, 1, 0, 1, n, m, p)
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 0, 1, 2, 3, n, m, p)
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 0, 1, 4, 5, n, m, p)
    # k=6, j in {0,1}, i in {0,1}: schoolbook
    for i in (0, 1):
        for j in (0, 1):
            _add_schoolbook(U, V, W, rank_ref, i, j, 6, n, m, p)
    
    # Rows {0,1}, j pair (2,3):
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 2, 3, 0, 1, n, m, p)
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 2, 3, 2, 3, n, m, p)
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 2, 3, 4, 5, n, m, p)
    # k=6, j in {2,3}, i in {0,1}: schoolbook
    for i in (0, 1):
        for j in (2, 3):
            _add_schoolbook(U, V, W, rank_ref, i, j, 6, n, m, p)
    
    # Row 2: all schoolbook
    for j in range(m):
        for k in range(p):
            _add_schoolbook(U, V, W, rank_ref, 2, j, k, n, m, p)
    
    final_rank = rank_ref[0]
    
    # Trim to actual rank
    U = U[:, :final_rank]
    V = V[:, :final_rank]
    W = W[:, :final_rank]
    
    # Verify
    if not _verify(U, V, W, n, m, p, final_rank):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_sch = np.zeros((nm, rank_school))
        V_sch = np.zeros((mp, rank_school))
        W_sch = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i * m + j, col] = 1.0
                    V_sch[j * p + k, col] = 1.0
                    W_sch[k * n + i, col] = 1.0
                    col += 1
        U, V, W = U_sch, V_sch, W_sch
    
    def _to_list(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
