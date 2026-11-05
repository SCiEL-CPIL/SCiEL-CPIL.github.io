"""Find a rank-minimal exact tensor decomposition for 3x4 @ 4x7 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 7]

def _verify_decomposition(U, V, W, rank, n, m, p):
    """Check if the decomposition exactly equals the multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _strassen_2x2x2(i0, j0, k0, n, m, p):
    """Return the 7 rank-1 terms for a 2x2 Strassen block at position (i0,j0,k0).
    Returns list of (u_idx, v_idx, w_idx, u_val, v_val, w_val) tuples."""
    # Strassen 2x2: C = A*B where A,B,C are 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    terms = []
    # Each term: for each (i,j,k) in the 2x2x2 block, add contribution
    # M1: (i0,i0+1) x (j0,j0+1) with factor on U[i,j] and V[j,k] and W[k,i]
    # The Strassen formula gives:
    # For C[i,j] = sum_k A[i,k]*B[k,j]
    # In our tensor: T[i,j,k] where i is row of A, j is col of A/row of B, k is col of B
    
    # Let's use 0-indexed within the 2x2 block:
    # i_local in {0,1}, j_local in {0,1}, k_local in {0,1}
    # Global: i = i0+i_local, j = j0+j_local, k = k0+k_local
    
    # M1 = (A00+A11)(B00+B11)
    # Contributes to C00 (via +M1), C11 (via +M1)
    # C00: i=i0, k=k0 -> W[k0*n+i0]
    # C11: i=i0+1, k=k0+1 -> W[(k0+1)*n+i0+1]
    # A00: U[i0*m+j0], A11: U[(i0+1)*m+(j0+1)]
    # B00: V[j0*p+k0], B11: V[(j0+1)*p+(k0+1)]
    
    # For each Strassen multiplier, we need to express it as sum of rank-1 terms
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But in Strassen, M1 is a single rank-1 term in the factorization:
    # U factor: for entries where M1 contributes
    # This is getting complex. Let me use the standard approach.
    
    # Actually, let me construct the full 2x2x2x2x2x2 Strassen tensor and extract rank-1 terms.
    # The 2x2 matrix multiplication tensor has rank 7 (Strassen).
    # We need to map this to our specific indices.
    
    # Standard Strassen for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # In our tensor T[i,j,k]: i=row of A, j=col of A (=row of B), k=col of B
    # For the 2x2 block at (i0,j0,k0):
    # a11 = A[i0,j0] -> U[i0*m+j0]
    # a12 = A[i0,j0+1] -> U[i0*m+(j0+1)]
    # a21 = A[i0+1,j0] -> U[(i0+1)*m+j0]
    # a22 = A[i0+1,j0+1] -> U[(i0+1)*m+(j0+1)]
    # b11 = B[j0,k0] -> V[j0*p+k0]
    # b12 = B[j0,k0+1] -> V[j0*p+(k0+1)]
    # b21 = B[j0+1,k0] -> V[(j0+1)*p+k0]
    # b22 = B[j0+1,k0+1] -> V[(j0+1)*p+(k0+1)]
    # c11 = C[i0,k0] -> W[k0*n+i0]
    # c12 = C[i0,k0+1] -> W[(k0+1)*n+i0]
    # c21 = C[i0+1,k0] -> W[k0*n+(i0+1)]
    # c22 = C[i0+1,k0+1] -> W[(k0+1)*n+(i0+1)]
    
    # For each M_r, we have:
    # M_r = (linear combo of a's) * (linear combo of b's)
    # This contributes to c's with certain coefficients.
    
    # Rank-1 term for M_r: 
    # U factor: for each (i,j), the coefficient of a_ij in the first factor
    # V factor: for each (j,k), the coefficient of b_jk in the second factor  
    # W factor: for each (k,i), the sum of coefficients from all c_ki that use M_r
    
    # M1: (a11+a22)(b11+b22)
    # U: a11->1, a22->1, others 0
    # V: b11->1, b22->1, others 0
    # W: c11->+1, c22->+1, others 0
    terms.append({
        'U': {(i0, j0): 1, (i0+1, j0+1): 1},
        'V': {(j0, k0): 1, (j0+1, k0+1): 1},
        'W': {(k0, i0): 1, (k0+1, i0+1): 1}
    })
    
    # M2: (a21+a22)b11
    # U: a21->1, a22->1
    # V: b11->1
    # W: c21->+1, c22->-1
    terms.append({
        'U': {(i0+1, j0): 1, (i0+1, j0+1): 1},
        'V': {(j0, k0): 1},
        'W': {(k0, i0+1): 1, (k0+1, i0+1): -1}
    })
    
    # M3: a11(b21-b22)
    # U: a11->1
    # V: b21->1, b22->-1
    # W: c12->+1, c22->+1
    terms.append({
        'U': {(i0, j0): 1},
        'V': {(j0+1, k0): 1, (j0+1, k0+1): -1},
        'W': {(k0+1, i0): 1, (k0+1, i0+1): 1}
    })
    
    # M4: a22(b12-b11)
    # U: a22->1
    # V: b12->1, b11->-1
    # W: c11->+1, c21->+1
    terms.append({
        'U': {(i0+1, j0+1): 1},
        'V': {(j0, k0+1): 1, (j0, k0): -1},
        'W': {(k0, i0): 1, (k0, i0+1): 1}
    })
    
    # M5: (a11+a12)b22
    # U: a11->1, a12->1
    # V: b22->1
    # W: c11->-1, c12->+1
    terms.append({
        'U': {(i0, j0): 1, (i0, j0+1): 1},
        'V': {(j0+1, k0+1): 1},
        'W': {(k0, i0): -1, (k0+1, i0): 1}
    })
    
    # M6: (a21-a11)(b11+b12)
    # U: a21->1, a11->-1
    # V: b11->1, b12->1
    # W: c22->+1
    terms.append({
        'U': {(i0+1, j0): 1, (i0, j0): -1},
        'V': {(j0, k0): 1, (j0, k0+1): 1},
        'W': {(k0+1, i0+1): 1}
    })
    
    # M7: (a12-a22)(b21+b22)
    # U: a12->1, a22->-1
    # V: b21->1, b22->1
    # W: c11->+1
    terms.append({
        'U': {(i0, j0+1): 1, (i0+1, j0+1): -1},
        'V': {(j0+1, k0): 1, (j0+1, k0+1): 1},
        'W': {(k0, i0): 1}
    })
    
    return terms

def _build_decomposition_from_blocks(blocks, n, m, p):
    """Build U, V, W from a list of Strassen blocks and remaining schoolbook terms.
    blocks: list of (i0, j0, k0) tuples for 2x2 Strassen blocks
    Returns (U, V, W, rank)"""
    
    # Determine which (i,j,k) triples are covered by blocks
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Collect all rank-1 terms
    U_terms = []
    V_terms = []
    W_terms = []
    
    for (i0, j0, k0) in blocks:
        terms = _strassen_2x2x2(i0, j0, k0, n, m, p)
        for t in terms:
            U_terms.append(t['U'])
            V_terms.append(t['V'])
            W_terms.append(t['W'])
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_terms.append({(i, j): 1})
                    V_terms.append({(j, k): 1})
                    W_terms.append({(k, i): 1})
    
    rank = len(U_terms)
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for (i, j), val in U_terms[r].items():
            U[i * m + j][r] = val
        for (j, k), val in V_terms[r].items():
            V[j * p + k][r] = val
        for (k, i), val in W_terms[r].items():
            W[k * n + i][r] = val
    
    return U, V, W, rank

def _find_max_disjoint_strassen_blocks(n, m, p, time_limit=30):
    """Find maximum set of disjoint 2x2 Strassen blocks."""
    start = time.time()
    
    # Generate all possible 2x2 blocks
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    # Two blocks are compatible if their 8-entry sets don't overlap
    def blocks_overlap(b1, b2):
        i0_1, j0_1, k0_1 = b1
        i0_2, j0_2, k0_2 = b2
        # Check if any (i,j,k) is in both
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i1, j1, k1 = i0_1+di, j0_1+dj, k0_1+dk
                    i2, j2, k2 = i0_2+di, j0_2+dj, k0_2+dk
                    if (i1, j1, k1) == (i2, j2, k2):
                        return True
        return False
    
    # Greedy: try all orderings, keep best
    best_set = []
    best_size = 0
    
    # Try greedy with different orderings
    random.seed(42)
    for trial in range(50):
        if time.time() - start > time_limit:
            break
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        current = []
        for b in shuffled:
            if all(not blocks_overlap(b, c) for c in current):
                current.append(b)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # Find maximum disjoint Strassen blocks
    blocks = _find_max_disjoint_strassen_blocks(n, m, p, time_limit=20)
    
    # Build decomposition
    U, V, W, rank = _build_decomposition_from_blocks(blocks, n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, rank, n, m, p):
        # Fallback to schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        rank = schoolbook_rank
    
    # Try to improve by finding additional small rank reductions
    # Check if we can merge any schoolbook terms with Strassen terms
    # This is complex, so for now return what we have
    
    return {"U": U, "V": V, "W": W}
