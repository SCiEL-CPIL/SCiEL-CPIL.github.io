import time
import sys
import random
import itertools

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 3x5x5 matrix multiplication.
    Uses Strassen decomposition for disjoint 2x2x2 sub-tensors to reduce rank.
    Target rank: 75 - 9 = 66.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # 1. Find disjoint 2x2x2 boxes
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # Box: I={i1, i2}, J={j1, j2}, K={k1, k2}
    
    # Generate all possible boxes
    i_pairs = list(itertools.combinations(range(n), 2))
    j_pairs = list(itertools.combinations(range(m), 2))
    k_pairs = list(itertools.combinations(range(p), 2))
    
    candidates = []
    for I in i_pairs:
        for J in j_pairs:
            for K in k_pairs:
                # Box is defined by (I, J, K)
                # The 8 triples are (i, j, k) for i in I, j in J, k in K
                candidates.append((I, J, K))
                
    # Shuffle to allow different packing results
    random.seed(42)
    random.shuffle(candidates)
    
    # used_jk[i] = set of (j, k) pairs used by boxes involving i
    used_jk = {i: set() for i in range(n)}
    boxes = []
    
    for I, J, K in candidates:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # The 4 (j, k) pairs used by this box
        jk_pairs = [(j1, k1), (j1, k2), (j2, k1), (j2, k2)]
        
        # Check availability for i1 and i2
        if all(p in used_jk[i1] for p in jk_pairs):
            continue
        if all(p in used_jk[i2] for p in jk_pairs):
            continue
            
        # Actually, we need to check if ANY of the pairs are used?
        # No, the pairs must be DISJOINT from previously used pairs for that i.
        # So if any pair in jk_pairs is already in used_jk[i1], we can't use this box for i1.
        if any(p in used_jk[i1] for p in jk_pairs):
            continue
        if any(p in used_jk[i2] for p in jk_pairs):
            continue
            
        # Add box
        boxes.append((I, J, K))
        for p in jk_pairs:
            used_jk[i1].add(p)
            used_jk[i2].add(p)
            
        # Optional: Stop if we reached theoretical max
        if len(boxes) >= 9:
            break
            
    # If we didn't find 9, try other seeds or just use what we found
    # But 9 should be easily reachable.
    
    # 2. Construct the decomposition
    # We will build a list of "terms".
    # Each term is a tuple (u_indices, v_indices, w_indices, coeffs)
    # But it's easier to just fill the matrices directly.
    
    # First, identify which triples are covered by boxes
    covered = set()
    box_terms = [] # List of (I, J, K)
    for I, J, K in boxes:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
        box_terms.append((I, J, K))
        
    # Uncovered triples get schoolbook terms
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_terms.append((i, j, k))
                    
    # Total rank
    rank = len(schoolbook_terms) + 7 * len(box_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add schoolbook terms
    for i, j, k in schoolbook_terms:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    # Add Strassen terms for each box
    for I, J, K in box_terms:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # Local indices:
        # i: 0->i1, 1->i2
        # j: 0->j1, 1->j2
        # k: 0->k1, 1->k2
        
        # M1 = (A00 + A11)(B00 + B11)
        U[i1 * m + j1][col] = 1
        U[i2 * m + j2][col] = 1
        V[j1 * p + k1][col] = 1
        V[j2 * p + k2][col] = 1
        W[k1 * n + i1][col] = 1 # C00
        W[k2 * n + i2][col] = 1 # C11
        col += 1
        
        # M2 = (A10 + A11)B00
        U[i2 * m + j1][col] = 1
        U[i2 * m + j2][col] = 1
        V[j1 * p + k1][col] = 1
        W[k1 * n + i2][col] = 1 # C10
        col += 1
        
        # M3 = A00(B01 - B11)
        U[i1 * m + j1][col] = 1
        V[j1 * p + k2][col] = 1
        V[j2 * p + k2][col] = -1
        W[k2 * n + i1][col] = 1 # C01
        col += 1
        
        # M4 = A11(B10 - B00)
        U[i2 * m + j2][col] = 1
        V[j2 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i1][col] = 1 # C00
        W[k1 * n + i2][col] = 1 # C10
        col += 1
        
        # M5 = (A00 + A01)B11
        U[i1 * m + j1][col] = 1
        U[i1 * m + j2][col] = 1
        V[j2 * p + k2][col] = 1
        W[k1 * n + i1][col] = -1 # C00
        W[k2 * n + i1][col] = 1 # C01
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i2 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k1][col] = 1
        V[j1 * p + k2][col] = 1
        W[k2 * n + i2][col] = 1 # C11
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i1 * m + j2][col] = 1
        U[i2 * m + j2][col] = -1
        V[j2 * p + k1][col] = 1
        V[j2 * p + k2][col] = 1
        W[k1 * n + i1][col] = 1 # C00
        W[k2 * n + i2][col] = 1 # C11
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
