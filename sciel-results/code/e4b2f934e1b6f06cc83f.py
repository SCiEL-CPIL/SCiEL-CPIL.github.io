"""Matrix multiplication tensor decomposition for 2x3x10.
Uses Strassen-style splitting to reduce rank below the schoolbook 60.
Splits m=3 into 2+1: Strassen(2x2x2)*5 + schoolbook(2x1x2)*10 = 35+20 = 55.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 130  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact tensor decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    # Build set of non-zero (a,b,c) triples
    nonzero = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero.add((i * m + j, j * p + k, k * n + i))
    
    # Check non-zero entries
    for (a, b, c) in nonzero:
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val - 1) > tol:
            return False
    
    # Check zero entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in nonzero:
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if abs(val) > tol:
                        return False
    return True

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U indices: row = i*2+j (i=0,1; j=0,1)
    # V indices: row = j*2+k (j=0,1; k=0,1)
    # W indices: row = k*2+i (k=0,1; i=0,1)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1              # B00
    W[2][1] = 1; W[3][1] = -1  # C10 += M2, C11 -= M2
    
    # M3: A00(B10-B11)
    U[0][2] = 1              # A00
    V[2][2] = 1; V[3][2] = -1  # B10 - B11
    W[1][2] = 1; W[3][2] = 1  # C01 += M3, C11 += M3
    
    # M4: A11(B01-B00)
    U[3][3] = 1              # A11
    V[1][3] = 1; V[0][3] = -1  # B01 - B00
    W[0][3] = 1; W[2][3] = 1  # C00 += M4, C10 += M4
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1              # B11
    W[0][4] = -1; W[1][4] = 1  # C00 -= M5, C01 += M5
    
    # M6: (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1  # A10 - A00
    V[0][5] = 1; V[2][5] = 1  # B00 + B10
    W[3][5] = 1              # C11 += M6
    
    # M7: (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1  # A01 - A11
    V[1][6] = 1; V[3][6] = 1  # B01 + B11
    W[0][6] = 1              # C00 += M7
    
    return U, V, W

def _build_rank55():
    """Build rank-55 decomposition by splitting m=3 into 2+1.
    
    Split the middle dimension j={0,1,2} into j={0,1} (Strassen) and j={2} (schoolbook).
    For each of the 5 column pairs (k0,k1) in the 10 columns of the second matrix:
      - Use Strassen (rank 7) for the 2x2 block (j=0,1) x 2 columns
    For the j=2 block:
      - Use schoolbook (rank 2 per column) x 10 columns = 20
    
    Total: 5*7 + 20 = 55
    """
    n, m, p = 2, 3, 10
    nm, mp, pn = n * m, m * p, p * n  # 6, 30, 20
    
    U = [[0] * 55 for _ in range(nm)]
    V = [[0] * 55 for _ in range(mp)]
    W = [[0] * 55 for _ in range(pn)]
    
    col = 0
    
    # Strassen part: j in {0,1}, k in pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    sU, sV, sW = _strassen_2x2x2()
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        for r in range(7):
            # U: rows are i*m + j where i in {0,1}, j in {0,1}
            # Strassen U row index: i*2 + j -> maps to i*3 + j
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j][col] = sU[i * 2 + j][r]
            
            # V: rows are j*p + k where j in {0,1}, k in {k0, k1}
            # Strassen V row index: j*2 + k_local -> maps to j*10 + (k0 + k_local)
            for j in range(2):
                for k_local in range(2):
                    V[j * 10 + k0 + k_local][col] = sV[j * 2 + k_local][r]
            
            # W: rows are k*n + i where k in {k0, k1}, i in {0,1}
            # Strassen W row index: k_local*2 + i -> maps to (k0+k_local)*2 + i
            for k_local in range(2):
                for i in range(2):
                    W[(k0 + k_local) * 2 + i][col] = sW[k_local * 2 + i][r]
            
            col += 1
    
    # Schoolbook part: j=2, all 10 columns
    for k in range(10):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[2 * 10 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build rank-55 decomposition
    best = _build_rank55()
    best_rank = 55
    
    # Verify
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook if verification fails
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
        best = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try to find rank 54 or lower using structured search
    # Strategy: try to merge/eliminate one term from the rank-55 decomposition
    # by finding a linear dependency or using a smarter split.
    
    # Alternative: try splitting m=3 as 1+2 instead of 2+1
    # This gives same structure but let's try different column pairings
    
    # Try: pair columns differently for Strassen
    # Original: (0,1),(2,3),(4,5),(6,7),(8,9)
    # Alternative: (0,2),(1,3),(4,6),(5,7),(8,9) or other pairings
    
    deadline = start_time + TIME_LIMIT - 10
    
    # Try alternative column pairings for the Strassen blocks
    # We need to partition 10 columns into 5 pairs
    # The Strassen decomposition requires the two k-values in each pair
    # to be "adjacent" in the sense that the W matrix maps correctly.
    # Actually, Strassen works for any two columns since it's a 2x2x2 block.
    
    # Let's try a few different pairings
    from itertools import combinations
    all_cols = list(range(10))
    # Generate a few random pairings
    random.seed(42)
    pairings_tried = set()
    pairings_tried.add(frozenset([frozenset([0,1]), frozenset([2,3]), frozenset([4,5]), frozenset([6,7]), frozenset([8,9])]))
    
    attempts = 0
    while time.time() < deadline and attempts < 200:
        # Generate random pairing
        cols = list(all_cols)
        random.shuffle(cols)
        pairs = []
        for i in range(5):
            pairs.append(frozenset([cols[2*i], cols[2*i+1]]))
        pairing_key = frozenset(pairs)
        if pairing_key in pairings_tried:
            attempts += 1
            continue
        pairings_tried.add(pairing_key)
        attempts += 1
        
        # Build decomposition with this pairing
        U = [[0] * 55 for _ in range(n * m)]
        V = [[0] * 55 for _ in range(m * p)]
        W = [[0] * 55 for _ in range(p * n)]
        
        col = 0
        sU, sV, sW = _strassen_2x2x2()
        
        for pair in pairs:
            pair_list = sorted(pair)
            k0, k1 = pair_list[0], pair_list[1]
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * 3 + j][col] = sU[i * 2 + j][r]
                for j in range(2):
                    for k_local, k_val in enumerate([k0, k1]):
                        V[j * 10 + k_val][col] = sV[j * 2 + k_local][r]
                for k_local, k_val in enumerate([k0, k1]):
                    for i in range(2):
                        W[k_val * 2 + i][col] = sW[k_local * 2 + i][r]
                col += 1
        
        # Schoolbook for j=2
        for k in range(10):
            for i in range(2):
                U[i * 3 + 2][col] = 1
                V[2 * 10 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
        
        if _verify(U, V, W, n, m, p):
            # Same rank 55, no improvement, but confirms correctness
            pass
    
    # Try to find rank 54: need to eliminate one term
    # Idea: if we can find a linear relation among the 55 terms, we can reduce rank
    # This is equivalent to finding a rank-54 decomposition, which is hard.
    
    # Alternative approach: try a different split
    # Split p=10 into 2+2+2+2+2 (five pairs) - already done
    # Split p=10 into 4+6: use 2x2x4 (Strassen-like) + 2x2x6
    # But 2x2x4 and 2x2x6 don't have known sub-schoolbook decompositions easily.
    
    # Try: split n=2 as 1+1, m=3 as 3, p=10 as 10
    # Each 1x3x10 block: schoolbook rank 30, total 60. No improvement.
    
    # Try: use the fact that we can apply Strassen to the full 2x3x10
    # by viewing it as 2x(2+1)x(5*2) and using a block Strassen.
    # This is what we already did.
    
    # For now, return rank 55
    return best
