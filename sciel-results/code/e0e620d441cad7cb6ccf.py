import numpy as np
import time
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7

    # We'll build the tensor decomposition using a combination of:
    # 1. A base decomposition covering the main block
    # 2. Additional terms for the remaining entries
    
    # The key insight: we can use Strassen-style 2x2x2 blocks efficiently
    # For 3x5x7, we can tile with 2x2x2 blocks and handle the remainder
    
    # Strategy: Use 2x2x2 Strassen blocks where possible
    # n=3: rows 0-1 covered by block at i0=0, row 2 is remainder
    # m=5: cols 0-1, 2-3 covered by blocks at j0=0, j0=2, col 4 is remainder
    # p=7: outs 0-1, 2-3, 4-5 covered by blocks at k0=0, k0=2, k0=4, out 6 is remainder
    
    # Let's use a systematic approach:
    # For each pair of consecutive rows (i0, i0+1), cols (j0, j0+1), outs (k0, k0+1)
    # that don't overlap with each other in a way that causes conflicts, 
    # apply Strassen's 7-term decomposition
    
    # Non-overlapping 2x2x2 blocks for 3x5x7:
    # Row pairs: (0,1) only (since 3 rows, only one pair of consecutive rows that doesn't overlap)
    # Actually, we can use (0,1) for rows, then handle row 2 separately
    # Col pairs: (0,1) and (2,3) - these don't overlap
    # Out pairs: (0,1), (2,3), (4,5) - these don't overlap
    
    # So we can use 1 * 2 * 3 = 6 non-overlapping 2x2x2 blocks
    # Each contributes 7 terms, so 6 * 7 = 42 terms
    # Plus remainder terms for the uncovered entries
    
    # Uncovered entries after placing blocks at:
    # (i0=0, j0=0, k0=0), (i0=0, j0=2, k0=0), (i0=0, j0=0, k0=2),
    # (i0=0, j0=2, k0=2), (i0=0, j0=0, k0=4), (i0=0, j0=2, k0=4)
    
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Uncovered: 
    #   i=2, any j, any k: 1*5*7 = 35 entries
    #   i in {0,1}, j=4, any k: 2*1*7 = 14 entries  
    #   i in {0,1}, j in {0,1,2,3}, k=6: 2*4*1 = 8 entries
    #   But wait, we need to be more careful about overlaps
    
    # Let me reconsider. The blocks cover:
    # Block (0,0,0): i in {0,1}, j in {0,1}, k in {0,1}
    # Block (0,2,0): i in {0,1}, j in {2,3}, k in {0,1}
    # Block (0,0,2): i in {0,1}, j in {0,1}, k in {2,3}
    # Block (0,2,2): i in {0,1}, j in {2,3}, k in {2,3}
    # Block (0,0,4): i in {0,1}, j in {0,1}, k in {4,5}
    # Block (0,2,4): i in {0,1}, j in {2,3}, k in {4,5}
    
    # So covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # That's 2 * 4 * 6 = 48 entries covered by the 6 blocks
    
    # Total entries: 3 * 5 * 7 = 105
    # Uncovered: 105 - 48 = 57 entries
    
    # Uncovered entries:
    # i=2, any j (0-4), any k (0-6): 1*5*7 = 35
    # i in {0,1}, j=4, any k (0-6): 2*1*7 = 14
    # i in {0,1}, j in {0,1,2,3}, k=6: 2*4*1 = 8
    # Total: 35 + 14 + 8 = 57 ✓
    
    # For the uncovered entries, we use schoolbook (1 term per entry)
    # Total rank: 42 + 57 = 99
    
    # But we can do better! Let's think about this differently.
    
    # Actually, let's try a different tiling. What if we use:
    # Row pairs: (0,1) 
    # Col pairs: (0,1), (2,3)
    # Out pairs: (0,1), (2,3), (4,5)
    # That's what we have. 6 blocks * 7 = 42 terms
    
    # Alternative: Can we share terms between adjacent blocks?
    # This is complex. Let's stick with the simple approach for now.
    
    # Build the Strassen decomposition for each 2x2x2 block
    U_terms = []  # Each term: dict mapping (i,j) to coefficient
    V_terms = []  # Each term: dict mapping (j,k) to coefficient
    W_terms = []  # Each term: dict mapping (k,i) to coefficient
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen's 7-term decomposition for a 2x2x2 block."""
        # M1 = (A00 + A11) * (B00 + B11)
        U = {(i0, j0): 1, (i0+1, j0+1): 1}
        V = {(j0, k0): 1, (j0+1, k0+1): 1}
        W = {(k0, i0): 1, (k0+1, i0+1): 1}
        U_terms.append(U); V_terms.append(V); W_terms.append(W)
        
        # M2 = (A10 + A11) * B00
        U = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        V = {(j0, k0): 1}
        W = {(k0, i0+1): 1, (k0+1, i0+1): -1}
        U_terms.append(U); V_terms.append(V); W_terms.append(W)
        
        # M3 = A00 * (B01 - B11)
        U = {(i0, j0): 1}
        V = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        W = {(k0+1, i0): 1, (k0+1, i0+1): 1}
        U_terms.append(U); V_terms.append(V); W_terms.append(W)
        
        # M4 = A11 * (B10 - B00)
        U = {(i0+1, j0+1): 1}
        V = {(j0+1, k0): 1, (j0, k0): -1}
        W = {(k0, i0): 1, (k0, i0+1): 1}
        U_terms.append(U); V_terms.append(V); W_terms.append(W)
        
        # M5 = (A00 + A01) * B11
        U = {(i0, j0): 1, (i0, j0+1): 1}
        V = {(j0+1, k0+1): 1}
        W = {(k0+1, i0): -1, (k0, i0): 1}
        U_terms.append(U); V_terms.append(V); W_terms.append(W)
        
        # M6 = (A10 - A00) * (B00 + B01)
        U = {(i0+1, j0): 1, (i0, j0): -1}
        V = {(j0, k0): 1, (j0, k0+1): 1}
        W = {(k0+1, i0+1): 1}
        U_terms.append(U); V_terms.append(V); W_terms.append(W)
        
        # M7 = (A01 - A11) * (B10 + B11)
        U = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        V = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        W = {(k0, i0): 1}
        U_terms.append(U); V_terms.append(V); W_terms.append(W)
    
    # Add the 6 Strassen blocks
    block_positions = [
        (0, 0, 0), (0, 2, 0),
        (0, 0, 2), (0, 2, 2),
        (0, 0, 4), (0, 2, 4)
    ]
    
    covered = set()
    for (i0, j0, k0) in block_positions:
        add_strassen_block(i0, j0, k0)
        for i in (i0, i0+1):
            for j in (j0, j0+1):
                for k in (k0, k0+1):
                    covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_terms.append({(i, j): 1})
                    V_terms.append({(j, k): 1})
                    W_terms.append({(k, i): 1})
    
    # Convert to matrices
    rank = len(U_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for (i, j), val in U_terms[c].items():
            U[i * m + j, c] = val
        for (j, k), val in V_terms[c].items():
            V[j * p + k, c] = val
        for (k, i), val in W_terms[c].items():
            W[k * n + i, c] = val
    
    # Verify correctness
    if time.time() < deadline - 10:
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_comp - T) ** 2)
        
        if error > 1e-9:
            # Fallback to pure schoolbook
            rank_fb = n * m * p
            U = np.zeros((n * m, rank_fb), dtype=float)
            V = np.zeros((m * p, rank_fb), dtype=float)
            W = np.zeros((p * n, rank_fb), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            rank = rank_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
