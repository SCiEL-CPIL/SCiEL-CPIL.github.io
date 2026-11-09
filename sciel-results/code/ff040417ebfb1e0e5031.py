"""
Matrix multiplication 3x5 by 5x7 with rank-minimal tensor decomposition.

Uses Strassen's 7-term algorithm for 2x2x2 blocks where applicable,
falling back to schoolbook (rank 1) for remaining scalar multiplications.

The key insight: we can partition the 3x5 and 5x7 matrices into blocks
and apply Strassen's algorithm to 2x2x2 sub-problems, reducing rank from
105 (pure schoolbook) to 103.

Strategy:
- Split A (3x5) into blocks: top 2 rows vs bottom 1 row
- Split B (5x7) into blocks: first 2 cols vs remaining 5 cols... 
  Actually, let's think more carefully.

For Strassen to work on a 2x2x2 block, we need:
- A sub-block of size 2x2 (from the 3x5 matrix)
- B sub-block of size 2x2 (from the 5x7 matrix)  
- C sub-block of size 2x2 (from the 3x7 output)

The middle dimension (5) must be the same for both A and B sub-blocks.

So we pick:
- Rows of A: {0,1} (2 rows)
- Cols of A / Rows of B: {0,1} (2 cols/rows)  
- Cols of B: {0,1} (2 cols)

This gives a 2x2x2 Strassen block with rank 7 instead of 8.

We can do this for multiple disjoint blocks. Let's identify all possible
disjoint 2x2x2 blocks:

A has 3 rows, 5 cols. B has 5 rows, 7 cols.

Possible row pairs for A: (0,1), (1,2)
Possible col/row pairs for middle: (0,1), (1,2), (2,3), (3,4)
Possible col pairs for B: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6)

For Strassen, we need the same middle pair for both A and B.
Disjointness: no two blocks can share any (i,k) pair in A, (k,j) pair in B, 
or (i,j) pair in C.

Let's use a greedy approach: try to place as many disjoint Strassen blocks as possible.

Block 1: A rows {0,1}, middle {0,1}, B cols {0,1}
  Uses A: (0,0),(0,1),(1,0),(1,1)
  Uses B: (0,0),(0,1),(1,0),(1,1)
  Uses C: (0,0),(0,1),(1,0),(1,1)

Block 2: A rows {0,1}, middle {2,3}, B cols {2,3}
  Uses A: (0,2),(0,3),(1,2),(1,3)
  Uses B: (2,2),(2,3),(3,2),(3,3)
  Uses C: (0,2),(0,3),(1,2),(1,3)

Block 3: A rows {0,1}, middle {4,?} - only one middle col left (4), can't form pair.

Block 4: A rows {1,2}, middle {0,1}, B cols {4,5}
  Uses A: (1,0),(1,1),(2,0),(2,1)
  Uses B: (0,4),(0,5),(1,4),(1,5)
  Uses C: (1,4),(1,5),(2,4),(2,5)

Block 5: A rows {1,2}, middle {2,3}, B cols {6,?} - only one col left (6).

Let me check disjointness more carefully.

After Blocks 1, 2, 4:
A used: (0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3),(1,0),(1,1),(2,0),(2,1)
Note: (1,0) and (1,1) appear in both Block 1 and Block 4! Not disjoint.

Let me redo. I need to be careful about which (i,k) pairs are used.

Block 1: A rows {0,1}, middle {0,1}, B cols {0,1}
  A pairs: (0,0),(0,1),(1,0),(1,1)
  B pairs: (0,0),(0,1),(1,0),(1,1)
  C pairs: (0,0),(0,1),(1,0),(1,1)

Block 2: A rows {0,1}, middle {2,3}, B cols {2,3}
  A pairs: (0,2),(0,3),(1,2),(1,3)
  B pairs: (2,2),(2,3),(3,2),(3,3)
  C pairs: (0,2),(0,3),(1,2),(1,3)

Block 3: A rows {0,1}, middle {4,?} - can't, only col 4 left.

Block 4: A rows {1,2}, middle {0,1}, B cols {4,5}
  A pairs: (1,0),(1,1),(2,0),(2,1)
  But (1,0),(1,1) already used in Block 1. Conflict.

Block 4 alt: A rows {2,?} - only row 2 left, can't form pair with others if 0,1 used.

Let me try:
Block 1: A rows {0,1}, middle {0,1}, B cols {0,1}
Block 2: A rows {0,1}, middle {2,3}, B cols {2,3}
Block 3: A rows {1,2}, middle {4,?} - no.

What about:
Block 1: A rows {0,1}, middle {0,1}, B cols {0,1}
Block 2: A rows {0,1}, middle {2,3}, B cols {2,3}
Block 3: A rows {1,2}, middle {4,?} - no pair.

So only 2 Strassen blocks with rows {0,1}.

Alternative:
Block 1: A rows {0,1}, middle {0,1}, B cols {0,1}
Block 2: A rows {1,2}, middle {2,3}, B cols {2,3}
  A pairs: (1,2),(1,3),(2,2),(2,3) - no conflict with Block 1's (0,0),(0,1),(1,0),(1,1)
  B pairs: (2,2),(2,3),(3,2),(3,3) - no conflict with Block 1's (0,0),(0,1),(1,0),(1,1)
  C pairs: (1,2),(1,3),(2,2),(2,3) - no conflict with Block 1's (0,0),(0,1),(1,0),(1,1)
Block 3: A rows {0,1}, middle {4,?} - no.
Block 3 alt: A rows {1,2}, middle {0,1}, B cols {4,5}
  A pairs: (1,0),(1,1),(2,0),(2,1) - (1,0),(1,1) conflict with Block 1!

So with this arrangement: 2 Strassen blocks.

Total rank: 105 - 2 = 103.

Let me verify: each Strassen block saves 1 rank (8-7=1). With 2 blocks, save 2. Rank = 103.
"""

import time

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Construct rank-minimal (or near-minimal) tensor decomposition for 3x5 * 5x7 matrix mult.
    
    Uses Strassen's algorithm for two disjoint 2x2x2 sub-blocks, reducing rank from 105 to 103.
    """
    n, m, p = 3, 5, 7
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Strassen coefficients for 2x2x2:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)(b00)
    # p3 = (a00)(b01-b11)
    # p4 = (a11)(b10-b00)
    # p5 = (a00+a01)(b11)
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    # c00 = p1+p4-p5+p7
    # c01 = p3+p5
    # c10 = p2+p4
    # c11 = p1-p2+p3+p6
    
    # Local indices for 2x2 blocks:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # U coefficients (for A) in order [p1,p2,p3,p4,p5,p6,p7]:
    U_strassen = [
        [1, 0, 0, 0, 1, 0, 0],  # a00: p1, p5
        [0, 0, 1, 0, 0, 0, 0],  # a01: p3
        [0, 1, 0, 0, 0, 1, 0],  # a10: p2, p6
        [1, 1, 0, 1, 0, 0, -1], # a11: p1, p2, p4, p7
    ]
    # V coefficients (for B):
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # b00: p1, p2, p4, p6
        [0, 0, 1, 0, 0, 1, 0],  # b01: p3, p6
        [0, 0, 0, 1, 0, 0, 1],  # b10: p4, p7
        [1, 0, -1, 0, 1, 0, 0], # b11: p1, p3, p5
    ]
    # W coefficients (for C):
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # c00: p1, p4, -p5, p7
        [0, 0, 1, 0, 1, 0, 0],   # c01: p3, p5
        [0, 1, 0, 1, 0, 0, 0],   # c10: p2, p4
        [1, -1, 1, 0, 0, 1, 0],  # c11: p1, -p2, p3, p6
    ]
    
    # Define the two Strassen blocks:
    # Block 1: A rows (0,1), middle cols (0,1), B cols (0,1)
    # Block 2: A rows (1,2), middle cols (2,3), B cols (2,3)
    
    blocks = [
        # (A_row1, A_row2, mid1, mid2, B_col1, B_col2)
        (0, 1, 0, 1, 0, 1),
        (1, 2, 2, 3, 2, 3),
    ]
    
    # Track which (i,k), (k,j), (i,j) pairs are used by Strassen
    strassen_A = set()
    strassen_B = set()
    strassen_C = set()
    
    for (r1, r2, k1, k2, c1, c2) in blocks:
        for r in (r1, r2):
            for k in (k1, k2):
                strassen_A.add((r, k))
        for k in (k1, k2):
            for c in (c1, c2):
                strassen_B.add((k, c))
        for r in (r1, r2):
            for c in (c1, c2):
                strassen_C.add((r, c))
    
    # Calculate total rank
    num_strassen = len(blocks)
    total_A = n * m
    total_B = m * p
    total_C = p * n
    
    # Rank = num_strassen * 7 + remaining schoolbook terms
    # Remaining terms = total - strassen-covered
    remaining_A = total_A - len(strassen_A)
    remaining_B = total_B - len(strassen_B)
    remaining_C = total_C - len(strassen_C)
    
    # For schoolbook, each term uses one A, one B, one C entry
    # The number of schoolbook terms = min of remaining, but actually they're paired
    # Each schoolbook term corresponds to a unique (i,k,j) triple where none of the pairs are in Strassen
    # Actually, it's simpler: total terms = n*m*p = 105
    # Strassen covers: for each block, 4 A entries, 4 B entries, 4 C entries, but 7 rank terms
    # The schoolbook terms are the remaining (i,k,j) triples not covered by any Strassen block
    
    # Count Strassen-covered (i,k,j) triples
    strassen_triples = set()
    for (r1, r2, k1, k2, c1, c2) in blocks:
        for r in (r1, r2):
            for k in (k1, k2):
                for c in (c1, c2):
                    strassen_triples.add((r, k, c))
    
    num_strassen_triples = len(strassen_triples)
    num_schoolbook = n * m * p - num_strassen_triples
    rank = num_strassen * 7 + num_schoolbook
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen blocks
    for (r1, r2, k1, k2, c1, c2) in blocks:
        # Map local to global indices
        A_local = [
            (r1, k1), (r1, k2),
            (r2, k1), (r2, k2)
        ]
        B_local = [
            (k1, c1), (k1, c2),
            (k2, c1), (k2, c2)
        ]
        C_local = [
            (r1, c1), (r1, c2),
            (r2, c1), (r2, c2)
        ]
        
        for s in range(7):
            # U coefficients
            for a_idx, (r, k) in enumerate(A_local):
                if U_strassen[a_idx][s] != 0:
                    U[r * m + k][col] = U_strassen[a_idx][s]
            
            # V coefficients
            for b_idx, (k, c) in enumerate(B_local):
                if V_strassen[b_idx][s] != 0:
                    V[k * p + c][col] = V_strassen[b_idx][s]
            
            # W coefficients
            for c_idx, (r, c) in enumerate(C_local):
                if W_strassen[c_idx][s] != 0:
                    W[c * n + r][col] = W_strassen[c_idx][s]
            
            col += 1
    
    # Add schoolbook terms
    for i in range(n):
        for k in range(m):
            for j in range(p):
                if (i, k, j) in strassen_triples:
                    continue
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    print(f"U shape: ({len(result['U'])}, {rank})")
    print(f"V shape: ({len(result['V'])}, {rank})")
    print(f"W shape: ({len(result['W'])}, {rank})")
