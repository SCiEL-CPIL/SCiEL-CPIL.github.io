"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    The decomposition should satisfy:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    and 0 for all other (a, b, c) combinations.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if nm > 0 else 0
    
    # Check all (i, j, k) triples: should sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    
    # Check that entries where the tensor is 0 are indeed 0
    # The tensor T[a,b,c] = 1 iff a = i*m+j, b = j'*p+k, c = k'*n+i' 
    # where i=i', j=j', k=k' (i.e., the middle index matches)
    # So T[a,b,c] = 1 iff (a mod m) == (b // p) and (b mod p) == (c // n) and (c mod n) == (a // m)
    # Actually, let's be more careful.
    # a = i*m + j where i in [0,n), j in [0,m)
    # b = j'*p + k where j' in [0,m), k in [0,p)
    # c = k'*n + i' where k' in [0,p), i' in [0,n)
    # T[a,b,c] = 1 iff i=i', j=j', k=k'
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # So T[a,b,c] = 1 iff a//m == c%n and a%m == b//p and b%p == c//n
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                j_prime = b // p
                k = b % p
                k_prime = c // n
                i_prime = c % n
                
                should_be_one = (i == i_prime) and (j == j_prime) and (k == k_prime)
                
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                
                if should_be_one:
                    if s != 1:
                        return False
                else:
                    if s != 0:
                        return False
    
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if nm > 0 else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m+j]
                v_val = V_arr[j*p+k]
                w_val = W_arr[k*n+i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1) > 1e-9:
                    return False
    
    # Check zero entries
    # T[a,b,c] = 1 iff a//m == c%n, a%m == b//p, b%p == c//n
    # For all other (a,b,c), the sum should be 0.
    # This is expensive to check all, but we can check a sample.
    # Actually, let's check all since nm*mp*pn = 15*30*18 = 8100 which is manageable.
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                j_prime = b // p
                k = b % p
                k_prime = c // n
                i_prime = c % n
                
                should_be_one = (i == i_prime) and (j == j_prime) and (k == k_prime)
                
                u_val = U_arr[a]
                v_val = V_arr[b]
                w_val = W_arr[c]
                s = np.dot(u_val, v_val * w_val)
                
                if should_be_one:
                    if abs(s - 1) > 1e-9:
                        return False
                else:
                    if abs(s) > 1e-9:
                        return False
    
    return True

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W where:
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 intermediate products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, r] for A indices (i,j), r in 0..6
    U = [[0]*7 for _ in range(4)]
    # A00: row 0
    U[0][0] = 1  # M1
    U[0][2] = 1  # M3
    U[0][4] = 1  # M5
    U[0][5] = -1 # M6
    # A01: row 1
    U[1][4] = 1  # M5
    U[1][6] = 1  # M7
    # A10: row 2
    U[2][1] = 1  # M2
    U[2][5] = 1  # M6
    # A11: row 3
    U[3][0] = 1  # M1
    U[3][1] = 1  # M2
    U[3][3] = 1  # M4
    U[3][6] = -1 # M7
    
    # V[j*2+k, r] for B indices (j,k), r in 0..6
    V = [[0]*7 for _ in range(4)]
    # B00: col 0
    V[0][0] = 1  # M1
    V[0][1] = 1  # M2
    V[0][3] = -1 # M4
    V[0][5] = 1  # M6
    # B01: col 1
    V[1][2] = 1  # M3
    V[1][5] = 1  # M6
    # B10: col 2
    V[2][4] = -1 # M4
    V[2][6] = 1  # M7
    # B11: col 3
    V[3][0] = 1  # M1
    V[3][2] = -1 # M3
    V[3][4] = 1  # M5
    V[3][6] = 1  # M7
    
    # W[k*2+i, r] for C indices (k,i), r in 0..6
    W = [[0]*7 for _ in range(4)]
    # C00: (k=0,i=0) -> index 0
    W[0][0] = 1  # M1
    W[0][3] = 1  # M4
    W[0][4] = -1 # M5
    W[0][6] = 1  # M7
    # C01: (k=0,i=1) -> index 1
    W[1][2] = 1  # M3
    W[1][4] = 1  # M5
    # C10: (k=1,i=0) -> index 2
    W[2][1] = 1  # M2
    W[2][3] = 1  # M4
    # C11: (k=1,i=1) -> index 3
    W[3][0] = 1  # M1
    W[3][1] = -1 # M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Deadline guard: 150s limit, use 130s for safety
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Trivial decomposition with rank 90
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Strategy: Use block Strassen decomposition.
    # Partition the 3x5x6 multiplication into blocks and apply Strassen where possible.
    # 
    # We can partition A (3x5) into blocks of size 2x2 and 1x1, etc.
    # One approach: use 2x2 blocks for the first 2 rows/cols, and handle the rest separately.
    #
    # Let's try: Partition A into top-left 2x2 block, top-right 2x3 block, bottom-left 1x2 block, bottom-right 1x3 block.
    # Similarly for B and C.
    #
    # Actually, a simpler approach: use Strassen on overlapping 2x2 sub-blocks.
    # For each pair of rows (i1, i2) from {0,1,2} and columns (j1, j2) from {0,...,4},
    # and for each pair of output columns (k1, k2) from {0,...,5}, we can potentially use Strassen.
    # But this gets complicated with overlapping blocks.
    #
    # Let's try a different approach: use the fact that we can decompose the problem into
    # independent sub-problems by partitioning the middle dimension (j).
    #
    # For each j in 0..4, we have a rank-1 term: A[:,j] * B[j,:].
    # We can group these terms and use Strassen-like combinations.
    #
    # Actually, let's try a concrete block decomposition:
    # Partition A into 3 blocks of 1 row each: A0 (1x5), A1 (1x5), A2 (1x5)
    # Partition B into 3 blocks of 2 columns each: B0 (5x2), B1 (5x2), B2 (5x2)
    # Then C = A0*B0 + A0*B1 + A0*B2 + A1*B0 + A1*B1 + A1*B2 + A2*B0 + A2*B1 + A2*B2
    # This is 9 terms, each of rank 5 (since it's a 1x5 times 5x2 = rank 5).
    # Total rank: 9*5 = 45. This is better than 90!
    #
    # Wait, that's not right. Each term A_i * B_j is a 1x2 matrix, which has rank at most 1.
    # So each term contributes rank 1, and we have 9 terms, total rank 9? No, that's not right either.
    #
    # Let me reconsider. The tensor decomposition rank is the minimum number of rank-1 tensors
    # needed to express the matrix multiplication tensor.
    #
    # For the 3x5x6 case, the trivial decomposition has rank 90.
    # We can do better by exploiting structure.
    #
    # One approach: use the fact that the middle dimension is 5, which is odd.
    # We can write 5 = 2 + 2 + 1.
    # Partition A into blocks: A_top (2x5), A_bottom (1x5)
    # Partition B into blocks: B_left (5x3), B_right (5x3)
    # Then C_top = A_top * B_left (2x3), C_top_right = A_top * B_right (2x3)
    # C_bottom = A_bottom * B_left (1x3), C_bottom_right = A_bottom * B_right (1x3)
    #
    # For the 2x5x3 sub-problems, we can potentially use Strassen-like techniques.
    #
    # Actually, let's try a more systematic approach using the known Strassen decomposition.
    # We'll partition the problem into 2x2x2 sub-problems where possible.
    
    # Try to build a better decomposition using block Strassen
    # Partition: A into 2+1 rows, B into 3+3 columns, C into (2+1) x (3+3)
    
    # Let's try: use Strassen on the 2x2x2 sub-blocks
    # For rows 0,1 and cols 0,1,2 of B (i.e., 2x2x2 sub-problem)
    # This gives rank 7 instead of rank 8
    
    # Build decomposition using blocks:
    # Block 1: rows 0,1 of A; cols 0,1 of B; outputs (0,1)x(0,1) -> 2x2x2, rank 7
    # Block 2: rows 0,1 of A; cols 2,3 of B; outputs (0,1)x(2,3) -> 2x2x2, rank 7
    # Block 3: rows 0,1 of A; col 4 of B; outputs (0,1)x(4) -> 2x1x1, rank 2
    # Block 4: row 2 of A; cols 0,1 of B; outputs (2)x(0,1) -> 1x2x2, rank 2
    # Block 5: row 2 of A; cols 2,3 of B; outputs (2)x(2,3) -> 1x2x2, rank 2
    # Block 6: row 2 of A; col 4 of B; outputs (2)x(4) -> 1x1x1, rank 1
    
    # Total rank: 7 + 7 + 2 + 2 + 2 + 1 = 21? That seems too good to be true.
    # Wait, I'm confusing myself. Let me think more carefully.
    
    # The matrix multiplication tensor for A(n x m) * B(m x p) = C(n x p)
    # has rank at most n*m*p (trivial) and at least max(n*m, m*p, n*p) (lower bound).
    # For 3x5x6: lower bound is max(15, 30, 18) = 30.
    
    # Let's try to construct a decomposition with rank around 30-40.
    
    # Approach: Use the fact that we can share computations between different output entries.
    # For each column j of A and B, we compute the outer product A[:,j] * B[j,:].
    # These outer products are rank-1 matrices (as 3x6 matrices).
    # We need to express the sum of these 5 outer products as a tensor decomposition.
    
    # Actually, the tensor decomposition is for the multiplication operation itself,
    # not for a specific pair of matrices. So we need to find U, V, W such that
    # for ALL A and B, C = A*B is computed correctly.
    
    # Let me try a different approach: use bilinear algorithms.
    # The rank of the matrix multiplication tensor is the minimum number of
    # scalar multiplications needed to compute the product.
    
    # For 3x5x6, known upper bounds are around 40-50.
    # Let's try to find one using a systematic search.
    
    # Since a full search is infeasible, let's try to construct one using blocks.
    
    # Partition the middle dimension (size 5) into groups of size 2, 2, 1.
    # For each group of 2 middle indices, we have a 3x2x6 sub-problem.
    # We can decompose this further.
    
    # Actually, let's just try to use the trivial decomposition for now,
    # but with some optimization: combine terms where possible.
    
    # For now, return the trivial decomposition.
    # Future work: implement a proper block Strassen decomposition.
    
    return best_result

# EVOLVE-BLOCK-END
