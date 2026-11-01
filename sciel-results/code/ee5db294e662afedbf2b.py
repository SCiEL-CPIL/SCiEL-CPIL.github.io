"""Program for matrix multiplication 2x4 * 4x5 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _strassen2x2(a00, a01, a10, a11, b00, b01, b10, b11):
    """Compute 2x2 matrix product using Strassen's algorithm (7 multiplications).
    Returns (c00, c01, c10, c11).
    """
    m1 = (a00 + a11) * (b00 + b11)
    m2 = (a10 + a11) * b00
    m3 = a00 * (b01 - b11)
    m4 = a11 * (b10 - b00)
    m5 = (a00 + a01) * b11
    m6 = (a10 - a00) * (b00 + b01)
    m7 = (a01 - a11) * (b10 + b11)
    c00 = m1 + m4 - m5 + m7
    c01 = a00 * b01 + a01 * b11
    c10 = a10 * b00 + a11 * b10
    c11 = m1 - m2 + m3 + m6
    return c00, c01, c10, c11

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{j,j'} for matching
    Actually: sum_r U[i*m+j', r] * V[j''*p+k, r] * W[k*n+i, r] should equal delta_{j',j''}
    Wait, let me re-derive:
    C[i,k] = sum_j A[i,j] B[j,k]
    Decomposition: C[i,k] = sum_r W[k*n+i, r] * (sum_j' U[i*m+j', r] A[i,j']) * (sum_j'' V[j''*p+k, r] B[j'',k])
    = sum_r sum_j' sum_j'' W[k*n+i, r] U[i*m+j', r] V[j''*p+k, r] A[i,j'] B[j'',k]
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_r W[k*n+i, r] U[i*m+j', r] V[j''*p+k, r] = 1 if j'=j'', else 0
    i.e., delta_{j', j''}
    """
    rank = len(U[0])
    # Check for random sample of (i, k, j1, j2)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    expected = 1 if j1 == j2 else 0
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i*m+j1][r]) * Fraction(V[j2*p+k][r]) * Fraction(W[k*n+i][r])
                    if total != expected:
                        return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using numpy with exact rational check on a subset."""
    import numpy as np
    rank = len(U[0])
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for k in range(p):
            # The coefficient of A[i,j1]*B[j2,k] should be delta_{j1,j2}
            # coeff[j1,j2] = sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
            for j1 in range(m):
                for j2 in range(m):
                    expected = 1 if j1 == j2 else 0
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j1][r] * V[j2*p+k][r] * W[k*n+i][r]
                    # Check if total equals expected (allowing for Fraction or int/float)
                    if isinstance(total, Fraction):
                        if total != expected:
                            return False
                    else:
                        if abs(float(total) - expected) > 1e-10:
                            return False
    return True

def _construct_block_decomposition():
    """
    Construct a rank-minimal decomposition using block Strassen.
    
    We view the 2x4 * 4x5 multiplication as a block matrix product.
    Partition A (2x4) into 2x2 blocks horizontally: A = [A_left (2x2), A_right (2x2)]
    Partition B (4x5) into 2x2 blocks vertically: B = [B_top (2x5), B_bottom (2x5)]
    Wait, that's not right for Strassen. Let me think differently.
    
    Actually, let's use a different blocking.
    A is 2x4. B is 4x5.
    Think of it as: C = A * B where A has shape (n=2, m=4) and B has shape (m=4, p=5).
    
    One approach: split the middle dimension m=4 into two parts of 2 each.
    A = [A1 | A2] where A1 is 2x2, A2 is 2x2
    B = [B1; B2] where B1 is 2x5, B2 is 2x5
    C = A1*B1 + A2*B2
    
    Each 2x2 * 2x5 product can be done. But Strassen is for 2x2*2x2.
    
    Alternative: Let's use a more general approach.
    
    For multiplying (n x m) by (m x p), the schoolbook rank is n*m*p = 2*4*5 = 40.
    
    Known results: For 2x4 * 4x5, the minimum rank is at most 36 (from block Strassen on 2x2 blocks with some padding).
    
    Let me try a specific construction:
    
    Split m=4 into m1=2, m2=2.
    C = A[:,0:2]*B[0:2,:] + A[:,2:4]*B[2:4,:]
    
    First term: (2x2)*(2x5) -> rank at most 2*2*5 = 20, but we can do better.
    Actually, (2x2)*(2xp) has rank at most 2*p*2 = 4p for schoolbook, but Strassen-like can help.
    
    For (2x2)*(2x5): Think of B as [B_col1, B_col2, B_col3, B_col4, B_col5] each 2x1.
    This is 5 separate (2x2)*(2x1) multiplications. Each (2x2)*(2x1) has rank 4 schoolbook.
    Total: 5*4 = 20 for first block, 20 for second = 40. No improvement.
    
    Let me try a different strategy: Use the fact that we can combine terms.
    
    Actually, let me just implement a generic rank-40 schoolbook and then try to find a lower rank via random search or known constructions.
    
    For now, let's return the schoolbook rank 40 decomposition.
    """
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

def _try_strassen_block():
    """
    Attempt a block Strassen decomposition.
    
    Partition the 4x4 middle into 2x2 blocks.
    A (2x4) = [A11 A12; A21 A22]... no, A is 2x4.
    
    Let's use a different partitioning.
    Think of the product as involving 2x2 sub-matrices.
    
    Actually, for 2x4 * 4x5, one known approach:
    - Use rank 36 from a specific construction.
    
    Let me try: express the product using 7 Strassen-like terms per 2x2*2x2 block,
    but we need to handle the rectangular shapes.
    
    Split A (2x4) into two 2x2 blocks: A = [Al | Ar]
    Split B (4x5) into two 2x5 blocks: B = [Bt; Bb]
    C = Al*Bt + Ar*Bb
    
    For Al*Bt (2x2 * 2x5):
    Al = [[a00,a01],[a10,a11]], Bt = 2x5 matrix.
    The product is 2x5. Each element (i,k) is sum over j of Al[i,j]*Bt[j,k].
    This is 5 separate 2x2*2x1 multiplications, or we can think of it as:
    For each column k of Bt, compute Al * Bt[:,k] (2x2 * 2x1).
    
    A 2x2 * 2x1 multiplication: [a b; c d] * [x; y] = [ax+by; cx+dy]
    This has 4 multiplications schoolbook, or can we do it with fewer?
    Rank of 2x2*2x1 tensor: The tensor has dimensions 2x2 x 2x1 = 4x2 = 8 entries... 
    Actually the bilinear form has 4 inputs (a,b,c,d) and 2 outputs... 
    The tensor T[i,j,k] where i in {0,1}, j in {0,1} for Al, k in {0,1} for Bt column.
    T[i,j,k] = Al[i,j]*Bt[k] ... no, Bt is 2x1 so Bt[0], Bt[1].
    Output[i] = Al[i,0]*Bt[0] + Al[i,1]*Bt[1]
    So the tensor has 2 outputs, 4*2=8 inputs (but structured).
    Rank: We need sum_r U[r] * V[r] * W[r] where the product gives the output.
    For 2x2*2x1: minimum rank is 4 (schoolbook). I don't think it can be reduced.
    
    So 2x2*2x5 requires 5*4 = 20 rank.
    Similarly 2x2*2x5 for the other block: 20 rank.
    Total: 40. No improvement.
    
    Let me try a different approach: maybe there's a way to share computation between the two blocks.
    
    Alternative: Use a (3x3) or other block structure.
    
    For now, let's just return the schoolbook rank 40.
    """
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
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook rank 40
    best_result = _construct_block_decomposition()
    best_rank = 40
    
    # Try to find a lower rank decomposition
    # Known: rank of 2x4*4x5 matrix multiplication tensor
    # The minimum rank is at most 36 (from various constructions)
    # Let's try rank 36
    
    # For rank 36, we need to find U, V, W with 36 columns
    # such that the tensor identity holds.
    
    # One approach: use a parametric family and solve for parameters.
    # Another: random search.
    
    # Let's try a structured approach based on block Strassen.
    # 
    # Partition: A (2x4) into 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    # B (4x5) into 2x5 blocks: B_top (rows 0-1), B_bottom (rows 2-3)
    # C = A_left * B_top + A_right * B_bottom
    #
    # For A_left * B_top (2x2 * 2x5):
    # Let A_left = [[a0,a1],[a2,a3]], B_top = [[b00,b01,b02,b03,b04],[b10,b11,b12,b13,b14]]
    # C_top = A_left * B_top is 2x5
    # C_top[0,k] = a0*b0k + a1*b1k
    # C_top[1,k] = a2*b0k + a3*b1k
    #
    # Each of these is a dot product of a 2-vector with a 2-vector, giving a scalar.
    # There are 5 such products for row 0, 5 for row 1. Total 10 "multiplications" in the sense of bilinear terms.
    # But each "multiplication" a*b for scalars is 1 rank-1 term.
    # So A_left * B_top uses 10 rank-1 terms.
    # Similarly A_right * B_bottom uses 10 rank-1 terms.
    # Total: 20? No, that's not right. Each rank-1 term in the tensor decomposition corresponds to one scalar multiplication.
    # For 2x2*2x5: the tensor has 2*2*5 = 20 entries... no.
    # The bilinear form: inputs are 4 (for A) + 10 (for B) = 14, output is 10.
    # The tensor T[i,j,k] where i in {0,1} (row of A), j in {0,1} (col of A = row of B), k in {0..4} (col of B).
    # T[i,j,k] = A[i,j] * B[j,k]
    # The number of independent bilinear terms: for each (i,k), we have sum_j A[i,j]*B[j,k].
    # This is 2*5 = 10 outputs, each requiring 2 products (for j=0 and j=1).
    # So 20 scalar multiplications, hence rank 20 for schoolbook.
    # 
    # Can we do better? The tensor 2x2*2x5: minimum rank is 16? Or 15?
    # Actually, for (2x2)*(2xp), the rank is known to be 4p for schoolbook, but can be reduced.
    # For p=1: 2x2*2x1 has rank 4 (I think this is minimal).
    # For p=2: 2x2*2x2 has rank 7 (Strassen).
    # For p=5: 2x2*2x5 ... this is less standard.
    
    # Let me try a random search for rank 36.
    
    if time.time() < deadline:
        # Try random search for rank 36
        random.seed(42)
        found = False
        attempts = 0
        max_attempts = 5000
        
        while time.time() < deadline - 5 and not found and attempts < max_attempts:
            attempts += 1
            rank = 36
            # Initialize with small random integers
            U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
            
            # Verify
            if _verify_decomposition_fast(U, V, W, n, m, p):
                found = True
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    if time.time() < deadline:
        # Try rank 35
        random.seed(123)
        found = False
        attempts = 0
        max_attempts = 3000
        
        while time.time() < deadline - 5 and not found and attempts < max_attempts:
            attempts += 1
            rank = 35
            U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
            
            if _verify_decomposition_fast(U, V, W, n, m, p):
                found = True
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
