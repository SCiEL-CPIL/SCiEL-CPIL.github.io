"""Find rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product
from math import gcd
from functools import reduce

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Check all entries of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    if j == l:
                        target = 1.0
                    else:
                        target = 0.0
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                    if abs(val - target) > 1e-9:
                        return False
    return True

def _to_fraction_matrix(M, max_den=10000):
    """Convert a float matrix to a fraction matrix if possible."""
    result = []
    for row in M:
        frac_row = []
        for x in row:
            if abs(x - round(x)) < 1e-9:
                frac_row.append(Fraction(int(round(x))))
            else:
                f = Fraction(x).limit_denominator(max_den)
                if abs(x - float(f)) > 1e-8:
                    return None
                frac_row.append(f)
        result.append(frac_row)
    return result

def _lcm_list(lst):
    """Compute LCM of a list of integers."""
    if not lst:
        return 1
    result = 1
    for x in lst:
        result = result * x // gcd(result, x)
    return result

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(g+h)
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+g)
    # P7 = (b-d)(f+h)
    # Result:
    # [a*e+b*g, a*f+b*h]
    # [c*e+d*g, c*f+d*h]
    
    # U corresponds to A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # V corresponds to B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # W corresponds to C indices: (0,0), (0,1), (1,0), (1,1)
    
    U = [
        [1, 0, 1, 0, 1, 0, 1],   # a
        [0, 0, 0, 0, 1, 0, -1],  # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, 0],   # d
    ]
    V = [
        [1, 0, 0, 0, 0, 1, 1],   # e
        [1, 0, 1, 0, 0, 0, 1],   # f
        [0, 1, 0, 1, 0, 1, 0],   # g
        [1, 1, -1, -1, 1, 0, -1] # h
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],   # (0,1)
        [0, 1, 0, 0, 0, 1, 0],   # (1,0)
        [1, 1, 1, 1, 0, 0, 0]    # (1,1)
    ]
    return U, V, W

def _build_block_strassen(n, m, p):
    """
    Build a block Strassen decomposition for n x m x p matrix multiplication.
    We split the m dimension into blocks of size b, and the p dimension into blocks of size c.
    For each combination of blocks, we use Strassen if the block sizes are 2x2x2.
    
    For 2x4x4: split m=4 into two 2x2 blocks, p=4 into two 2x2 blocks.
    The 2x2 blocks in the first dimension (n=2) are just the full n dimension.
    
    Actually, let's think more carefully. We have A (n x m), B (m x p), C (n x p).
    Split A into column blocks: A1 (n x 2), A2 (n x 2)
    Split B into row/column blocks: B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2)
    Split C into column blocks: C1 (n x 2), C2 (n x 2)
    
    C1 = A1 * B11 + A2 * B21
    C2 = A1 * B12 + A2 * B22
    
    For each of these 4 products (A_i * B_jk), if A_i is 2x2 and B_jk is 2x2,
    we can use Strassen's rank-7 algorithm.
    
    So we have 4 products, each with rank 7, total rank = 28.
    """
    n, m, p = 2, 4, 4
    b = 2  # block size for m
    c = 2  # block size for p
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # There are (m//b) blocks in m direction and (p//c) blocks in p direction
    # For each (block_m, block_p) pair, we do a 2x2x2 multiplication
    for bm in range(m // b):
        for bp in range(p // c):
            U_s, V_s, W_s = _strassen_2x2x2()
            
            # U_s is 4x7, corresponding to A indices within this block
            # A indices: row i (0..1), col j (bm*2..bm*2+1)
            # V_s is 4x7, corresponding to B indices within this block
            # B indices: row j (bm*2..bm*2+1), col k (bp*2..bp*2+1)
            # W_s is 4x7, corresponding to C indices within this block
            # C indices: row i (0..1), col k (bp*2..bp*2+1)
            
            U_blocks.append(U_s)
            V_blocks.append(V_s)
            W_blocks.append(W_s)
    
    # Now combine into full matrices
    rank = 7 * (m // b) * (p // c)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    offset = 0
    for bm in range(m // b):
        for bp in range(p // c):
            U_s, V_s, W_s = _strassen_2x2x2()
            for i in range(n):
                for j in range(b):
                    u_idx = i * b + j
                    for r in range(7):
                        U[i * m + bm * b + j][offset + r] = U_s[u_idx][r]
            for j in range(b):
                for k in range(c):
                    v_idx = j * c + k
                    for r in range(7):
                        V[(bm * b + j) * p + bp * c + k][offset + r] = V_s[v_idx][r]
            for k in range(c):
                for i in range(n):
                    w_idx = k * n + i
                    for r in range(7):
                        W[(bp * c + k) * n + i][offset + r] = W_s[w_idx][r]
            offset += 7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Try block Strassen decomposition (rank 28 for 2x4x4)
    U_s, V_s, W_s = _build_block_strassen(n, m, p)
    if _verify_decomposition(U_s, V_s, W_s, n, m, p):
        best_result = {"U": U_s, "V": V_s, "W": W_s}
        best_rank = len(U_s[0])
    
    # Now try to improve further with random search for lower ranks
    # Try ranks from best_rank-1 down to 20
    for rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        max_attempts = 2000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            seed = attempt * 7919 + rank * 104729 + 12345
            np.random.seed(seed)
            
            # Generate U and V with entries in {-2, -1, 0, 1, 2}
            sparsity = 0.15 + 0.35 * (attempt % 10) / 10.0
            
            U_r = np.where(np.random.rand(n * m, rank) < sparsity, 
                          np.random.choice([-2, -1, 1, 2], size=(n * m, rank)), 0)
            V_r = np.where(np.random.rand(m * p, rank) < sparsity, 
                          np.random.choice([-2, -1, 1, 2], size=(m * p, rank)), 0)
            
            U_np = U_r.astype(float)
            V_np = V_r.astype(float)
            
            W_np = np.zeros((p * n, rank))
            valid = True
            
            for i in range(n):
                if not valid:
                    break
                for k in range(p):
                    if not valid:
                        break
                    
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U_np[i * m + j, r] * V_np[j * p + k, r]
                    
                    try:
                        W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-8:
                            valid = False
                            break
                        W_np[k * n + i, :] = W_col
                    except np.linalg.LinAlgError:
                        valid = False
                        break
            
            if valid:
                frac_w = _to_fraction_matrix(W_np, max_den=5000)
                
                if frac_w is not None:
                    dens = [f.denominator for row in frac_w for f in row]
                    common_den = _lcm_list(dens)
                    
                    if common_den <= 100000:
                        W_int = [[int(f * common_den) for f in row] for row in frac_w]
                        U_int = [[int(x) for x in row] for row in U_np.tolist()]
                        V_int = [[int(x) for x in row] for row in V_np.tolist()]
                        
                        if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                            best_result = {"U": U_int, "V": V_int, "W": W_int}
                            best_rank = rank
                            break
    
    return best_result
# EVOLVE-BLOCK-END
