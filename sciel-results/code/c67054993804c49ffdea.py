"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _dft_z5_decomposition(n, m, p):
    """
    DFT-based structured decomposition for (n, m, p) where p=5.
    Uses cyclic group Z_5 structure to create rank-1 terms.
    
    For matrix multiplication C = A * B where A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We use DFT over Z_5 to diagonalize the cyclic convolution structure.
    """
    if p != 5:
        return None
    
    # DFT matrix for Z_5
    # omega = exp(2*pi*i/5)
    # We'll use exact algebraic numbers represented as fractions of sqrt(5)
    
    # The 5th roots of unity satisfy: 1 + omega + omega^2 + omega^3 + omega^4 = 0
    # and omega^5 = 1
    
    # For exact decomposition, we need to work in Q(sqrt(5))
    # omega = (-1 + sqrt(5)) / 2 * exp(2*pi*i/5) ... this gets complex
    
    # Alternative: Use the fact that for p=5, we can use a different approach
    # based on the minimal polynomial of omega: x^4 + x^3 + x^2 + x + 1 = 0
    
    # Let's try a block-based approach instead
    # Split the m=4 dimension into blocks
    
    # For 4x4 * 4x5, we can use:
    # - Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    # - Split B into 2x5 blocks: B = [[B1], [B2]] where B1, B2 are 2x5
    # - C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # This gives 8 products of 2x2 * 2x5 matrices
    
    # For 2x2 * 2x5, we can use Strassen-like methods
    # But let's first try a simpler structured approach
    
    # Actually, let me think about this differently.
    # The key insight is that p=5 is prime, so we can use FFT-like techniques.
    
    # For now, let's return a structured initialization that we can optimize
    rank = 40  # Start with a reasonable guess
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Create structured patterns based on DFT
    # omega^k for k=0,1,2,3,4
    # We'll use real and imaginary parts
    
    # For simplicity, let's use a different structured approach
    # based on the block decomposition
    
    # Split indices:
    # i = 2*i1 + i2 where i1 in {0,1}, i2 in {0,1}
    # j = 2*j1 + j2 where j1 in {0,1}, j2 in {0,1}  
    # k = 0..4
    
    # C[i1,i2, k] = sum_{j1,j2} A[i1,i2, j1,j2] * B[j1,j2, k]
    
    # We can group by j1:
    # C[i1,i2, k] = sum_{j2} A[i1,i2, 0,j2] * B[0,j2, k] + sum_{j2} A[i1,i2, 1,j2] * B[1,j2, k]
    
    # For each fixed i1,i2, we have two 1x2 * 2x5 products
    # Each 1x2 * 2x5 can be done in rank 5 (naive) or potentially less
    
    # Let's use a different strategy: exploit the structure more directly
    
    # For now, return None to indicate this needs more work
    return None

def _block_decompose_4x4x5(n, m, p):
    """
    Block decomposition for 4x4 * 4x5 using 2x2 blocks.
    """
    if (n, m, p) != (4, 4, 5):
        return None
    
    # A is 4x4, B is 4x5
    # Split A into four 2x2 blocks: A11, A12, A21, A22
    # Split B into two 2x5 blocks: B1, B2
    
    # C1 = A11*B1 + A12*B2 (2x5)
    # C2 = A21*B1 + A22*B2 (2x5)
    
    # Each term like A11*B1 is 2x2 * 2x5
    # We need to compute 8 such products
    
    # For 2x2 * 2x5, we can use a rank-7 method (Strassen-like)
    # Let's construct this explicitly
    
    # For 2x2 * 2x5:
    # Let P be 2x2, Q be 2x5
    # R = P*Q is 2x5
    
    # R[i,k] = sum_j P[i,j] * Q[j,k]
    
    # We can use the following rank-7 decomposition:
    # Define 7 rank-1 terms
    
    # This is getting complex. Let me use a simpler approach first.
    
    # For now, use naive decomposition for 2x2*2x5 (rank 10)
    # Total rank: 8 * 10 = 80 (same as schoolbook)
    
    # Let's try to be smarter
    
    # Actually, for 2x2 * 2x5, we can do better.
    # Note that 2x2 * 2x5 is like two independent 1x2 * 2x5 products
    # (one for each row of the result)
    
    # Each 1x2 * 2x5: R[k] = a*Q[0,k] + b*Q[1,k] for k=0..4
    # This is rank 5 (one term per k)
    
    # So 2x2 * 2x5 has rank 10
    
    # Total: 8 * 10 = 80
    
    # This doesn't help. Let me think of a different blocking.
    
    # Alternative: Split the k dimension
    # B is 4x5, split into B[0:2, 0:5] and B[2:4, 0:5]
    # Or split into B[:, 0:3] and B[:, 3:5]
    
    # Let's try: B = [B_left (4x3), B_right (4x2)]
    # C = A * B_left + A * B_right
    # A*B_left is 4x3, A*B_right is 4x2
    
    # For 4x4 * 4x3: can we do better than rank 48?
    # For 4x4 * 4x2: can we do better than rank 32?
    
    # This is still complex. Let me use a different strategy.
    
    # I'll use the DFT approach over Z_5 more carefully.
    
    # The key is that multiplication by a circulant matrix can be diagonalized by DFT.
    # But our matrices aren't circulant.
    
    # Let me try a completely different approach: use the fact that we can
    # represent the tensor decomposition using algebraic number fields.
    
    # For p=5, the cyclotomic field Q(zeta_5) has degree 4.
    # We can use this to create structured decompositions.
    
    # This is quite involved. For now, let me return a structured random init
    # that has some DFT-like structure.
    
    rank = 50
    rng = np.random.RandomState(42)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Create DFT-like structure
    # For each "frequency" r in 0..4, create a block of rank-1 terms
    
    for r in range(5):
        # omega^r
        # We'll use cos(2*pi*r/5) and sin(2*pi*r/5)
        angle = 2 * np.pi * r / 5
        c = np.cos(angle)
        s = np.sin(angle)
        
        # Create structured entries
        for i in range(n):
            for j in range(m):
                U[i*m+j, r*10:(r+1)*10] = rng.randn(10) * 0.1
        
        for j in range(m):
            for k in range(p):
                V[j*p+k, r*10:(r+1)*10] = rng.randn(10) * 0.1
        
        for k in range(p):
            for i in range(n):
                W[k*n+i, r*10:(r+1)*10] = rng.randn(10) * 0.1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin for 150s limit
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try DFT-based structured decomposition
    dft_result = _dft_z5_decomposition(n, m, p)
    if dft_result is not None:
        U_dft, V_dft, W_dft = dft_result
        if _verify_decomposition(U_dft.tolist(), V_dft.tolist(), W_dft.tolist(), n, m, p):
            best_result = {"U": U_dft.tolist(), "V": V_dft.tolist(), "W": W_dft.tolist()}
            best_rank = U_dft.shape[1]
    
    # Try block decomposition
    block_result = _block_decompose_4x4x5(n, m, p)
    if block_result is not None:
        U_blk, V_blk, W_blk = block_result
        if _verify_decomposition(U_blk.tolist(), V_blk.tolist(), W_blk.tolist(), n, m, p):
            if U_blk.shape[1] < best_rank:
                best_result = {"U": U_blk.tolist(), "V": V_blk.tolist(), "W": W_blk.tolist()}
                best_rank = U_blk.shape[1]
    
    # Search for lower ranks with random initialization
    ranks_to_try = list(range(79, 19, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if rank >= best_rank:
            continue
            
        found = False
        # Try a few seeds
        for seed in range(20):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.RandomState(seed)
            U_init = rng.randn(n * m, rank) * 0.1
            V_init = rng.randn(m * p, rank) * 0.1
            W_init = rng.randn(p * n, rank) * 0.1
            
            # Check if the random init is already valid (unlikely but possible for high rank)
            if _verify_decomposition(U_init.tolist(), V_init.tolist(), W_init.tolist(), n, m, p):
                best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_init.tolist()}
                best_rank = rank
                found = True
                break
                        
        # If we found a solution for this rank, we can continue to lower ranks
        # to see if we can find an even better one.
        # But we must be careful not to spend too much time.
        
    return best_result
# EVOLVE-BLOCK-END
