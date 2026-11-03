"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]
_TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _schoolbook():
    """Standard O(nmp) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _verify(U, V, W):
    """
    Verify that the decomposition exactly matches matrix multiplication.
    C[i,k] = sum_j A[i,j] * B[j,k]
    W index: k*n + i (trace convention)
    """
    n, m, p = DIMENSIONS
    # Build the multiplication tensor T[a,b,c] where
    #   a = i*m+j (A index), b = j*p+k (B index), c = k*n+i (C index)
    # T[a,b,c] = 1 iff a[1]==b[0] and c[1]==a[0] and c[0]==b[1]
    # i.e., j from A matches j from B, i from A matches i in C, k from B matches k in C
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    rank = U_mat.shape[1]
    if V_mat.shape[1] != rank or W_mat.shape[1] != rank:
        return False
    if U_mat.shape[0] != n * m or V_mat.shape[0] != m * p or W_mat.shape[0] != p * n:
        return False
    
    # Compute the tensor from decomposition: T_decomp[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Compare with true tensor T_true[a,b,c]
    
    # Efficient check: for each pair (a,b), compute sum_r U[a,r]*V[b,r] and check
    # that it equals sum over valid c of T_true[a,b,c]*W[c,r]... 
    # Actually simpler: build full tensor and compare.
    
    # True tensor
    T_true = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0
    
    # Decomposed tensor
    # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = (U @ V.T)[a,b] * ... no, it's a 3-way product.
    # T_decomp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
    T_decomp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
    
    return np.allclose(T_decomp, T_true, atol=1e-9, rtol=1e-9)

def _try_winograd_2x2_blocks():
    """
    Apply Winograd-style 2x2 block transforms to reduce rank.
    
    For 3x4 x 4x4:
    - n=3 rows of A, p=4 cols of B
    - We can group the 4 columns of B into 2x2 blocks: (k0,k1) and (k2,k3)
    - For each 2x2 block, Winograd reduces 4 multiplications to 3.
    
    Winograd F_2 transform for 2x2:
      A_2x2 * B_2x2: 4 mults -> 3 mults
      U: [a0+a1, a0, a1]  (3 values from A)
      V: [b0+b1, b0, b1]  (3 values from B)  
      W: reconstruct C = [c00,c01;c10,c11] from 3 products
      
    For our case, we process the j dimension (4 cols of A / 4 rows of B) in 2x2 blocks.
    
    Block structure for j: (j0,j1) and (j2,j3)
    
    For each i (3 values) and each 2x2 j-block:
      Standard: 2 j-values * 4 k-values = 8 mults per i
      Winograd on j: 3 j-values * 4 k-values = 12... that's worse.
      
    Actually Winograd works best when applied to the output dimension.
    Let me reconsider.
    
    For C[i,k] where k is in a 2x2 block (k0,k1):
      C[i,k0] = sum_j A[i,j]*B[j,k0]
      C[i,k1] = sum_j A[i,j]*B[j,k1]
      
      Winograd on k: 2 outputs -> 3 products
      But we have 4 j-values, so: 4 j * 3 k-products = 12 vs 4*2=8 standard. Worse!
    
    Winograd on j (4 -> 2 blocks of 2):
      For each 2x2 j-block and each k:
      C[i,k] gets contribution from 2 j's.
      Winograd: 2 j's -> 3 products per k.
      3 k's... wait p=4.
      
      Per i, per 2x2 j-block: 3 (winograd j) * 4 (k) = 12 mults
      Standard: 2 (j) * 4 (k) = 8 mults. Worse!
    
    Winograd is beneficial when both dimensions are >= 3.
    For 2x2, it's 3 mults vs 4, saving 1 per 2x2 block.
    
    Let's try: group j into 2 blocks of 2, and k into 2 blocks of 2.
    For each (i, j_block, k_block): 2x2 sub-problem.
    Winograd: 3 mults instead of 4.
    
    Total: 3 (i) * 2 (j_blocks) * 2 (k_blocks) * 3 (winograd) = 36
    vs standard: 3*4*4 = 48. Saves 12.
    """
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # j blocks: (0,1), (2,3) -> 2 blocks
    # k blocks: (0,1), (2,3) -> 2 blocks
    
    # For each (i, jb, kb): 3 Winograd multiplications
    # Total rank = 3 * 2 * 2 * 3 = 36
    
    rank = n * 2 * 2 * 3  # 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for jb in range(2):  # j block index
            j0 = jb * 2
            j1 = jb * 2 + 1
            for kb in range(2):  # k block index
                k0 = kb * 2
                k1 = kb * 2 + 1
                
                # Winograd 2x2: 
                # A_2x2 = [[A[i,j0], A[i,j1]]] (row vector, 1x2)
                # B_2x2 = [[B[j0,k0], B[j0,k1]], [B[j1,k0], B[j1,k1]]] (2x2)
                # C_1x2 = A_1x2 * B_2x2 = [c0, c1]
                #
                # Winograd for 1x2 * 2x2 -> 1x2:
                # Standard: 4 mults: A[j0]*B[j0,k0] + A[j1]*B[j1,k0], A[j0]*B[j0,k1] + A[j1]*B[j1,k1]
                # Winograd: 3 mults
                #   p1 = (A[j0]+A[j1]) * (B[j0,k0]+B[j1,k0])
                #   p2 = A[j0] * B[j0,k1]
                #   p3 = A[j1] * B[j1,k0]
                #   c0 = p1 - p2 - p3
                #   c1 = p1 - p2  ... wait let me redo this.
                #
                # Actually for 1x2 * 2x2:
                # c0 = A0*B00 + A1*B10
                # c1 = A0*B01 + A1*B11
                #
                # Winograd:
                # m1 = (A0+A1)*(B00+B10)
                # m2 = A0*B01
                # m3 = A1*B10
                # c0 = m1 - m2 - m3  ... no
                # c0 = A0*B00 + A1*B10 = (A0+A1)*(B00+B10) - A0*B10 - A1*B00
                # That's 4 mults again.
                #
                # For 1x2 * 2x2, Winograd doesn't help (need at least 2x2 * 2x2 for 4->3).
                #
                # Let me reconsider. The standard Winograd is for 2x2 * 2x2 -> 2x2.
                # In our case, for fixed i, we have:
                #   C[i,k0] = sum_{j in block} A[i,j] * B[j,k0]
                #   C[i,k1] = sum_{j in block} A[i,j] * B[j,k1]
                #
                # This is a 1x2 (A row) times 2x2 (B submatrix) = 1x2.
                # Winograd for this: 
                #   m1 = (A0+A1) * (B00+B10)
                #   m2 = (A0+A1) * (B01+B11)  
                #   m3 = A0 * B10
                #   m4 = A1 * B01
                #   c0 = m1 - m3 - A1*B00... 
                #
                # This is getting complicated. Let me use the known result:
                # 2x2 matrix multiplication needs 7 mults (Strassen).
                # 1x2 * 2x2 needs... let's think.
                # c0 = A0*B00 + A1*B10
                # c1 = A0*B01 + A1*B11
                # Minimum rank: at least 3 (since we have 2 independent outputs each depending on 2 inputs).
                # Actually, can we do it in 3?
                # m1 = A0*(B00+B11)
                # m2 = A1*(B10+B01)
                # m3 = (A0+A1)*(B00+B10)
                # c0 = m1 - m3 + A1*B10... no, that introduces more mults.
                #
                # For 1x2 * 2x2, the tensor rank is 4 (same as standard). No savings.
                #
                # So Winograd 2x2 doesn't help for 1x2 * 2x2.
                # We need 2x2 * 2x2 for Strassen.
                #
                # Alternative: group i into pairs? n=3, so we can't pair evenly.
                # Group i as (0,1) and handle i=2 separately.
                #
                # For (i0,i1) x (j0,j1) x (k0,k1): 2x2 * 2x2 -> 2x2
                # Strassen: 7 mults instead of 8.
                #
                # Let's try:
                # i blocks: (0,1), (2) -> 2 blocks (one 2x2, one 1x2)
                # j blocks: (0,1), (2,3) -> 2 blocks
                # k blocks: (0,1), (2,3) -> 2 blocks
                #
                # For i-block (0,1), j-block jb, k-block kb: 2x2*2x2 -> 2x2, Strassen 7 mults
                # For i=2, j-block jb, k-block kb: 1x2*2x2 -> 1x2, 4 mults
                #
                # Total: 2 (i-blocks with Strassen) * 2 (j-blocks) * 2 (k-blocks) * 7 + 1 * 2 * 2 * 4
                #       = 56 + 16 = 72. Worse than 48!
                
                # This approach isn't working. Let me try a different strategy.
                pass
                col += 3  # placeholder
    
    # The Winograd approach isn't straightforward for these dimensions.
    # Let's try a different approach: random search with verification.
    return None

def _try_random_reduction(target_rank, seed=42, max_iters=5000):
    """
    Try to find a decomposition of the given rank using random initialization
    and iterative refinement.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, target_rank)
    V = rng.randn(m * p, target_rank)
    W = rng.randn(p * n, target_rank)
    
    # Build true tensor
    T_true = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0
    
    best_error = float('inf')
    
    for it in range(max_iters):
        # Check time
        if time.time() > _TIME_LIMIT - 10:
            break
        
        # Compute current tensor
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_decomp - T_true)
        
        if error < best_error:
            best_error = error
        
        if error < 1e-10:
            return U, V, W, target_rank
        
        # Gradient-free update: perturb one matrix at a time
        mat = it % 3
        if mat == 0:
            U += 0.01 * rng.randn(n * m, target_rank)
        elif mat == 1:
            V += 0.01 * rng.randn(m * p, target_rank)
        else:
            W += 0.01 * rng.randn(p * n, target_rank)
    
    return None

def _try_systematic_reduction(start_rank=48, end_rank=30):
    """
    Try ranks from start_rank down to end_rank, using a structured approach.
    """
    n, m, p = DIMENSIONS
    
    for rank in range(start_rank, end_rank - 1, -1):
        if time.time() > _TIME_LIMIT - 10:
            break
        
        # Try a few seeds for random initialization
        for seed in range(5):
            result = _try_random_reduction(rank, seed=seed, max_iters=200)
            if result is not None:
                U, V, W, r = result
                if _verify(U, V, W):
                    return U, V, W, r
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.
    Starts from schoolbook (rank 48) and attempts to reduce rank.
    """
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    # Try Winograd-based approach
    winograd_result = _try_winograd_2x2_blocks()
    if winograd_result is not None:
        Uw, Vw, Ww, rw = winograd_result
        if _verify(Uw, Vw, Ww) and rw < best_rank:
            best_rank = rw
            best_result = {"U": Uw.tolist(), "V": Vw.tolist(), "W": Ww.tolist()}
    
    # Try systematic rank reduction
    if time.time() < _TIME_LIMIT - 20:
        reduction_result = _try_systematic_reduction(start_rank=best_rank - 1, end_rank=max(20, best_rank - 20))
        if reduction_result is not None:
            Ur, Vr, Wr, rr = reduction_result
            if _verify(Ur, Vr, Wr) and rr < best_rank:
                best_rank = rr
                best_result = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    
    # Ensure we return before the time limit
    if time.time() > _TIME_LIMIT - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
