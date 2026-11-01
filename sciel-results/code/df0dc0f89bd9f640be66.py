"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    def to_list(mat):
        return mat.tolist()
    
    def verify_fast(U, V, W, tolerance=1e-9):
        """Fast verification using float, for initial screening."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(actual - 1.0) > tolerance:
                        return False
        
        # Check all zero entries (small enough for n=2,m=3,p=7)
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k1 = divmod(b, p)
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    if not (j1 == j2 and i1 == i2 and k1 == k2):
                        actual = 0.0
                        for r in range(rank):
                            actual += U[a, r] * V[b, r] * W[c, r]
                        if abs(actual) > tolerance:
                            return False
        return True
    
    def verify_exact(U, V, W):
        """Verify that U, V, W correctly decompose the tensor using exact arithmetic."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Check all n*m*p required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = Fraction(0)
                    for r in range(rank):
                        actual += Fraction(int(round(U[i * m + j, r] * 2)), 2) * \
                                   Fraction(int(round(V[j * p + k, r] * 2)), 2) * \
                                   Fraction(int(round(W[k * n + i, r] * 2)), 2)
                    if actual != Fraction(1):
                        return False
        
        # Check all zero entries
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k1 = divmod(b, p)
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    if not (j1 == j2 and i1 == i2 and k1 == k2):
                        actual = Fraction(0)
                        for r in range(rank):
                            actual += Fraction(int(round(U[a, r] * 2)), 2) * \
                                       Fraction(int(round(V[b, r] * 2)), 2) * \
                                       Fraction(int(round(W[c, r] * 2)), 2)
                        if actual != Fraction(0):
                            return False
        return True
    
    # Build schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def try_strassen_like_search():
        """Try Strassen-like block decomposition to reduce rank."""
        if time.time() > deadline:
            return None
        
        # Key insight: For m=3, we can use bilinear combinations
        # The matrix product C = A*B where A is 2x3, B is 3x7
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Strategy: For each j, we need to compute sum over j
        # We can use the fact that 3 = 2 + 1, so we might be able to
        # combine some terms using Strassen-like ideas
        
        # Let's try a specific construction:
        # For j=0,1,2, we have contributions A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        # One approach: use the identity (x+y)(z+w) = xz + xw + yz + yw
        # to combine terms. But this increases rank, not decreases it.
        
        # Better: look for known algorithms for small matrix multiplication
        
        # For 2x3 * 3x7, the rank is at most 42 (schoolbook)
        # Known bounds: rank >= n*m + m*p + n*p - m = 6 + 21 + 14 - 3 = 38
        
        # Let's try to find rank 41, 40, 39, 38
        
        # Approach: Start from schoolbook and try to merge pairs of terms
        # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists
        # a rank-1 decomposition that covers both
        
        # Actually, let's try a different approach: use the structure that
        # for fixed i, we have a 1x3 vector A[i,:] multiplied by 3x7 matrix B
        # This is equivalent to computing 7 outputs from 3 inputs
        
        # For each i, the computation is: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
        
        # We can view this as a bilinear map from (a0,a1,a2) x (b0k,b1k,b2k for all k) to c_k
        
        # Let's try to find a rank-3 decomposition for the 3-input to 7-output map
        # that works for both i=0 and i=1 simultaneously
        
        # Actually, the key is that A[i,j] and B[j,k] are independent variables
        # so we need the full tensor structure
        
        # Let me try a more systematic approach: use the fact that the tensor
        # has a specific sparsity pattern and try to find a minimal decomposition
        
        # For now, let's try random search with better initialization
        np.random.seed(42)
        
        half_ints = [Fraction(x, 2) for x in range(-6, 7)]  # -3 to 3 in steps of 0.5
        half_ints_float = [float(h) for h in half_ints]
        
        # Try ranks from 41 down to 38
        for target_rank in [41, 40, 39, 38]:
            if time.time() > deadline:
                break
            
            # Number of attempts based on rank
            max_attempts = 200 if target_rank >= 40 else 100
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                # Initialize with schoolbook terms, then modify
                U = np.zeros((n * m, target_rank), dtype=np.float64)
                V = np.zeros((m * p, target_rank), dtype=np.float64)
                W = np.zeros((p * n, target_rank), dtype=np.float64)
                
                # Copy first min(target_rank, 42) schoolbook terms
                for r_idx in range(min(target_rank, rank_schoolbook)):
                    i = r_idx // (m * p)
                    rem = r_idx % (m * p)
                    j = rem // p
                    k = rem % p
                    
                    U[i * m + j, r_idx] = 1.0
                    V[j * p + k, r_idx] = 1.0
                    W[k * n + i, r_idx] = 1.0
                
                # For the "extra" ranks (if target_rank < 42, we need to merge)
                # This approach won't work directly. Let's try a different strategy.
                
                # Instead, let's try to find a sparse decomposition from scratch
                U = np.zeros((n * m, target_rank), dtype=np.float64)
                V = np.zeros((m * p, target_rank), dtype=np.float64)
                W = np.zeros((p * n, target_rank), dtype=np.float64)
                
                # Each rank-1 term should be sparse
                for r_idx in range(target_rank):
                    # Choose which (i,j) positions to use in U
                    num_u = np.random.randint(1, 4)
                    u_positions = np.random.choice(n * m, size=num_u, replace=False)
                    for pos in u_positions:
                        U[pos, r_idx] = float(np.random.choice(half_ints_float))
                    
                    # Choose which (j,k) positions to use in V
                    num_v = np.random.randint(1, 4)
                    v_positions = np.random.choice(m * p, size=num_v, replace=False)
                    for pos in v_positions:
                        V[pos, r_idx] = float(np.random.choice(half_ints_float))
                    
                    # Choose which (k,i) positions to use in W
                    num_w = np.random.randint(1, 4)
                    w_positions = np.random.choice(p * n, size=num_w, replace=False)
                    for pos in w_positions:
                        W[pos, r_idx] = float(np.random.choice(half_ints_float))
                
                # Quick check
                if verify_fast(U, V, W):
                    # Exact check
                    if verify_exact(U, V, W):
                        return U, V, W
        
        return None
    
    def try_winograd_search():
        """Try Winograd-like algorithm for small matrix multiplication."""
        if time.time() > deadline:
            return None
        
        # Winograd's algorithm for 3x3 * 3x3 uses 15 multiplications instead of 27
        # For our case, we can try to adapt this idea
        
        # The key idea is to use the fact that (a+b+c) can be computed with
        # fewer operations if we share intermediate results
        
        # For C[i,k] = sum_j A[i,j]*B[j,k], we can try to compute
        # partial sums and share them
        
        # Let's try a specific construction for m=3:
        # Define:
        # s0 = A[i,0] + A[i,1]
        # s1 = A[i,1] + A[i,2]  
        # s2 = A[i,0] + A[i,2]
        
        # Similarly for B columns:
        # t0k = B[0,k] + B[1,k]
        # t1k = B[1,k] + B[2,k]
        # t2k = B[0,k] + B[2,k]
        
        # Then: A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        # = (s0*s1 - s2*t2k)/2 + ... (need to verify)
        
        # Actually, this is getting complicated. Let me try a different approach.
        
        # Use the known result that for 2x3 * 3x7, we can potentially use
        # rank 39 or 40 by exploiting the small dimensions
        
        # Let's try to construct a decomposition with rank 41 first
        # by merging two schoolbook terms
        
        # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged if:
        # U[i1*m+j1, r] * V[j1*p+k1, r] * W[k1*n+i1, r] +
        # U[i2*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i2, r]
        # can be represented as a single rank-1 term
        
        # This is generally not possible, so let's try a different strategy
        
        # Strategy: Use the fact that for fixed j, we have n*p = 14 terms
        # and try to reduce them
        
        # For j=0: terms are (i,0,k) for i in {0,1}, k in {0,...,6}
        # These are: A[i,0]*B[0,k] for all i,k
        
        # We can write this as: for each k, A[:,0] * B[0,k]
        # This is an outer product, which has rank 1 in the (i,k) space for fixed k
        
        # But we need to sum over j, so it's more complex
        
        # Let me try a computational approach: use linear programming or
        # integer programming to find a minimal decomposition
        
        # For now, let's try random search with a bias toward structured solutions
        
        np.random.seed(999)
        
        # Try to find rank 41 by starting from schoolbook and trying to
        # replace two terms with one
        
        for target_rank in [41, 40]:
            if time.time() > deadline:
                break
            
            max_attempts = 300
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                # Start with schoolbook
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
                
                # Try to modify some entries to merge terms
                # Pick two random columns to potentially merge
                if target_rank < rank_schoolbook:
                    col1 = np.random.randint(0, target_rank)
                    col2 = np.random.randint(0, target_rank)
                    
                    # Try to set one column to be a combination of the other
                    # and adjust other columns to compensate
                    # This is heuristic and may not work
                    
                    # Instead, let's try a random sparse initialization
                    U = np.zeros((n * m, target_rank), dtype=np.float64)
                    V = np.zeros((m * p, target_rank), dtype=np.float64)
                    W = np.zeros((p * n, target_rank), dtype=np.float64)
                
                # Random sparse initialization
                for r_idx in range(target_rank):
                    # Each term should cover a small subset of (i,j), (j,k), (k,i)
                    num_u = np.random.randint(1, 3)
                    num_v = np.random.randint(1, 3)
                    num_w = np.random.randint(1, 3)
                    
                    u_pos = np.random.choice(n * m, size=num_u, replace=False)
                    v_pos = np.random.choice(m * p, size=num_v, replace=False)
                    w_pos = np.random.choice(p * n, size=num_w, replace=False)
                    
                    for pos in u_pos:
                        U[pos, r_idx] = float(np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0]))
                    for pos in v_pos:
                        V[pos, r_idx] = float(np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0]))
                    for pos in w_pos:
                        W[pos, r_idx] = float(np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0]))
                
                if verify_fast(U, V, W):
                    if verify_exact(U, V, W):
                        return U, V, W
        
        return None
    
    # Try different search strategies
    result = try_strassen_like_search()
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    result = try_winograd_search()
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Return schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
