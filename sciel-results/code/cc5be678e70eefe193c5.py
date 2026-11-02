"""Matrix multiplication 2x2x4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal sum_j A[i,j]*B[j,k]
    """
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = len(U[0])
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal delta (i.e., T[i,j,k] = 1 if we're computing C[i,k]=sum_j A[i,j]B[j,k])
    # Actually T[i,j,k] should be 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, no. The tensor for matrix mult is: T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if total != 1:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using numpy with float, then exact check on promising candidates."""
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should all equal 1.
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build tensor via einsum or manual
    # U reshaped: (n, m, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if all entries are approximately 1
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    
    # Exact verification with fractions
    return _verify_decomposition(U, V, W, n, m, p)

def _schoolbook_terms(n, m, p):
    """Return the 16 schoolbook rank-1 terms for n x m x p matrix mult.
    Term (i,j,k): U has 1 at position (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
    """
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u, v, w))
    return terms

def _try_rank13_shared_term(n, m, p, deadline):
    """Try to find rank-13 by sharing one term between the two Strassen blocks.
    
    Structure: 6 terms for block1 (cols 0,1), 6 terms for block2 (cols 2,3), 
    and 1 shared term contributing to both blocks.
    """
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    rank = 13
    
    # The shared term (index 12) will have W entries in both block1 and block2
    # W is indexed W[k*n+i, r], so:
    # Block1 W entries: k=0,1 -> indices 0,1,2,3
    # Block2 W entries: k=2,3 -> indices 4,5,6,7
    
    # Try various sparse patterns for the shared term's W vector
    # W_shared[12] should have entries in both {0,1,2,3} and {4,5,6,7}
    
    # Pattern: shared term contributes to c00 and c12 (or similar cross terms)
    # Let's try: W_shared has 1 at position (k=0,i=0)=0 and (k=2,i=1)=5
    # Or W_shared has 1 at (k=0,i=0)=0 and (k=3,i=0)=4
    
    # For each pattern, set up the linear system for U and V
    # Given W, the constraint for each (i,j,k) is:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is bilinear in U and V for fixed W.
    
    # Strategy: fix W, then for each (i,k), the constraint becomes:
    #   sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Group by r: sum_r W[k*n+i, r] * (sum_j U[i*m+j,r] * V[j*p+k,r]) = 1
    
    # Let's try a specific pattern and solve
    # Pattern A: shared W has entries at (k=0,i=0) and (k=2,i=1)
    # i.e., W_shared[0] = 1, W_shared[5] = 1 (since k=2,i=1 -> 2*2+1=5)
    
    patterns = [
        # (list of (k,i) positions for shared W, values)
        [(0,0,1), (2,1,1)],
        [(0,0,1), (3,0,1)],
        [(0,1,1), (2,0,1)],
        [(0,1,1), (3,1,1)],
        [(1,0,1), (2,1,1)],
        [(1,0,1), (3,0,1)],
        [(1,1,1), (2,0,1)],
        [(1,1,1), (3,1,1)],
    ]
    
    for pat in patterns:
        if time.time() > deadline:
            break
        
        W = [[0]*rank for _ in range(w_dim)]
        
        # Set up the shared term (column 12)
        for k, i, val in pat:
            W[k*n+i][12] = val
        
        # Now we need to fill U (4x13) and V (8x13) and W (8x13) for columns 0-11
        # such that all 16 constraints are satisfied.
        
        # For columns 0-5: handle block1 (k=0,1)
        # For columns 6-11: handle block2 (k=2,3)
        # Column 12: shared, already set in W
        
        # The 16 constraints:
        # (i=0,j=0,k=0): sum_r U[0,r]*V[0,r]*W[0,r] = 1
        # (i=0,j=1,k=0): sum_r U[1,r]*V[0,r]*W[0,r] = 1
        # (i=0,j=0,k=1): sum_r U[0,r]*V[1,r]*W[2,r] = 1
        # (i=0,j=1,k=1): sum_r U[1,r]*V[1,r]*W[2,r] = 1
        # (i=1,j=0,k=0): sum_r U[2,r]*V[0,r]*W[1,r] = 1
        # (i=1,j=1,k=0): sum_r U[3,r]*V[0,r]*W[1,r] = 1
        # (i=1,j=0,k=1): sum_r U[2,r]*V[1,r]*W[3,r] = 1
        # (i=1,j=1,k=1): sum_r U[3,r]*V[1,r]*W[3,r] = 1
        # (i=0,j=0,k=2): sum_r U[0,r]*V[2,r]*W[4,r] = 1
        # (i=0,j=1,k=2): sum_r U[1,r]*V[2,r]*W[4,r] = 1
        # (i=0,j=0,k=3): sum_r U[0,r]*V[3,r]*W[6,r] = 1
        # (i=0,j=1,k=3): sum_r U[1,r]*V[3,r]*W[6,r] = 1
        # (i=1,j=0,k=2): sum_r U[2,r]*V[2,r]*W[5,r] = 1
        # (i=1,j=1,k=2): sum_r U[3,r]*V[2,r]*W[5,r] = 1
        # (i=1,j=0,k=3): sum_r U[2,r]*V[3,r]*W[7,r] = 1
        # (i=1,j=1,k=3): sum_r U[3,r]*V[3,r]*W[7,r] = 1
        
        # For the shared term (r=12), W[0,12] and W[5,12] (or similar) are set.
        # The contribution of term 12 to each constraint:
        # (i=0,j=0,k=0): U[0,12]*V[0,12]*W[0,12]
        # (i=0,j=1,k=0): U[1,12]*V[0,12]*W[0,12]
        # (i=0,j=0,k=1): U[0,12]*V[1,12]*W[2,12]  (W[2,12]=0 typically)
        # (i=0,j=1,k=1): U[1,12]*V[1,12]*W[2,12]  (W[2,12]=0)
        # (i=1,j=0,k=0): U[2,12]*V[0,12]*W[1,12]  (W[1,12]=0)
        # (i=1,j=1,k=0): U[3,12]*V[0,12]*W[1,12]  (W[1,12]=0)
        # (i=1,j=0,k=1): U[2,12]*V[1,12]*W[3,12]  (W[3,12]=0)
        # (i=1,j=1,k=1): U[3,12]*V[1,12]*W[3,12]  (W[3,12]=0)
        # (i=0,j=0,k=2): U[0,12]*V[2,12]*W[4,12]  (W[4,12]=0)
        # (i=0,j=1,k=2): U[1,12]*V[2,12]*W[4,12]  (W[4,12]=0)
        # (i=0,j=0,k=3): U[0,12]*V[3,12]*W[6,12]  (W[6,12]=0)
        # (i=0,j=1,k=3): U[1,12]*V[3,12]*W[6,12]  (W[6,12]=0)
        # (i=1,j=0,k=2): U[2,12]*V[2,12]*W[5,12]
        # (i=1,j=1,k=2): U[3,12]*V[2,12]*W[5,12]
        # (i=1,j=0,k=3): U[2,12]*V[3,12]*W[7,12]  (W[7,12]=0)
        # (i=1,j=1,k=3): U[3,12]*V[3,12]*W[7,12]  (W[7,12]=0)
        
        # So the shared term only affects constraints where W[shared_pos, 12] != 0.
        # For pattern [(0,0,1),(2,1,1)]: W[0,12]=1, W[5,12]=1
        # Affected constraints: (0,0,0), (0,1,0), (1,0,2), (1,1,2)
        
        # The remaining 12 constraints must be satisfied by terms 0-11.
        # And the 4 affected constraints must have their shared-term contribution
        # compensated by terms 0-11.
        
        # This is a complex bilinear system. Let me try a different approach:
        # Use the known Strassen structure for 6 terms per block and see if the
        # shared term can absorb the "extra" contributions.
        
        # Actually, let me try a more direct approach: parameterize U and V 
        # with free variables and solve the resulting polynomial system.
        
        # For simplicity, let me try: terms 0-5 use Strassen-like structure for block1,
        # terms 6-11 for block2, and term 12 is the shared term.
        
        # Let me try to solve this as a system of 16 equations in the unknowns.
        # Unknowns: U has 4*13=52 entries, V has 8*13=104, W has 8*13=104.
        # But W is mostly determined by the pattern. Let me fix W more.
        
        # Fix W for terms 0-11 to be sparse (Strassen-like):
        # Term r in block1: W[k*n+i, r] nonzero only for k in {0,1}
        # Term r in block2: W[k*n+i, r] nonzero only for k in {2,3}
        
        # Let me use the Strassen W patterns for 6 terms each block.
        # Strassen 7 terms use all 4 W positions per block. With 6 terms,
        # we need to merge two Strassen terms.
        
        # This is getting complex. Let me try a numerical approach:
        # Fix W (all 13 columns), then solve for U and V.
        
        # Fix W using Strassen patterns for 6 terms per block + shared term
        W = [[0]*rank for _ in range(w_dim)]
        
        # Block1 W patterns (6 terms, k=0,1 -> W indices 0,1,2,3)
        # Strassen 7 terms for 2x2x2 have W:
        # T0: W[0]=1, W[3]=1  (c00, c11)
        # T1: W[1]=1, W[3]=-1 (c10, c11)
        # T2: W[2]=1, W[3]=1  (c01, c11)
        # T3: W[0]=1, W[1]=1  (c00, c10)
        # T4: W[2]=1, W[0]=-1 (c01, c00)
        # T5: W[3]=1           (c11)
        # T6: W[0]=1           (c00)
        # 
        # With 6 terms, merge T5 and T6: new T5' has W[3]=1, W[0]=1
        # But that changes the U and V structure too.
        
        # Let me just try to solve numerically with scipy.
        # Fix W to a specific pattern, then solve bilinear system for U,V.
        
        # W for block1 (terms 0-5), using 6-term "Strassen":
        # Merge terms to reduce from 7 to 6
        block1_W = [
            [1,0,0,1],  # T0: c00,c11
            [0,1,0,-1], # T1: c10,c11
            [0,0,1,1],  # T2: c01,c11
            [1,1,0,0],  # T3: c00,c10
            [-1,0,1,0], # T4: c00,c01
            [1,0,0,1],  # T5: c00,c11 (merged T5+T6)
        ]
        # Hmm, this doesn't work cleanly. Let me try a different approach.
        
        # Let me just do a random search with the shared term structure.
        pass
    
    return None

def _random_rank_r_search(n, m, p, rank, num_trials, deadline):
    """Try to find a rank-r decomposition by random linear combinations of schoolbook terms."""
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    schoolbook = _schoolbook_terms(n, m, p)
    num_schoolbook = len(schoolbook)
    
    best_rank = rank + 1  # We'll try to find rank-r
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Randomly select 'rank' terms and random integer coefficients
        # Actually, we need to form rank linear combinations of the schoolbook terms
        # such that the resulting rank-r decomposition is exact.
        
        # Strategy: pick 'rank' random subsets of schoolbook terms with random coefficients
        # and check if the resulting tensor equals the matrix mult tensor.
        
        # A rank-r term is: sum_{s=1}^{r} c_s * (u_s, v_s, w_s)
        # where each (u_s, v_s, w_s) is a linear combination of schoolbook terms.
        
        # Let's generate random linear combinations:
        U = [[0]*rank for _ in range(u_dim)]
        V = [[0]*rank for _ in range(v_dim)]
        W = [[0]*rank for _ in range(w_dim)]
        
        for r in range(rank):
            # Pick a random linear combination of schoolbook terms
            # Use small random integer coefficients
            coeffs = [random.randint(-2, 2) for _ in range(num_schoolbook)]
            # Ensure not all zero
            if all(c == 0 for c in coeffs):
                coeffs[0] = 1
            
            for s, (u_s, v_s, w_s) in enumerate(schoolbook):
                c = coeffs[s]
                if c != 0:
                    for idx in range(u_dim):
                        U[idx][r] += c * u_s[idx]
                    for idx in range(v_dim):
                        V[idx][r] += c * v_s[idx]
                    for idx in range(w_dim):
                        W[idx][r] += c * w_s[idx]
        
        # Verify
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_rank13_algebraic(n, m, p, deadline):
    """Try to find rank-13 by solving the bilinear system with a fixed W structure.
    
    Fix W to have a specific sparse pattern, then solve for U and V.
    The system is: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    """
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    rank = 13
    
    # Strategy: Fix W to be sparse (each column has 1-2 nonzero entries).
    # Then the system becomes: for each (i,j,k), a sum of products = 1.
    # If W is fixed, this is bilinear in U and V.
    # We can fix one of U or V and solve linearly for the other.
    
    # Let's fix U to be a specific matrix and solve for V.
    # Or fix V and solve for U.
    
    # Actually, let's try: fix W, fix U, solve linear system for V.
    # For fixed W and U, the constraint for (i,j,k) is:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Group by (j,k): for each j,k, we have:
    #   sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for each i
    # This gives n equations for the p*m unknowns V[j*p+k, r].
    # Total: m*p*n = 16 equations, m*p*rank = 8*13 = 104 unknowns.
    # Underdetermined! So we can fix W and U, and there are many V solutions.
    
    # But we also need V to produce the right structure. Let me think differently.
    
    # Better approach: fix W and V, solve for U. Same underdetermined system.
    
    # Let me try: fix W to a structured pattern, then use iterative method.
    
    # Fix W: each column r has W[k*n+i, r] nonzero for specific (k,i) pairs.
    # Use Strassen-like patterns.
    
    # For 13 terms, let's use:
    # Terms 0-5: block1 (k=0,1), Strassen-like 6-term pattern
    # Terms 6-11: block2 (k=2,3), Strassen-like 6-term pattern  
    # Term 12: shared, contributes to both blocks
    
    # W patterns for 6-term Strassen-like:
    # We need to cover all (k,i) pairs: (0,0),(0,1),(1,0),(1,1) for block1
    # and (2,0),(2,1),(3,0),(3,1) for block2
    
    # 6 terms to cover 4 (k,i) pairs: some terms cover 2 pairs.
    # Strassen uses 7 terms where each term covers 1-2 output positions.
    
    # Let me define W explicitly:
    W = [[0]*rank for _ in range(w_dim)]
    
    # Block1 (terms 0-5): W indices 0-3 correspond to (k,i) = (0,0),(0,1),(1,0),(1,1)
    # Strassen 7-term W patterns (adapted to 6 terms by merging):
    # Original 7: [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0], [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    # Merge last two: [1,0,0,1] -> but that's same as term 0. 
    # Let me just use 6 distinct patterns:
    block1_W = [
        [1, 0, 0, 1],   # c00, c11
        [0, 1, 0, -1],  # c10, c11
        [0, 0, 1, 1],   # c01, c11
        [1, 1, 0, 0],   # c00, c10
        [-1, 0, 1, 0],  # c00, c01
        [1, 0, 0, 1],   # c00, c11 (redundant but let's try)
    ]
    
    # Hmm, having duplicate W patterns means those terms are linearly dependent in W.
    # That's fine as long as U and V are different.
    
    # Block2 (terms 6-11): W indices 4-7 correspond to (k,i) = (2,0),(2,1),(3,0),(3,1)
    block2_W = [
        [1, 0, 0, 1],   # c20, c31
        [0, 1, 0, -1],  # c30, c31
        [0, 0, 1, 1],   # c21, c31
        [1, 1, 0, 0],   # c20, c30
        [-1, 0, 1, 0],  # c20, c21
        [1, 0, 0, 1],   # c20, c31
    ]
    
    # Shared term (12): contributes to both blocks
    # Let's say W[0,12]=1 (k=0,i=0) and W[5,12]=1 (k=2,i=1)
    W[0][12] = 1
    W[5][12] = 1
    
    for r in range(6):
        for ki in range(4):
            W[ki][r] = block1_W[r][ki]
    for r in range(6):
        for ki in range(4):
            W[4+ki][6+r] = block2_W[r][ki]
    
    # Now fix U to a Strassen-like pattern and solve for V.
    # U is 4x13. U[i*m+j, r] for i=0,1; j=0,1.
    # Strassen U patterns for 2x2:
    # T0: a00+a11 -> [1,0,0,1] (U[0,0]=1, U[1,1]=1)
    # T1: a10+a11 -> [0,0,1,1] (U[2,0]=1, U[3,1]=1) wait, U[i*m+j]
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # T0: U[0]=1, U[3]=1
    # T1: U[2]=1, U[3]=1
    # T2: U[0]=1
    # T3: U[3]=1
    # T4: U[0]=1, U[1]=1
    # T5: U[2]=1, U[0]=-1
    # T6: U[1]=1, U[3]=-1
    
    U = [[0]*rank for _ in range(u_dim)]
    
    # Block1 U (terms 0-5):
    block1_U = [
        [1, 0, 0, 1],   # a00+a11
        [0, 0, 1, 1],   # a10+a11
        [1, 0, 0, 0],   # a00
        [0, 0, 0, 1],   # a11
        [1, 1, 0, 0],   # a00+a01
        [1, 0, 1, 0],   # a00+a10 (modified)
    ]
    
    # Block2 U (terms 6-11): same A factors
    block2_U = block1_U
    
    # Shared term U (12):
    shared_U = [1, 0, 0, 1]  # a00+a11
    
    for r in range(6):
        for idx in range(4):
            U[idx][r] = block1_U[r][idx]
    for r in range(6):
        for idx in range(4):
            U[idx][6+r] = block2_U[r][idx]
    for idx in range(4):
        U[idx][12] = shared_U[idx]
    
    # Now solve for V (8x13) from the 16 constraints.
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # Unknowns: V[j*p+k, r] for j=0,1; k=0,1,2,3; r=0,...,12 -> 8*13=104 unknowns
    # Equations: 16
    # Underdetermined. Let's fix most of V and solve for the free variables.
    
    # Actually, let's set up the linear system properly.
    # For each (i,j,k): sum_r (U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r] = 1
    # Let A_{(i,j,k), (j,k,r)} = U[i*m+j, r] * W[k*n+i, r]
    # Then A * V_flat = 1 (16x104 system)
    
    # Build the 16x104 matrix
    import numpy as np
    
    # Index mapping:
    # Equation (i,j,k) -> eq_idx = i*m*p + j*p + k = i*8 + j*4 + k
    # Unknown (j,k,r) -> var_idx = j*p*rank + k*rank + r = j*52 + k*13 + r
    
    A_mat = np.zeros((16, 104), dtype=float)
    rhs = np.ones(16, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for r in range(rank):
                    var_idx = j * p * rank + k * rank + r
                    A_mat[eq_idx, var_idx] = U[i*m+j][r] * W[k*n+i][r]
    
    # Solve the underdetermined system: find V_flat such that A_mat @ V_flat = rhs
    # Use least squares (minimum norm solution)
    try:
        V_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
    except:
        return None
    
    # Check if the solution is exact
    if not np.allclose(A_mat @ V_flat, rhs, atol=1e-10):
        return None
    
    # Reshape V_flat to V (8x13)
    V = [[0]*rank for _ in range(v_dim)]
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                V[j*p+k][r] = V_flat[j * p * rank + k * rank + r]
    
    # Check if entries are close to integers or half-integers
    # If not, this particular W,U pattern doesn't yield a clean solution.
    # But we can still use it if it's exact.
    
    # Verify the full decomposition
    if _verify_decomposition_fast(U, V, W, n, m, p):
        # Round to nearest integer or half-integer if close
        U_rounded = [[round(x*2)/2 for x in row] for row in U]
        V_rounded = [[round(x*2)/2 for x in row] for row in V]
        W_rounded = [[round(x*2)/2 for x in row] for row in W]
        if _verify_decomposition_fast(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded
        else:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 4
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Known rank-14 decomposition (Strassen-based)
    rank14 = _strassen_rank14(n, m, p)
    
    # Try to find rank-13 using algebraic approach
    result_13 = _try_rank13_algebraic(n, m, p, deadline)
    
    if result_13 is not None:
        U, V, W = result_13
        return {"U": U, "V": V, "W": W}
    
    # Try randomized search for rank-13
    random.seed(42)
    result_13_rand = _random_rank_r_search(n, m, p, 13, num_trials=20000, deadline=deadline)
    
    if result_13_rand is not None:
        U, V, W = result_13_rand
        return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-14
    U, V, W = rank14
    return {"U": U, "V": V, "W": W}

def _strassen_rank14(n, m, p):
    """Construct the known rank-14 Strassen-based decomposition for 2x2x4."""
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    rank = 14
    
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen 7-term decomposition for 2x2x2:
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    # M2 = (a10+a11)b00       -> contributes to c10, c11
    # M3 = a00(b01-b11)       -> contributes to c01, c11
    # M4 = a11(b10-b00)       -> contributes to c00, c10
    # M5 = (a00+a01)b11       -> contributes to c01, c00
    # M6 = (a10-a00)(b00+b01) -> contributes to c11
    # M7 = (a01-a11)(b10+b11) -> contributes to c00
    
    # For 2x2x4, output C has 2 rows (i=0,1) and 4 columns (k=0,1,2,3)
    # W is indexed as W[k*n + i, r]
    # U is indexed as U[i*m + j, r]
    # V is indexed as V[j*p + k, r]
    
    # Block 1: columns k=0,1 of B
    terms_block1 = [
        # (u_entries: list of (i, j, val), v_entries: list of (j, k, val), w_entries: list of (k, i, val))
        # M1: (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2: (a10+a11)b00 -> c10 += M2, c11 -= M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        # M3: a00(b01-b11) -> c01 += M3, c11 += M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        # M4: a11(b10-b00) -> c00 += M4, c10 += M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # M5: (a00+a01)b11 -> c01 += M5, c00 -= M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1),(0,0,-1)]),
        # M6: (a10-a00)(b00+b01) -> c11 += M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7: (a01-a11)(b10+b11) -> c00 += M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Block 2: columns k=2,3 of B
    terms_block2 = [
        # M1: (a00+a11)(b02+b13) -> c02 += M1, c13 += M1
        ([(0,0,1),(1,1,1)], [(0,2,1),(1,3,1)], [(2,0,1),(3,1,1)]),
        # M2: (a10+a11)b02 -> c12 += M2, c13 -= M2
        ([(1,0,1),(1,1,1)], [(0,2,1)], [(2,1,1),(3,1,-1)]),
        # M3: a00(b03-b13) -> c03 += M3, c13 += M3
        ([(0,0,1)], [(0,3,1),(1,3,-1)], [(3,0,1),(3,1,1)]),
        # M4: a11(b12-b02) -> c02 += M4, c12 += M4
        ([(1,1,1)], [(1,2,1),(0,2,-1)], [(2,0,1),(2,1,1)]),
        # M5: (a00+a01)b13 -> c03 += M5, c02 -= M5
        ([(0,0,1),(0,1,1)], [(1,3,1)], [(3,0,1),(2,0,-1)]),
        # M6: (a10-a00)(b02+b03) -> c13 += M6
        ([(1,0,1),(0,0,-1)], [(0,2,1),(0,3,1)], [(3,1,1)]),
        # M7: (a01-a11)(b12+b13) -> c02 += M7
        ([(0,1,1),(1,1,-1)], [(1,2,1),(1,3,1)], [(2,0,1)]),
    ]
    
    for r in range(7):
        u_ent, v_ent, w_ent = terms_block1[r]
        col = r
        for i, j, val in u_ent:
            U[i*m+j][col] += val
        for j, k, val in v_ent:
            V[j*p+k][col] += val
        for k, i, val in w_ent:
            W[k*n+i][col] += val
    
    for r in range(7):
        u_ent, v_ent, w_ent = terms_block2[r]
        col = 7 + r
        for i, j, val in u_ent:
            U[i*m+j][col] += val
        for j, k, val in v_ent:
            V[j*p+k][col] += val
        for k, i, val in w_ent:
            W[k*n+i][col] += val
    
    return U, V, W
# EVOLVE-BLOCK-END
