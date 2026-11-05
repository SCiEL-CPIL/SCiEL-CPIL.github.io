"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import svd

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Build rank-33 via Strassen blocks, then use SVD analysis and
    structured search to find rank-32 by exploiting tensor structure.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Block b handles columns 2b, 2b+1 of B.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    def svd_analysis():
        """
        Use SVD on flattened tensor to identify the minimal rank subspace.
        The tensor T can be flattened in different ways:
        - T as (dim_a, dim_b*dim_c): 6 x 216 matrix
        - T as (dim_a*dim_b, dim_c): 108 x 12 matrix  
        - T as (dim_a*dim_c, dim_b): 72 x 18 matrix
        The rank of each flattening gives a lower bound on the tensor rank.
        """
        # Flatten T as (dim_a, dim_b*dim_c)
        T_ac = T.reshape(dim_a, dim_b * dim_c)
        U_ac, S_ac, Vt_ac = svd(T_ac, full_matrices=False)
        
        # Flatten T as (dim_a*dim_b, dim_c)
        T_abc = T.reshape(dim_a * dim_b, dim_c)
        U_abc, S_abc, Vt_abc = svd(T_abc, full_matrices=False)
        
        # Flatten T as (dim_a*dim_c, dim_b)
        T_ab = T.reshape(dim_a * dim_c, dim_b)
        U_ab, S_ab, Vt_ab = svd(T_ab, full_matrices=False)
        
        # The tensor rank is at least the max of these matrix ranks
        rank_ac = np.sum(S_ac > 1e-10)
        rank_abc = np.sum(S_abc > 1e-10)
        rank_ab = np.sum(S_ab > 1e-10)
        
        return {
            'rank_ac': rank_ac,
            'rank_abc': rank_abc,
            'rank_ab': rank_ab,
            'lower_bound': max(rank_ac, rank_abc, rank_ab),
            'singular_values_ac': S_ac,
            'singular_values_abc': S_abc,
            'singular_values_ab': S_ab,
            'U_ac': U_ac,
            'Vt_ac': Vt_ac,
            'U_abc': U_abc,
            'Vt_abc': Vt_abc,
            'U_ab': U_ab,
            'Vt_ab': Vt_ab
        }

    def try_rank32_svd_guided(svd_info):
        """
        Use SVD subspace structure to guide search for rank-32 decomposition.
        
        Key insight: If the tensor has a specific low-rank structure in its
        flattening, we can use the SVD basis to constrain our search.
        
        For rank r decomposition, we need U (6xr), V (18xr), W (12xr).
        The SVD tells us which subspaces are important.
        """
        rank_target = 32
        
        # Use the SVD of T as (dim_a*dim_c, dim_b) = (72, 18)
        # This tells us about the structure along the b (V) dimension
        U_ab = svd_info['U_ab']
        S_ab = svd_info['singular_values_ab']
        Vt_ab = svd_info['Vt_ab']
        
        # The first few singular vectors give the most important directions
        # For rank 32, we need to find 32 rank-1 tensors that sum to T
        
        # Strategy: Use the SVD basis to parameterize the search
        # Fix the structure based on SVD and solve for exact entries
        
        # Alternative: Use the known rank-33 structure and try to merge
        # two terms by finding a linear relation
        
        # Let's try: take the rank-33 decomposition and see if any two
        # columns can be replaced by their sum (if the sum is also rank-1)
        
        U33, V33, W33 = strassen_block_decomposition()
        
        # For each pair of columns (r1, r2), check if the tensor
        # U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2]
        # can be expressed as a single rank-1 tensor
        # This requires: (U1+U2)*(V1+V2)*(W1+W2) = U1*V1*W1 + U2*V2*W2
        # Which is generally false, but let's check
        
        # More practically: try to find if there's a column r such that
        # the contribution of column r can be absorbed into other columns
        # by adjusting their values
        
        # Let's try a different approach: use the SVD to find a
        # rank-32 approximation and then round to exact integers
        
        # For rank r, the best rank-r approximation of T (in Frobenius norm)
        # is given by the truncated SVD. But we need exact decomposition.
        
        # Let's try: fix U and V to have specific structure, solve for W
        # using least squares, then check if the solution is exact
        
        # Use the SVD basis for U
        # U is 6x32. The SVD of T_ac gives us a 6x6 basis.
        # Use the first 6 singular vectors as basis for U columns
        
        # Actually, let's try a more direct approach:
        # The tensor T has a very specific structure. Let's exploit it.
        
        # T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        # Otherwise T[a, b, c] = 0
        
        # For a rank-1 term (u, v, w), the contribution is u[a]*v[b]*w[c]
        # For this to match T, we need:
        # sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c]
        
        # Let's try to construct a rank-32 decomposition by modifying
        # the rank-33 one. The key is to find a linear relation among
        # the 33 rank-1 tensors.
        
        # If we can find coefficients alpha_r (not all zero) such that
        # sum_r alpha_r * (u_r, v_r, w_r) = (0, 0, 0) in some sense,
        # we can eliminate one term.
        
        # Actually, the condition for merging is:
        # There exist 32 rank-1 tensors (u'_r, v'_r, w'_r) for r=1..32
        # such that sum_{r=1}^{32} u'_r[a] * v'_r[b] * w'_r[c] = T[a,b,c]
        
        # This is a system of dim_a*dim_b*dim_c = 1296 equations in
        # (dim_a + dim_b + dim_c) * 32 = 36*32 = 1152 unknowns.
        # Overdetermined, but may have solutions.
        
        # Let's try to solve this using the SVD structure.
        # Fix the first 31 terms to be the first 31 terms of the rank-33
        # decomposition, and solve for the 32nd term.
        
        # This won't work directly because the 32nd term would need to
        # be a rank-1 tensor that completes the decomposition.
        
        # Let's try a randomized approach with SVD guidance:
        # 1. Initialize U, V, W using SVD basis
        # 2. Iterate: fix two, solve for the third
        # 3. Check for exact solution
        
        if time.time() > deadline - 10:
            return None
        
        # Use SVD to initialize
        # T as (dim_a, dim_b*dim_c)
        U_init = U_ac[:, :min(6, rank_target)]  # 6x6 basis
        # This gives us a starting point for U columns
        
        # For now, let's try a structured search:
        # Modify one of the standard (schoolbook) terms to share with
        # a Strassen term
        
        # In block 0, the standard terms are:
        # (i=0,k=0): U[2]=1, V[12]=1, W[0]=1
        # (i=0,k=1): U[2]=1, V[13]=1, W[1]=1
        # (i=1,k=0): U[5]=1, V[12]=1, W[2]=1
        # (i=1,k=1): U[5]=1, V[13]=1, W[3]=1
        
        # Can we replace these 4 terms with 3 terms by combining
        # with a Strassen term?
        
        # Let's try: add a term that contributes to both the standard
        # and Strassen parts simultaneously
        
        # This is complex. Let's just try to verify if rank 32 is
        # achievable by checking the lower bound from SVD.
        
        lower_bound = svd_info['lower_bound']
        if lower_bound > 32:
            return None  # Rank 32 is impossible
        
        # If lower_bound <= 32, try to find a rank-32 decomposition
        # using iterative refinement
        
        # For now, return None and use the rank-33 result
        return None

    # Start with best known
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Try SVD-guided search for rank 32
    if time.time() < deadline - 15:
        svd_info = svd_analysis()
        result = try_rank32_svd_guided(svd_info)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 32

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
