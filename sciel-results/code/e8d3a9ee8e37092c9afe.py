import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 135  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i,j,k] = 1 if (i,m_idx) and (j,p_idx) and (k,n_idx) align
    # Using trace convention: W[k*n+i, r]
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Sum over r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k) for matching j
    
    # Actually, for matrix multiplication C = A * B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j; j',k; i',k'] = delta(i,i') * delta(j,j') * delta(k,k')
    # But in our indexing: U[i*m+j, r], V[j'*p+k, r], W[k'*n+i', r]
    # We need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta(i,i')*delta(j,j')*delta(k,k')
    
    # So T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if i==i' and j==j' and k==k', else 0
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # --- Dual polynomial decomposition: rank 24 ---
    # Interpret the 4-dimensional inner product as coefficient-wise multiplication
    # of two linear forms in variable x:
    #   a(x) = sum_{j=0}^{3} A[i,j] * x^j
    #   b(x) = sum_{j=0}^{3} B[j,k] * x^j
    # The product c(x) = a(x)*b(x) is a polynomial of degree up to 6.
    # However, since we only need C[i,k] = sum_j A[i,j]*B[j,k] (the "trace" of the
    # coefficient-wise product), we can use the fact that:
    #   sum_j A[i,j]*B[j,k] = (1/(2*pi*i)) * integral_0^{2*pi} a(e^{it}) * b(e^{-it}) dt
    # For exact half-integer decomposition, we use evaluation at roots of unity.
    #
    # Alternative approach: Use the structure that the 4x4 matrix multiplication
    # tensor has rank 28 (Strassen-like), and we can exploit the 3-row / 8-col
    # structure.
    #
    # Let's try a direct construction:
    # For the 4x4 part, use a rank-28 decomposition (known minimal for 4x4 is 28).
    # But we need to handle the 3 rows and 8 columns.
    #
    # Actually, let's try a simpler structured approach:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # We can view this as: for each i, compute the dot product of row i of A with
    # each column of B.
    #
    # Using the "dual polynomial" idea:
    # Let omega = e^{2*pi*i/8} (8th root of unity)
    # Define for each i: f_i(x) = sum_j A[i,j] * x^j (polynomial of degree 3)
    # Define for each k: g_k(x) = sum_j B[j,k] * x^j (polynomial of degree 3)
    # Then C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # We can use the fact that for polynomials of degree <= 3:
    # sum_j a_j * b_j = (1/8) * sum_{t=0}^{7} f_i(omega^t) * g_k(omega^{-t})
    # This is because the Fourier basis diagonalizes the coefficient-wise product.
    #
    # So: C[i,k] = (1/8) * sum_{t=0}^{7} [sum_j A[i,j]*omega^{t*j}] * [sum_j B[j,k]*omega^{-t*j}]
    #
    # This gives a rank 3*8 = 24 decomposition if the values are half-integers.
    # But omega^t are complex and not half-integers!
    #
    # Let's use a different approach: exploit that we only need exact half-integer
    # entries. Let's try to find a rank-24 decomposition directly.
    
    def try_dual_polynomial():
        """
        Construct rank-24 decomposition using polynomial evaluation structure.
        
        For the 4x4 inner product, we use the identity:
        sum_{j=0}^{3} a_j * b_j = (1/4) * sum_{t=0}^{3} (sum_j a_j * omega^{t*j}) * (sum_j b_j * omega^{-t*j})
        where omega = e^{2*pi*i/4} = i (imaginary unit).
        
        But i is not a half-integer. So this doesn't directly work.
        
        Instead, let's use real-valued evaluations. We can use the fact that:
        sum_{j=0}^{3} a_j * b_j can be computed using 4 evaluations at specific points
        if we choose the right basis.
        
        Actually, let's just try to find a rank-24 decomposition by solving the
        linear system directly with half-integer constraints.
        """
        target_rank = 24
        nm, mp, pn = n * m, m * p, p * n
        
        # Try to construct using Kronecker structure
        # The 3x4x8 tensor can be seen as 3 copies of 1x4x8 with shared B
        # Or as 8 copies of 3x4x1 with shared A
        
        # Let's try: rank = 3 * 8 = 24
        # For each (i, k) pair, we need sum_j A[i,j]*B[j,k]
        # This is a 3x8 matrix of dot products, each of dimension 4.
        
        # Use the fact that a 4-dimensional dot product can be computed with
        # rank 4 (schoolbook) or potentially less.
        # But we need a SINGLE decomposition that works for all 3*8=24 pairs.
        
        # Let's try: U has rows indexed by (i,j), V by (j,k), W by (k,i)
        # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,i')*delta(j,j')*delta(k,k')
        
        # For rank 24, let r index (a, b) where a in 0..2, b in 0..3
        # U[i*m+j, (a,b)] = delta(i,a) * u[j,b]
        # V[j'*p+k, (a,b)] = v[j',b] * delta(k, something)
        # This is getting complicated. Let's just use the random search with
        # a better initialization.
        
        return None
    
    result = try_dual_polynomial()
    if result is not None:
        U, V, W, rank = result
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    def to_half_integer(x):
        """Round to nearest half-integer"""
        return np.round(x * 2) / 2.0
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact"""
        # Compute sum_r U[:,r] * V[:,r] * W[:,r] and compare to T
        # Use einsum
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(result, T, atol=tol)
    
    def search_random(rank, max_iters=500):
        """Try random initialization with half-integer constraints"""
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            # Snap to half-integers
            U = to_half_integer(U)
            V = to_half_integer(V)
            W = to_half_integer(W)
            if verify(U, V, W):
                return U, V, W, rank
        return None
    
    def local_refine(U, V, W, rank, max_steps=200):
        """Try to improve by local perturbations"""
        U, V, W = U.copy(), V.copy(), W.copy()
        current_ok = verify(U, V, W)
        for step in range(max_steps):
            if time.time() > DEADLINE:
                break
            # Try flipping a random entry
            mat = random.choice([U, V, W])
            r, c = random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1)
            old_val = mat[r, c]
            # Try adding or subtracting 0.5
            delta = random.choice([0.5, -0.5])
            mat[r, c] = old_val + delta
            if verify(mat if mat is U else U, mat if mat is V else V, mat if mat is W else W):
                # Keep the change
                continue
            else:
                mat[r, c] = old_val
        return U, V, W, rank
    
    # Try to find lower rank decompositions
    # First try structured rank-24 approach, then random search
    # Start with a structured initialization for rank 24
    
    def structured_rank24():
        """
        Try to construct a rank-24 decomposition using the polynomial structure.
        
        Key insight: C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        We can write this as a sum of 4 rank-1 terms for each (i,k) pair:
        C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        But we need a global decomposition. Let's use:
        r = (j, t) where j in 0..3, t in 0..5 (6 values to cover 3*8=24)
        
        Actually, let's try: r = (j, s) where j in 0..3, s in 0..5
        U[i*m+j, (j',s)] = delta(j,j') * u_i_s  (depends on i and s)
        V[j'*p+k, (j',s)] = v_j'_k_s  (depends on j', k, s)
        W[k'*n+i', (j',s)] = w_k'_i'_s  (depends on k', i', s)
        
        Sum over j',s: delta(j,j') * u_i_s * v_j_k_s * w_k_i_s
        = sum_s u_i_s * v_j_k_s * w_k_i_s
        
        We need this to equal delta(i,i')*delta(j,j')*delta(k,k')
        
        This is complex. Let's just do a focused random search for rank 24.
        """
        target_rank = 24
        nm, mp, pn = n * m, m * p, p * n
        
        # Initialize with a structured pattern
        for attempt in range(50):
            if time.time() > DEADLINE:
                break
            
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Structured initialization: use the schoolbook pattern but with
            # 24 columns instead of 96
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Assign to a column based on (i,j,k) mod 24
                        col = (i * m + j) * p + k  # 0..95
                        col = col % target_rank
                        U[i*m+j, col] += 1.0
                        V[j*p+k, col] += 1.0
                        W[k*n+i, col] += 1.0
            
            # Snap to half-integers
            U = to_half_integer(U)
            V = to_half_integer(V)
            W = to_half_integer(W)
            
            if verify(U, V, W):
                return U, V, W, target_rank
            
            # Try local refinement
            U, V, W, _ = local_refine(U, V, W, target_rank, max_steps=50)
            if verify(U, V, W):
                return U, V, W, target_rank
        
        return None
    
    result = structured_rank24()
    if result is not None:
        U, V, W, rank = result
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try random search for ranks below schoolbook
    if best_rank > 24:
        for target_rank in range(min(best_rank - 1, 30), 23, -1):
            if time.time() > DEADLINE:
                break
            # Try random search for this rank
            result = search_random(target_rank, max_iters=50)
            if result:
                U, V, W, rank = result
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break
    
    # Convert to list format
    U_out = [[U_out_val for U_out_val in row] for row in best_U.tolist()]
    V_out = [[V_out_val for V_out_val in row] for row in best_V.tolist()]
    W_out = [[W_out_val for W_out_val in row] for row in best_W.tolist()]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
