"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from fractions import Fraction

def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    produce the matrix multiplication tensor under the trace convention W[k*n+i, r]."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                expected = 1
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - expected) > tol:
                    return False, (A, B, C, expected, actual)
    return True, None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_rank11_decomposition():
    """
    Construct a rank-11 decomposition for M_{2,3,3}.
    
    Strategy: Use block structure. Split the 3 columns of A into 1+2,
    and the 3 rows of B into 1+2. This gives a 2x2 block structure
    where we can apply Strassen-like tricks to the 2x2 blocks.
    
    A = [a0 | a1 a2]  (2x3, split into 2x1 and 2x2)
    B = [b0; b1 b2]   (3x3, split into 1x3 and 2x3)
    
    C = a0*b0 + [a1 a2]*[b1; b2]
    
    The second term is a 2x2 times 2x3 multiplication.
    We can use Strassen for the 2x2 part embedded in this.
    
    Actually, let me try a direct construction using known results.
    For M_{2,3,3}, I'll use the following approach:
    
    Label A entries: A[i,j] for i=0,1 and j=0,1,2
    Label B entries: B[j,k] for j=0,1,2 and k=0,1,2
    Output C[k,i] = sum_j A[i,j]*B[j,k]
    
    I'll try to find 11 bilinear forms that compute all 18 outputs.
    """
    # Let me try a systematic construction.
    # The 18 products needed are A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # These contribute to C[k,i].
    
    # Group by j:
    # j=0: A[i,0]*B[0,k] for i=0,1; k=0,1,2 -> 6 products, contribute to C[k,i]
    # j=1: A[i,1]*B[1,k] for i=0,1; k=0,1,2 -> 6 products
    # j=2: A[i,2]*B[2,k] for i=0,1; k=0,1,2 -> 6 products
    
    # Total 18 products, each contributing to one C[k,i].
    
    # Idea: For each j, the 6 products form a 2x3 outer product structure.
    # A[:,j] (2x1) outer B[j,:] (1x3) = 2x3 matrix of products.
    
    # Can we combine across j values?
    
    # Let me try: define intermediate sums
    # S0 = A[:,0] (2x1 column)
    # S1 = A[:,1]
    # S2 = A[:,2]
    
    # T0 = B[0,:] (1x3 row)
    # T1 = B[1,:]
    # T2 = B[2,:]
    
    # C[k,i] = S0[i]*T0[k] + S1[i]*T1[k] + S2[i]*T2[k]
    
    # This is: C = S0*T0 + S1*T1 + S2*T2 (sum of 3 rank-1 matrices)
    
    # Each S_j*T_k is a 2x3 matrix computed from 6 multiplications.
    # Total: 18 multiplications.
    
    # To reduce: use the fact that S0, S1, S2 are 2-vectors and T0, T1, T2 are 3-vectors.
    # We can form linear combinations.
    
    # Let me try: 
    # m1 = (S0+S1+S2) * (T0+T1+T2)  -- but this gives sum of all 9 cross terms
    # That's not directly useful.
    
    # Alternative: use Strassen on a sub-problem.
    # Consider the 2x2 sub-matrix from columns 1,2 of A and rows 1,2 of B:
    # A_sub = [A[0,1] A[0,2]; A[1,1] A[1,2]]  (2x2)
    # B_sub = [B[1,1] B[1,2] B[1,3]; B[2,1] B[2,2] B[2,3]]  (2x3)
    
    # Hmm, this is 2x2 times 2x3, which has rank 12 in schoolbook.
    
    # Let me try yet another approach: direct search with structured initialization.
    
    # I'll set up the problem as: find U (6x11), V (9x11), W (6x11)
    # such that for all (i,j,k): sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*2+i, t] = 1
    # and for all other (A,B,C) triples: sum = 0.
    
    # This is 18 equations (for the support) plus many zero equations.
    
    # Let me try a numerical approach: fix some entries and solve.
    
    # Actually, let me try a known decomposition from the literature.
    # Based on work by Bläser and others, M_{2,3,3} can be done in rank 11.
    
    # Here's a construction:
    # Use the following 11 bilinear forms:
    
    # Let a_i = A[i,0], b_i = A[i,1], c_i = A[i,2] for i=0,1
    # Let x_k = B[0,k], y_k = B[1,k], z_k = B[2,k] for k=0,1,2
    
    # C[k,0] = a_0*x_k + b_0*y_k + c_0*z_k
    # C[k,1] = a_1*x_k + b_1*y_k + c_1*z_k
    
    # Define:
    # m1 = a_0*x_0
    # m2 = a_0*x_1  
    # m3 = a_0*x_2
    # m4 = a_1*x_0
    # m5 = a_1*x_1
    # m6 = a_1*x_2
    # m7 = (a_0+a_1)*(x_0+x_1+x_2)  -- this gives sum of a_i*x_k for all i,k
    # ... this is getting complicated.
    
    # Let me try a different strategy: use the fact that we can add and subtract.
    # For the 2x2 case (M_{2,2,2}), Strassen uses 7 instead of 8.
    # The key is: (a00+a11)(b00+b11) - a00*b00 - a11*b11 = a00*b11 + a11*b00
    
    # For our case, let me try to apply similar tricks.
    
    # Group the 18 products into pairs that can be combined:
    # Consider products involving A[0,0]*B[0,k] and A[1,0]*B[0,k] for k=0,1,2
    # These are: a00*x0, a00*x1, a00*x2, a10*x0, a10*x1, a10*x2
    
    # We can compute (a00+a10)*(x0+x1+x2) = a00*x0+a00*x1+a00*x2+a10*x0+a10*x1+a10*x2
    # That's 1 multiplication for 6 products! But we need them separately.
    
    # However, if we also have individual x_k available, we can extract:
    # a00*xk = ? Not directly from the sum.
    
    # OK, let me try a completely different approach: use a SAT/SMT-like search
    # with small integer coefficients.
    
    # For rank 11, try all combinations of small integers (-2 to 2) for a structured subset.
    
    # Actually, let me just try to find ANY rank < 18 decomposition via random search
    # with better initialization.
    
    # Strategy: start from schoolbook rank 18, then try to merge pairs of terms
    # using the identity: if we have terms u1*v1*w1 and u2*v2*w2, can we replace
    # them with fewer terms?
    
    # This is equivalent to checking if two rank-1 tensors sum to something of rank 1,
    # which only happens if they're proportional.
    
    # So we need to find linear dependencies among the 18 rank-1 components.
    
    # The 18 components are: for each (i,j,k), the tensor e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # (using appropriate index conventions).
    
    # These 18 tensors are linearly independent (they have disjoint supports in the
    # (A,B,C) space), so no linear combination gives zero. Hence rank >= 18?
    
    # Wait, that's not right. Tensor rank is not the same as the number of non-zero
    # entries. The rank-1 tensors in a decomposition don't need to have disjoint supports.
    
    # A rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors. The support of such
    # a tensor is the full Cartesian product of supports of u, v, w.
    
    # So a rank-1 tensor u⊗v⊗w has component T[A,B,C] = u[A]*v[B]*w[C] for ALL A,B,C,
    # not just where the original tensor is non-zero.
    
    # The constraint is: sum_t u_t[A]*v_t[B]*w_t[C] = T[A,B,C] for ALL A,B,C.
    
    # Where T[A,B,C] = 1 if (A,B,C) is in the support, 0 otherwise.
    
    # So we have 6*9*6 = 324 equations, of which 18 are =1 and 306 are =0.
    
    # This is a system of 324 polynomial equations in 11*(6+9+6) = 231 unknowns (for rank 11).
    
    # Let me try to solve this numerically.
    
    # I'll use a simple iterative approach: initialize randomly, then do gradient descent
    # on the sum of squared residuals.
    
    start_time = time.time()
    n, m, p = 2, 3, 3
    dim_A = n * m  # 6
    dim_B = m * p  # 9
    dim_C = p * n  # 6
    
    # Build the target tensor
    T = np.zeros((dim_A, dim_B, dim_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                T[A, B, C] = 1.0
    
    # Try rank 11
    for rank in [11, 12, 13]:
        if time.time() - start_time > 60:
            break
            
        best_loss = float('inf')
        best_params = None
        
        for attempt in range(50):
            if time.time() - start_time > 60:
                break
                
            rng = np.random.RandomState(attempt * 1000 + rank)
            U = rng.randn(dim_A, rank)
            V = rng.randn(dim_B, rank)
            W = rng.randn(dim_C, rank)
            
            # Scale to reasonable magnitude
            U /= np.sqrt(rank)
            V /= np.sqrt(rank)
            W /= np.sqrt(rank)
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() - start_time > 60:
                    break
                    
                # Compute loss: sum of (sum_t U[A,t]*V[B,t]*W[C,t] - T[A,B,C])^2
                # This is expensive to compute directly. Use vectorized form.
                
                # For each t, the contribution is outer(U[:,t], outer(V[:,t], W[:,t]))
                # Sum over t, then compare to T.
                
                # Efficient computation:
                # L = sum_t U_t ⊗ V_t ⊗ W_t - T
                # loss = ||L||^2
                
                # Gradient w.r.t. U[A,t] = sum_{B,C} L[A,B,C] * V[B,t] * W[C,t]
                # Similarly for V and W.
                
                # Compute L[A,B,C] = sum_t U[A,t]*V[B,t]*W[C,t]
                # = sum_t U[A,t] * (V[:,t] * W[:,t])[B] ... no, that's not right.
                # V[B,t]*W[C,t] is elementwise, so for fixed t: V_t * W_t is not defined
                # as they have different sizes. We need the outer product.
                
                # L[A,B,C] = sum_t U[A,t] * V[B,t] * W[C,t]
                # This can be computed as: for each t, add U[:,t] * V[:,t] * W[:,t] (broadcast)
                
                # Vectorized: 
                # L = np.einsum('at,bt,ct->abc', U, V, W)
                # But this creates a 6x9x6 tensor, which is 324 elements. Fine.
                
                L = np.einsum('at,bt,ct->abc', U, V, W)
                residual = L - T
                loss = np.sum(residual**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-10:
                    break
                
                # Compute gradients
                # dL/dU[A,t] = sum_{B,C} residual[A,B,C] * V[B,t] * W[C,t]
                # = sum_{B,C} residual[A,B,C] * V[B,t] * W[C,t]
                # For fixed t: sum_{B,C} residual[:,B,C] * V[B,t] * W[C,t]
                # = einsum('abc,bt,ct->at', residual, V, W)
                
                grad_U = np.einsum('abc,bt,ct->at', residual, V, W)
                grad_V = np.einsum('abc,at,ct->bt', residual, U, W)
                grad_W = np.einsum('abc,at,bt->ct', residual, U, V)
                
                # Adam-like update (simple version)
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Occasionally rescale to prevent drift
                if step % 100 == 0 and step > 0:
                    # Normalize each column
                    for t in range(rank):
                        nu = np.linalg.norm(U[:,t])
                        nv = np.linalg.norm(V[:,t])
                        nw = np.linalg.norm(W[:,t])
                        if nu > 0 and nv > 0 and nw > 0:
                            # Keep product of norms constant
                            target = (nu * nv * nw) ** (1/3)
                            U[:,t] *= target / nu
                            V[:,t] *= target / nv
                            W[:,t] *= target / nw
            
            if best_loss < 1e-8:
                break
        
        if best_loss < 1e-8 and best_params is not None:
            U_opt, V_opt, W_opt = best_params
            # Round to nearest half-integer
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            # Verify
            ok, _ = _verify_factors(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p)
            if ok:
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), rank
            
            # Try without rounding
            ok, _ = _verify_factors(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p)
            if ok:
                return U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), rank
    
    return None

def _structured_decomposition_attempt(n, m, p, target_rank=11):
    """Attempt a structured decomposition for M_{2,3,3} with rank < 18."""
    if n == 2 and m == 3 and p == 3:
        result = _build_rank11_decomposition()
        return result
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120 - 5  # safety margin
    
    # First, try known lower-rank constructions if available
    result = _structured_decomposition_attempt(n, m, p)
    if result is not None:
        U, V, W, rank = result
        ok, _ = _verify_factors(U, V, W, n, m, p)
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
