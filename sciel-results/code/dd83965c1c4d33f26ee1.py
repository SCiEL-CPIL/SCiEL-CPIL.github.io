"""Rewritten program: search for rank-minimal tensor decomposition of 2x2 * 2x15 matrix multiplication."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def verify_full(U, V, W, n, m, p):
        """Full verification: check all entries of the tensor."""
        rank = U.shape[1]
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check all zero entries
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                if (i == i2 and j == j2 and k == k2):
                                    continue
                                a = i * m + j
                                b = j2 * p + k
                                c = k2 * n + i2
                                val = 0.0
                                for r in range(rank):
                                    val += U[a, r] * V[b, r] * W[c, r]
                                if abs(val) > 1e-9:
                                    return False
        return True
    
    # Schoolbook solution (rank 60) as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try rank 30 with structured approach
    # Key insight: use V as a general invertible matrix to provide enough DOF
    # For rank 30: U is 4x30, V is 30x30, W is 30x4
    # 
    # Strategy: Fix V to be a random invertible 30x30 matrix, then solve for U and W
    # using the linear system approach.
    #
    # For fixed V, the equation sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] is bilinear
    # in U and W. We can fix U and solve linearly for W, or vice versa.
    #
    # Better approach: For each row a of U, we have U[a,:] as a 30-dim vector.
    # The tensor slice T[a,b,c] as a function of (b,c) must equal 
    # sum_r U[a,r] * V[b,r] * W[c,r] = (U[a,:] * V) elementwise-product with W^T... 
    # Actually: for fixed a, define M_a[b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is an outer-product-like structure: M_a = V^T * diag(U[a,:]) * W^T... 
    # No: M_a[b,c] = sum_r (V[b,r] * U[a,r]) * W[c,r]
    # Let X_a[r] = U[a,r], then M_a[b,c] = sum_r V[b,r]*X_a[r]*W[c,r]
    # = sum_r (V[b,r]*W[c,r]) * X_a[r]
    # This is not a simple matrix product.
    #
    # Alternative: fix W, then for each c, the equation becomes:
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # Let Y_c[r] = W[c,r], then sum_r (U[a,r]*V[b,r]) * Y_c[r] = T[a,b,c]
    # For fixed c: sum_r U[a,r]*V[b,r]*Y_c[r] = T[a,b,c] for all a,b
    # This is: (U * diag(Y_c) * V^T)[a,b] = T[a,b,c]... 
    # Actually: sum_r U[a,r]*Y_c[r]*V[b,r] = (U * diag(Y_c) * V^T)[a,b]
    # So for fixed c: U * diag(W[c,:]) * V^T = T[:,:,c]
    #
    # If V is invertible, then: U * diag(W[c,:]) = T[:,:,c] * (V^T)^{-1}
    # Let D_c = diag(W[c,:]), then U * D_c = T[:,:,c] * (V^T)^{-1}
    # So U[:,r] * W[c,r] = (T[:,:,c] * (V^T)^{-1})[:,r] for each r
    # This means: U[:,r] is proportional to (T[:,:,c] * (V^T)^{-1})[:,r] / W[c,r]
    # But this must hold for ALL c, so U[:,r] must be consistent across all c.
    #
    # This is getting complex. Let me try a different approach:
    # Use the fact that the tensor is very sparse (60 nonzeros out of 3600).
    # 
    # Let me try: for rank 30, use the following structure:
    # r = 0..29, where r = j*15 + k for j in {0,1}, k in 0..14
    # 
    # V[j*p+k, r] = V[j*15+k, j'*15+k'] 
    # Let V be a block-diagonal structure or something clever.
    #
    # Actually, let me try a completely different approach:
    # Use algebraic geometry / numerical algebraic geometry to find the decomposition.
    # 
    # For now, let me try random search with a smarter initialization and 
    # iterative refinement.
    
    # Try rank 30 with random search
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    np.random.seed(42)
    
    for rank in [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with a structured guess
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Iterative refinement: alternate solving for U, V, W
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U: fix V, W
                # For each a: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # This is a system of equations in U[a,:] for each a
                # For fixed a: the equations are sum_r U[a,r]*(V[b,r]*W[c,r]) = T[a,b,c]
                # Let M_a[b,c] = sum_r V[b,r]*W[c,r] (outer product of V^T and W^T... 
                # Actually M_a doesn't depend on a! M[b,c] = sum_r V[b,r]*W[c,r]
                # So: sum_r U[a,r] * (V[:,r] * W[:,r])[b,c]... no.
                # M[b,c] = sum_r V[b,r]*W[c,r] is fixed.
                # Then: sum_r U[a,r] * ??? 
                # Actually the equation is: for each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # Group by a: for fixed a, we have equations over (b,c):
                # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # This is not linear in U[a,:] in a simple way because of the V[b,r]*W[c,r] coupling.
                # 
                # Better: vectorize. For fixed a, define the vector u_a = U[a,:] (length rank)
                # The equations are: for all b,c: sum_r u_a[r]*V[b,r]*W[c,r] = T[a,b,c]
                # Let's define for each r: v_r = V[:,r] (length m*p), w_r = W[:,r] (length p*n)
                # Then: sum_r u_a[r] * v_r[b] * w_r[c] = T[a,b,c]
                # This is a trilinear form. For fixed a, it's linear in u_a.
                # The coefficient matrix for u_a is: A_a where A_a[(b,c), r] = V[b,r]*W[c,r]
                # But this is the same for all a! A[(b,c),r] = V[b,r]*W[c,r]
                # So: A * u_a = T[a,:,:] (vectorized)
                # where A is (m*p * p*n) x rank
                # 
                # If A is full column rank (rank), we can solve: u_a = A^+ * T[a,:,:]
                # where A^+ is the pseudo-inverse.
                
                # Build A: A[(b,c), r] = V[b,r]*W[c,r]
                # A has shape (m*p*p*n, rank) = (30*30, rank) = (900, rank)
                # T vectorized has shape (m*p*p*n,) for each a
                
                # Actually, let's be more careful. The tensor T has shape (n*m, m*p, p*n) = (4, 30, 30)
                # For each a in 0..3: T[a, :, :] is a 30x30 matrix
                # We need: sum_r U[a,r] * V[:,r] * W[:,r]^T (outer product) = T[a,:,:]
                # i.e., sum_r U[a,r] * (V[:,r] * W[:,r]^T) = T[a,:,:]
                
                # Let's solve for U given V, W using least squares
                # Stack all equations: for each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # Total equations: 4*30*30 = 3600, unknowns: 4*rank
                # 
                # For each a: 30*30 = 900 equations in rank unknowns
                # A_a[r, (b,c)] = V[b,r]*W[c,r]
                # A_a * U[a,:] = T[a,:,:].flatten()
                
                # Build the system
                A_mat = np.zeros((m*p*p*n, rank))  # (900, rank)
                for r in range(rank):
                    # V[:,r] has shape (m*p,) = (30,)
                    # W[:,r] has shape (p*n,) = (30,)
                    # Outer product: (30, 30) -> flatten to (900,)
                    A_mat[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                # For each a, solve A_mat * U[a,:] = T[a,:,:].flatten()
                # T[a,b,c] = 1 if b = j*p+k and c = k*n+i where a = i*m+j
                # i.e., T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
                # For a = i*m+j: T[a, j*p+k, k*n+i] = 1 for all k, else 0
                
                T_flat = np.zeros((n*m, m*p*p*n))
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        for k in range(p):
                            b = j * p + k
                            c = k * n + i
                            T_flat[a, b * p*n + c] = 1.0
                
                # Solve for U
                try:
                    U_sol, _, _, _ = np.linalg.lstsq(A_mat, T_flat.T, rcond=None)
                    U = U_sol.T  # shape (n*m, rank)
                except:
                    pass
                
                # Solve for W given U, V
                # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # For each c: sum_r (U[:,r]*V[:,r])[a,b] * W[c,r] = T[:,:,c]
                # B[c, (a,b), r] = U[a,r]*V[b,r]
                # B_c * W[c,:] = T[:,:,c].flatten()
                
                B_mat = np.zeros((n*m*m*p, rank))  # (4*30, rank) = (120, rank)
                for r in range(rank):
                    B_mat[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                T_flat_c = np.zeros((p*n, n*m*m*p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            T_flat_c[c, a * m*p + b] = 1.0
                
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(B_mat, T_flat_c.T, rcond=None)
                    W = W_sol.T  # shape (p*n, rank)
                except:
                    pass
                
                # Solve for V given U, W
                # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # For each b: sum_r (U[:,r]*W[:,r])[a,c] * V[b,r] = T[:,b,:]
                # C[b, (a,c), r] = U[a,r]*W[c,r]
                
                C_mat = np.zeros((n*m*p*n, rank))  # (4*30, rank) = (120, rank)
                for r in range(rank):
                    C_mat[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
                T_flat_b = np.zeros((m*p, n*m*p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            T_flat_b[b, a * p*n + c] = 1.0
                
                try:
                    V_sol, _, _, _ = np.linalg.lstsq(C_mat, T_flat_b.T, rcond=None)
                    V = V_sol.T  # shape (m*p, rank)
                except:
                    pass
                
                # Check convergence
                if it % 10 == 0 or it == 199:
                    if verify_full(U, V, W, n, m, p):
                        found = True
                        break
            
            if found:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    if best_rank < rank_schoolbook:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Return schoolbook solution
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
