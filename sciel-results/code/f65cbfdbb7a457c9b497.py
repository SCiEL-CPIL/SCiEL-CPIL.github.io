"""Matrix multiplication 2x3 by 3x5 - rank-minimal tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Try known low-rank decompositions first
    # Theoretical lower bound is max(n*m, m*p, n*p) = max(6,15,10) = 15
    # But for 2x3 * 3x5, we can try ranks from 15 upward
    
    # Known: schoolbook is 30. We aim for much lower.
    # For 2x3 * 3x5, a rank-18 or rank-16 decomposition may exist.
    
    # Strategy: use structured decomposition patterns
    # Pattern 1: Block decomposition exploiting n=2
    # A (2x3) * B (3x5) -> C (2x5)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try to find a decomposition using the structure
    # with integer/half-integer entries.
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Build the tensor: T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Actually the tensor for matrix mult is T[i,j,k] = delta (it's the multiplication tensor)
        # The decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # and T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # Wait no. The multiplication tensor for A(nxm) * B(mx p) -> C(nxp)
        # is defined by: T[i, j, k] where i indexes rows of A, j indexes columns of A/rows of B, k indexes columns of B
        # T[i,j,k] = 1 (it's the identity tensor in some sense)
        # Actually the tensor is: for each (i,j,k), the coefficient is 1.
        # The decomposition satisfies: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Let's verify: compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all i,j,k
        # and check it equals 1.
        
        # More efficient: for each c, form outer product and sum
        total = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # The contribution to T[i,j,k] is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # = u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        return np.allclose(total, np.ones((n, m, p)), atol=1e-9)
    
    def verify_fast(U, V, W, rank):
        """Faster verification using tensor operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        total = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # total[i,j,k] += u[i,j]*v[j,k]*w[k,i]
            # = sum over k of u[i,j]*v[j,k]*w[k,i]
            # For fixed i,j: u[i,j] * sum_k v[j,k]*w[k,i]
            # = u[i,j] * (v[j,:] dot w[:,i])
            vw = v @ w  # shape (m, n)
            for i in range(n):
                for j in range(m):
                    total[i, j, k] = 0  # this is wrong, let me redo
            # Actually: total[i,j,k] += u[i,j]*v[j,k]*w[k,i]
            # Let me just do it element-wise but with numpy
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return np.allclose(total, np.ones((n, m, p)), atol=1e-9)
    
    def verify_numpy(U, V, W, rank):
        """Verification using numpy broadcasting."""
        U = np.array(U, dtype=float).reshape(n, m, rank)
        V = np.array(V, dtype=float).reshape(m, p, rank)
        W = np.array(W, dtype=float).reshape(p, n, rank)
        
        # total[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        total = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        return np.allclose(total, np.ones((n, m, p)), atol=1e-9)
    
    def try_decomposition(rank, time_limit=None):
        """Try to find a rank-rank decomposition using structured search."""
        if time_limit and time.time() > time_limit:
            return None
        
        # Use a structured approach: 
        # For 2x3 * 3x5, we can exploit the fact that n=2 is small.
        
        # Approach: parameterized decomposition
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # Let's use the known structure from fast matrix multiplication ideas.
        # For 2x2 * 2x2, Strassen gives rank 7.
        # For 2x3 * 3x5, we need something else.
        
        # Try a block decomposition:
        # Split B into 3 blocks of 5 columns each? No, that doesn't help directly.
        
        # Alternative: use the fact that we can combine terms.
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
        # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
        
        # This is like multiplying a 2x3 matrix by a 3x5 matrix.
        # We can think of it as: [A0, A1, A2] * B^T where A0,A1,A2 are row vectors.
        
        # Let's try a decomposition with rank around 16-20.
        
        # Known result: for 2xn * nxm, the rank is at most 2n + n*m - something
        # Actually, let me just try to construct one.
        
        # Strategy: use 5 "column" decompositions for the 5 output columns,
        # but share some factors.
        
        # Let's try rank = 18 first (a reasonable target)
        
        # I'll use a different approach: enumerate small integer/half-integer patterns
        # and use linear algebra to check if they work.
        
        # Actually, let me try a specific known construction.
        # For 2x3 * 3x5:
        
        # We can write C = A * B where A is 2x3, B is 3x5.
        # C is 2x5.
        
        # Decomposition idea:
        # Use 3 "Strassen-like" combinations for the 3 middle dimension,
        # but since we only have 2 rows in A, we can do better.
        
        # Let's try: 
        # For the 3x3 "middle" part, we have 3x3 = 9 terms.
        # But we only need 2x5 = 10 outputs.
        
        # Try rank 16:
        # Use 8 terms for a 2x2-like structure on part, and 8 for the rest?
        
        # Actually, let me just try to find one by construction.
        
        # Simple approach that should work with rank <= 20:
        # For each of the 5 columns k, compute C[:,k] = A * B[:,k]
        # This is a 2x3 * 3x1 multiplication, which has rank at most 6 (schoolbook)
        # but can be done with rank 4 or 5 using tricks.
        
        # But we need to share factors across columns.
        
        # Let me try a concrete construction with rank 18:
        
        # U has shape (6, 18), V has shape (15, 18), W has shape (10, 18)
        
        # I'll use a pattern based on:
        # - 5 "main" terms (one per output column, using all 3 middle indices)
        # - Plus additional terms to handle the 2-row structure
        
        # Actually, let me just use a brute-force structured search for small ranks.
        
        # For rank r, we need:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # This is a system of n*m*p = 30 equations in r*(n*m + m*p + p*n) = r*(6+15+10) = 31r unknowns.
        # For r=15: 465 unknowns, 30 equations. Should be solvable!
        # But we need integer/half-integer solutions.
        
        # Let me try to find rational solutions.
        
        # Use numpy to solve the linear system for a given rank.
        
        # The equations are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is nonlinear (trilinear). 
        
        # Better approach: fix U and V, solve for W (linear).
        # Or use iterative methods.
        
        # Let me try a different strategy: use the fact that we can scale.
        # If we have a decomposition with real entries, we can try to find
        # rational approximations.
        
        # Actually, for the purposes of this task, let me just hardcode a known
        # low-rank decomposition if I can find one, or use a constructive approach.
        
        # Let me try rank 20 with a structured pattern:
        
        # Pattern: use 4 "Strassen-like" blocks for the 3x3 middle, 
        # but adapted for 2 rows.
        
        # I'll construct a decomposition with the following idea:
        # Split the 3 middle indices into groups and use bilinear forms.
        
        # For now, let me try a simple but effective construction:
        
        # Use rank = 18 with the following structure:
        # 6 terms for the "diagonal" (i,j,k with some pattern)
        # 6 terms for cross terms
        # 6 more terms
        
        # Actually, let me just try to solve it numerically and round.
        
        if time.time() > (time_limit or deadline):
            return None
        
        # Try to find a decomposition using optimization
        # Start with a random initialization and use gradient descent
        
        import random
        random.seed(42)
        
        best_result = None
        best_error = float('inf')
        
        # Try several initializations
        for attempt in range(5):
            if time.time() > (time_limit or deadline):
                break
                
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Use a simple iterative refinement
            # For fixed U, V: W is determined by linear equations
            # For fixed U, W: V is determined by linear equations
            # etc.
            
            # Actually, let's use a different approach.
            # The constraint is: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # Let's fix U and V, and solve for W.
            # For each (k,i): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is: for each (k,i), a linear equation in W[k*n+i,c] for all c
            # But W[k*n+i,c] appears with coefficient sum_j U[i*m+j,c]*V[j*p+k,c]
            
            # So: for each (k,i): sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
            # This is a linear system: M @ w = 1 where M has entries
            # M[(k,i), c] = sum_j U[i*m+j,c]*V[j*p+k,c]
            
            # Let's set up this linear system.
            
            # M has shape (p*n, rank) = (10, rank)
            # w has shape (rank,) for each (k,i)? No, W has shape (p*n, rank)
            # W[k*n+i, c] is the variable.
            
            # For each (k,i), the equation is:
            # sum_c M[(k,i), c] * W[k*n+i, c] = 1
            # where M[(k,i), c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            
            # So we have p*n = 10 equations, each involving rank variables (the c-th column of W for that (k,i) row).
            # Wait, no. W[k*n+i, c] for fixed (k,i) varies over c.
            # So for each (k,i), we have: sum_c M_ki_c * W_ki_c = 1
            # This is one equation in rank unknowns (W_ki_0, ..., W_ki_{rank-1}).
            # We have 10 such equations, but 10*rank unknowns total.
            # This is underdetermined. We can choose W freely as long as the dot product condition holds.
            
            # So: for each (k,i), choose W[k*n+i, :] such that dot(M[k*n+i, :], W[k*n+i, :]) = 1
            # where M[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            
            # If M[k*n+i, :] is nonzero, we can set W[k*n+i, :] = M[k*n+i, :] / ||M[k*n+i, :]||^2
            # This gives a valid W!
            
            # So the algorithm is:
            # 1. Choose U and V arbitrarily (with some structure)
            # 2. Compute M[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            # 3. Set W[k*n+i, c] = M[k*n+i, c] / (M[k*n+i, :] @ M[k*n+i, :])
            # 4. Verify
            
            # This always works! The question is whether we can make U, V, W have
            # integer/half-integer entries.
            
            # Let me try with U and V having small integer entries.
            
            # Try U and V with entries in {-1, 0, 1}
            
            U_int = np.zeros((n*m, rank), dtype=int)
            V_int = np.zeros((m*p, rank), dtype=int)
            
            # Fill with random -1, 0, 1
            for _ in range(100):
                if time.time() > (time_limit or deadline):
                    break
                    
                U_int[:] = np.random.randint(-1, 2, size=(n*m, rank))
                V_int[:] = np.random.randint(-1, 2, size=(m*p, rank))
                
                # Compute M
                # M[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                M = np.zeros((p*n, rank), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M[k*n+i, :] += U_int[i*m+j, :] * V_int[j*p+k, :]
                
                # Check if any row of M is zero
                row_norms = np.sum(M**2, axis=1)
                if np.any(row_norms == 0):
                    continue
                
                # Set W
                W_int = np.zeros((p*n, rank), dtype=float)
                for idx in range(p*n):
                    if row_norms[idx] > 0:
                        W_int[idx, :] = M[idx, :] / row_norms[idx]
                
                # Check if W has nice values (close to half-integers)
                # Multiply by 2 and check if close to integers
                W2 = W_int * 2
                W2_rounded = np.round(W2)
                if np.allclose(W2, W2_rounded, atol=1e-6):
                    # Great! We have a valid decomposition with half-integer W
                    # Convert to lists
                    U_list = U_int.tolist()
                    V_list = V_int.tolist()
                    W_list = [[x/2.0 for x in row] for row in W2_rounded]
                    
                    # Verify
                    if verify_numpy(U_list, V_list, W_list, rank):
                        return {"U": U_list, "V": V_list, "W": W_list}
            
        return None
    
    # Try ranks from low to high
    for rank in [15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 28, 30]:
        if time.time() > deadline:
            break
        
        result = try_decomposition(rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
