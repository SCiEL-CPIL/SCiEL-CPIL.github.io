"""Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]
n, m, p = DIMENSIONS
N = n * m  # 20
M = m * p  # 25
P = p * n  # 20

# The matrix multiplication tensor T[i, j, k] = 1 if we're multiplying A[i,j] * B[j,k]
# We want: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
# where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
# Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]

def construct():
    """Construct a low-rank decomposition of the 4x5*5x5 matrix multiplication tensor."""
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s limit
    
    # Try different ranks, starting from lower bounds
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Also: rank >= max(n, m, p) * something... 
    # Actually the trivial lower bound is max(n*m, m*p, p*n) = 25
    # But we can potentially do better with clever decompositions
    
    best_result = None
    
    # First, let's try to find a decomposition with rank as low as possible
    # Using a systematic approach: try ranks from 25 upward
    
    # For small matrices, we can use algebraic approaches
    # The key insight: we can group terms
    
    # Let's try rank = 25 first (the trivial lower bound from max dimension product)
    # Actually, the schoolbook is 4*5*5 = 100
    # The lower bound is max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    
    # Let's construct a decomposition of rank 25 using the following idea:
    # For each pair (j, k) in the "inner" dimension, we can potentially share
    # but that's not quite right.
    
    # Alternative approach: use the fact that n=4, m=5, p=5
    # We can try to find a rank-r decomposition by solving a system of equations
    
    # Let's try a structured approach:
    # U[i*m+j, c] = sum over some grouping
    # 
    # Actually, let's use a different strategy:
    # The tensor T has entries T[i,j,k] where i in [0,4), j in [0,5), k in [0,5)
    # T[i,j,k] = 1 for all valid (i,j,k) -- wait, no.
    # 
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 (it's the identity in the "multiplication" sense)
    # Actually T[i,j,k] = delta (always 1) for the standard matmul tensor.
    # The tensor entry at (i,j,k) is 1, meaning A[i,j]*B[j,k] contributes to C[i,k].
    
    # So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for the "wrong" combinations (where the indices don't match for matmul),
    # the sum should be 0. But actually in the standard tensor formulation:
    # The matmul tensor has T[i,j,k] = 1 for all i,j,k in their ranges,
    # and the decomposition is: A[i,j]*B[j,k] -> C[i,k] via sum_c
    # 
    # Wait, I need to be more careful. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # has a 3-tensor T of shape (n*m, m*p, p*n) where:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # and all other entries are 0.
    #
    # So the constraint is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other (a, b, c) where the indices don't form a valid (i,j,k) triple:
    #   sum_c U[a, c] * V[b, c] * W[c_idx, c] = 0
    #
    # More precisely, a = i*m+j, b = j'*p+k, c_idx = k'*n+i'
    # The sum is 1 iff j=j' and k=k' and i=i', else 0.
    
    # This is a very structured problem. Let's use the approach of:
    # Fixing some indices and solving for the rest.
    
    # Strategy: Try to find a decomposition of rank r by:
    # 1. Start with a random or structured initialization
    # 2. Use iterative methods to converge
    
    # For rank 25, let's try a structured approach:
    # U[i*m+j, c] where c indexes 0..24
    # Let c = j*5 + k (so c ranges over j in [0,5), k in [0,5))
    # Then V[j*p+k, c] = V[j*5+k, j'*5+k'] = delta_{j,j'} * delta_{k,k'}
    # And we need: sum_{j',k'} U[i*5+j, j'*5+k'] * delta_{j,j'} * delta_{k,k'} * W[k*5+i, j'*5+k']
    # = U[i*5+j, j*5+k] * W[k*5+i, j*5+k] = 1
    # So we need U[i*5+j, j*5+k] * W[k*5+i, j*5+k] = 1 for all i,j,k
    # This means U[i*5+j, j*5+k] = 1/W[k*5+i, j*5+k]
    # 
    # But U and W are independent matrices, so we can set:
    # U[i*5+j, j*5+k] = 1 and W[k*5+i, j*5+k] = 1
    # Then the product is 1. 
    #
    # But wait, we also need the "cross terms" to be zero.
    # The cross term for (a, b, c_idx) = (i*m+j, j'*p+k, k'*n+i') where not all match:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    # = sum_{j'',k''} U[i*5+j, j''*5+k''] * delta_{j',j''} * delta_{k,k''} * W[k'*5+i', j''*5+k'']
    # = U[i*5+j, j'*5+k] * W[k'*5+i', j'*5+k]
    #
    # We need this to be 0 unless i=i', j=j', k=k'.
    # With U[i*5+j, j'*5+k] = 1 and W[k'*5+i', j'*5+k] = 1, we get 1 always, which is wrong.
    
    # So the simple structured approach doesn't work directly. We need a more clever decomposition.
    
    # Let's try a different structured approach. 
    # Idea: Use the fact that n=4 is small.
    # 
    # Alternative: use a "block" approach where we decompose based on the i-index.
    
    # Actually, let me reconsider. The standard result is that the rank of the (n,m,p) matmul tensor
    # is at least max(n*m, m*p, p*n) and at most n*m*p.
    # For (4,5,5), the lower bound is 25 and upper bound is 100.
    #
    # Known results: the rank of (n,m,p) matmul is exactly n*m*p for "generic" triples,
    # but for special cases it can be lower. For (2,2,2) it's 7 (Strassen).
    # For (4,5,5), I'm not sure of the exact rank, but let's try to find a good one.
    
    # Let's use a numerical optimization approach:
    # Minimize the residual subject to the constraint that the solution is exact.
    
    # Approach: For a given rank r, we have r^2 * (N + M + P) variables (actually r*(N+M+P)).
    # The constraints are: for each valid (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # and for invalid triples: sum_c U[a,c]*V[b,c]*W[c_idx,c] = 0
    
    # Total number of constraints: N*M*P = 20*25*20 = 10000
    # But many are automatically satisfied by structure.
    
    # Let's use a simpler approach: try small ranks and use random initialization + optimization.
    
    # Actually, for this problem size, let's try to use the following known technique:
    # The (4,5,5) matmul can be done with rank at most... let's just try to find one.
    
    # Let's try rank 50 first as a starting point, then try to reduce.
    
    def try_rank(r, max_iter=500, seed=42):
        """Try to find a rank-r decomposition."""
        if time.time() - start_time > deadline - 5:
            return None
        
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        # Initialize
        U = np_rng.standard_normal((N, r)) * 0.1
        V = np_rng.standard_normal((M, r)) * 0.1
        W = np_rng.standard_normal((P, r)) * 0.1
        
        # Build the constraint matrix for the "on-diagonal" terms
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of n*m*p = 100 equations.
        
        # For the "off-diagonal" terms, we need them to be 0.
        # Total off-diagonal terms: N*M*P - n*m*p = 10000 - 100 = 9900
        
        # Let's focus on satisfying the 100 "on-diagonal" equations first,
        # and hope the off-diagonal ones work out, or add them as soft constraints.
        
        # Actually, let's use a different approach: parameterize and use least squares.
        
        # For fixed V and W, the problem in U is linear:
        # For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        # This is a linear system in U's entries.
        
        # Let's use alternating minimization:
        # 1. Fix V, W, solve for U (linear least squares)
        # 2. Fix U, W, solve for V
        # 3. Fix U, V, solve for W
        
        def solve_for_U(V, W, rhs_diag, rhs_off):
            """Solve for U given V, W."""
            # For each row a of U (a = i*m+j), we need:
            # sum_c U[a,c] * V[b,c] * W[c_idx,c] = target(a,b,c_idx)
            # for all b, c_idx.
            # This is: U[a,:] dot (V[b,:] * W[c_idx,:]) = target
            # For fixed a, this is a linear system in U[a,:].
            # But it's overdetermined (M*P = 500 equations for r unknowns).
            
            # Let's use least squares.
            # Build A matrix: for each (b, c_idx), row = V[b,:] * W[c_idx,:]
            # But that's 500 rows, which is fine.
            
            U_sol = np.zeros((N, r))
            for a in range(N):
                # Build the system
                A = np.zeros((M * P, r))
                rhs = np.zeros(M * P)
                idx = 0
                for b in range(M):
                    for c_idx in range(P):
                        A[idx, :] = V[b, :] * W[c_idx, :]
                        # Determine target
                        # a = i*m+j, b = j'*p+k, c_idx = k'*n+i'
                        i = a // m
                        j = a % m
                        j_prime = b // p
                        k = b % p
                        k_prime = c_idx // n
                        i_prime = c_idx % n
                        if i == i_prime and j == j_prime and k == k_prime:
                            rhs[idx] = 1.0
                        else:
                            rhs[idx] = 0.0
                        idx += 1
                # Solve least squares
                U_sol[a, :], residuals, rank_, sv = np.linalg.lstsq(A, rhs, rcond=None)
            return U_sol
        
        def solve_for_V(U, W, rhs_diag, rhs_off):
            """Solve for V given U, W."""
            V_sol = np.zeros((M, r))
            for b in range(M):
                A = np.zeros((N * P, r))
                rhs = np.zeros(N * P)
                idx = 0
                for a in range(N):
                    for c_idx in range(P):
                        A[idx, :] = U[a, :] * W[c_idx, :]
                        i = a // m
                        j = a % m
                        j_prime = b // p
                        k = b % p
                        k_prime = c_idx // n
                        i_prime = c_idx % n
                        if i == i_prime and j == j_prime and k == k_prime:
                            rhs[idx] = 1.0
                        else:
                            rhs[idx] = 0.0
                        idx += 1
                V_sol[b, :], residuals, rank_, sv = np.linalg.lstsq(A, rhs, rcond=None)
            return V_sol
        
        def solve_for_W(U, V, rhs_diag, rhs_off):
            """Solve for W given U, V."""
            W_sol = np.zeros((P, r))
            for c_idx in range(P):
                A = np.zeros((N * M, r))
                rhs = np.zeros(N * M)
                idx = 0
                for a in range(N):
                    for b in range(M):
                        A[idx, :] = U[a, :] * V[b, :]
                        i = a // m
                        j = a % m
                        j_prime = b // p
                        k = b % p
                        k_prime = c_idx // n
                        i_prime = c_idx % n
                        if i == i_prime and j == j_prime and k == k_prime:
                            rhs[idx] = 1.0
                        else:
                            rhs[idx] = 0.0
                        idx += 1
                W_sol[c_idx, :], residuals, rank_, sv = np.linalg.lstsq(A, rhs, rcond=None)
            return W_sol
        
        # Alternating minimization
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 5:
                return None
            
            U = solve_for_U(V, W, None, None)
            V = solve_for_V(U, W, None, None)
            W = solve_for_W(U, V, None, None)
        
        # Check residual
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c_idx, :])
                    max_res = max(max_res, abs(val - 1.0))
        
        # Check off-diagonal
        for a in range(N):
            for b in range(M):
                for c_idx in range(P):
                    i = a // m
                    j = a % m
                    j_prime = b // p
                    k = b % p
                    k_prime = c_idx // n
                    i_prime = c_idx % n
                    if not (i == i_prime and j == j_prime and k == k_prime):
                        val = np.dot(U[a, :], V[b, :] * W[c_idx, :])
                        max_res = max(max_res, abs(val))
        
        if max_res < 1e-8:
            return U, V, W, max_res
        
        return None
    
    # Try ranks from 25 upward
    for r in range(25, 101):
        if time.time() - start_time > deadline - 10:
            break
        
        result = try_rank(r, max_iter=200, seed=r)
        if result is not None:
            U, V, W, max_res = result
            # Convert to lists
            U_list = [[float(U[i, j]) for j in range(r)] for i in range(N)]
            V_list = [[float(V[i, j]) for j in range(r)] for i in range(M)]
            W_list = [[float(W[i, j]) for j in range(r)] for i in range(P)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
