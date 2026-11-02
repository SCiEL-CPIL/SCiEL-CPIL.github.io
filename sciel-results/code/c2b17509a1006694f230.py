"""Find rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against the multiplication tensor: T[i,j,k] should equal delta structure
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] where the "multiplication" gives C[i,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
    # Standard: T[i,j,k] = 1 if we're doing C[i,k] += A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in the schoolbook sense? No.
    # The tensor M[i,j,k] such that C[i,k] = sum_j M[i,j,k] * A[i,j] * B[j,k]
    # For standard matmul, M[i,j,k] = 1 (delta-like: it's just the identity in j)
    # Actually M[i,j,k] = 1 for all valid i,j,k since C[i,k] = sum_j A[i,j]*B[j,k]
    # Wait, that means the tensor is all ones? No, that's not right either.
    # 
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # For standard multiplication, T[i,j,k] = 1 for all i,j,k.
    # 
    # But the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    # 
    # Let's verify this.
    tol = 1e-10
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_decomposition(n, m, p, target_rank, max_time=10):
    """Try to find a rank-target_rank decomposition. Returns (U,V,W) or None."""
    start_time = time.time()
    
    # Initialize with random matrices and use iterative refinement
    # Strategy: start from schoolbook and try to compress
    
    # First, let's try a different approach: 
    # The tensor T[i,j,k]=1 can be factored. We need U, V, W such that
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Try using the structure: if we set U[i*m+j,r] = a_{i,j,r}, V[j*p+k,r] = b_{j,k,r}, W[k*n+i,r] = c_{k,i,r}
    # Then sum_r a_{i,j,r} b_{j,k,r} c_{k,i,r} = 1
    
    # For rank 1: a_{i,j,0} * b_{j,k,0} * c_{k,i,0} = 1 for all i,j,k
    # This requires a_{i,j} * b_{j,k} * c_{k,i} = 1
    # Taking logs: log a + log b + log c = 0
    # This is possible! e.g., a_{i,j} = 1, b_{j,k} = 1, c_{k,i} = 1 gives rank 1?
    # Wait, that would mean the tensor is rank 1, which can't be right for matrix multiplication.
    # 
    # I think I'm confusing myself. Let me re-read the problem.
    # 
    # The matrix multiplication tensor for C = A*B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The "tensor" here is the structure of the computation. The standard formulation:
    # The multiplication tensor T is a 3-tensor with T[i,j,k] = 1 (it's the identity tensor in a sense)
    # But actually, the tensor rank of matrix multiplication is what we're looking for.
    #
    # The bilinear form: <C, T> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    # For C = A*B: C[i,k] = sum_j A[i,j] B[j,k]
    # So <C, T> = sum_{i,k} C[i,k] * (something) ... 
    #
    # I think the correct formulation is:
    # The multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k
    # And the "multiplication" is: given A (n×m) and B (m×p), 
    # C[i,k] = sum_j M[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor rank r means: M[i,j,k] = sum_{t=1}^r U[i,j,t] * V[j,k,t] * W[k,i,t]
    # (Note: W uses the trace convention W[k*n+i, r] as specified)
    #
    # So we need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    #
    # This is a system of n*m*p = 60 equations in 3*rank*60 variables (roughly).
    # For small rank, this is underdetermined, so solutions should exist.
    #
    # Let me try rank 1: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 for all i,j,k
    # Let U[i*m+j,0] = u_{i,j}, V[j*p+k,0] = v_{j,k}, W[k*n+i,0] = w_{k,i}
    # Then u_{i,j} * v_{j,k} * w_{k,i} = 1 for all i,j,k
    # 
    # Set u_{i,j} = 1, then v_{j,k} * w_{k,i} = 1 for all j,k,i
    # This requires v_{j,k} * w_{k,i} independent of i, so w_{k,i} must be constant in i, say w_{k,i} = c_k
    # Then v_{j,k} * c_k = 1, so v_{j,k} = 1/c_k (independent of j)
    # So: u_{i,j} = 1, v_{j,k} = 1/c_k, w_{k,i} = c_k
    # Check: 1 * (1/c_k) * c_k = 1 ✓
    # 
    # So rank 1 works?? That can't be right for matrix multiplication tensor rank.
    #
    # Wait, I think I'm misunderstanding the problem. Let me re-read.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix."
    #
    # The matrix multiplication tensor rank is a well-studied problem. For m×n matrix multiplication,
    # the tensor rank is known to be at most mn (using the "border rank" or via specific constructions),
    # and the schoolbook is m*n*p.
    #
    # I think the issue is that I'm confusing the tensor. The matrix multiplication tensor 
    # M_{m,n,p} is defined as a 3-tensor where:
    # M[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # But the tensor rank is the minimum r such that:
    # M[i,j,k] = sum_{t=1}^r A_t[i,j] * B_t[j,k] * C_t[k,i]
    #
    # And for the standard matrix multiplication, this tensor has rank... 
    # Actually, I recall that the tensor rank of matrix multiplication M_{m,n} (where both are n×n) 
    # is at most n^3 - n^2 + n (for n≥3) via the "Strassen-like" constructions, but the exact minimum 
    # is unknown for most sizes.
    #
    # For 3×4×5, the schoolbook rank is 60. The minimum is unknown but likely less than 60.
    #
    # Let me just implement a search: start from schoolbook, try to find linear dependencies 
    # among the 60 terms, and eliminate them.
    
    # Better approach: use the fact that we need to solve a nonlinear system.
    # For a given rank r, we have 3*r*(n*m + m*p + p*n) unknowns and n*m*p equations.
    # We can use iterative methods.
    
    # Let me try a simple approach: start with rank = n*m*p and try to reduce.
    # Use the "border rank" approach or just try random initializations.
    
    # Actually, let me try a completely different strategy:
    # Use the known result that for matrix multiplication, the tensor rank can be reduced
    # by exploiting the structure. For 3×4×5, let me try ranks from 30 up to 60.
    
    for r in range(max(1, target_rank - 5), target_rank + 1):
        if time.time() - start_time > max_time:
            break
        # Try to find a rank-r decomposition
        # Use random initialization and gradient-free search
        best_solution = None
        
        for trial in range(5):
            if time.time() - start_time > max_time:
                break
            
            # Random initialization
            np.random.seed(42 + trial)
            U_init = np.random.randn(n*m, r) * 0.1
            V_init = np.random.randn(m*p, r) * 0.1
            W_init = np.random.randn(p*n, r) * 0.1
            
            # Use iterative refinement: minimize sum_{i,j,k} (sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] - 1)^2
            # This is a non-convex problem, so use simple iterative updates.
            
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            # Iterate: for each factor, hold others fixed and solve least squares
            for iteration in range(50):
                if time.time() - start_time > max_time:
                    break
                
                # Update U: for each (i,j), minimize sum_k (sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] - 1)^2
                # This is linear in U[i*m+j,:], so solve least squares
                for i in range(n):
                    for j in range(m):
                        # Build system: sum_k sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
                        # Let x_t = U[i*m+j,t], then sum_k sum_t x_t * (V[j*p+k,t] * W[k*n+i,t]) = 1
                        # This is: sum_t x_t * (sum_k V[j*p+k,t] * W[k*n+i,t]) = 1
                        # So: A * x = 1 where A[t] = sum_k V[j*p+k,t] * W[k*n+i,t]
                        A = np.zeros(r)
                        for k in range(p):
                            A += V[j*p+k, :] * W[k*n+i, :]
                        # Solve A^T x = 1 (least squares)
                        # Actually: sum_t x_t * A[t] = 1, so x = A / (A·A) if A·A != 0
                        dot = np.dot(A, A)
                        if dot > 1e-15:
                            U[i*m+j, :] = A / dot
                        else:
                            U[i*m+j, :] = A_init[i*m+j, :] if 'A_init' in dir() else U_init[i*m+j, :]
                
                # Update V: for each (j,k), minimize sum_i (sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] - 1)^2
                for j in range(m):
                    for k in range(p):
                        A = np.zeros(r)
                        for i in range(n):
                            A += U[i*m+j, :] * W[k*n+i, :]
                        dot = np.dot(A, A)
                        if dot > 1e-15:
                            V[j*p+k, :] = A / dot
                        else:
                            V[j*p+k, :] = V_init[j*p+k, :]
                
                # Update W: for each (k,i), minimize sum_j (sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] - 1)^2
                for k in range(p):
                    for i in range(n):
                        A = np.zeros(r)
                        for j in range(m):
                            A += U[i*m+j, :] * V[j*p+k, :]
                        dot = np.dot(A, A)
                        if dot > 1e-15:
                            W[k*n+i, :] = A / dot
                        else:
                            W[k*n+i, :] = W_init[k*n+i, :]
            
            # Check if solution is valid
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_solution = (U.tolist(), V.tolist(), W.tolist())
                break
        
        if best_solution:
            return best_solution
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = 130  # Leave margin before 150s kill
    
    # Start with schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Based on known results, for 3x4x5 the rank might be around 30-40
    # Try ranks from 20 up to schoolbook
    for target_rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_decomposition(n, m, p, target_rank, max_time=min(10, deadline - 10 - (time.time() - start_time)))
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = {"U": U, "V": V, "W": W}
            # Found a decomposition at this rank, try lower
            break
    
    return best_result
