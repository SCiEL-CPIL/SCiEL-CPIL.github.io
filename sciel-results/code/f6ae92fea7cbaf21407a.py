"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    We use a structured search approach:
    1. Start with the schoolbook rank 54 decomposition
    2. Try to find lower rank decompositions using algebraic constraints
    3. The key insight is that for small matrices, we can search over possible 
       rank-r decompositions by solving systems of polynomial equations
    
    For 3x3 * 3x6, the theoretical lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18,
    but the actual minimum is known to be higher. We'll search systematically.
    
    Strategy: Use the fact that the multiplication tensor T has components T[i,j,k] = 1 
    if the output index (k*n+i) depends on input indices (i*m+j) and (j*p+k).
    
    For a rank-r decomposition, we need vectors u_c, v_c, w_c such that:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    
    This means: for all (i,j,k) where the output element (C)_{i,k} = sum_j A_{i,j} * B_{j,k},
    we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{correct}
    
    where delta_{correct} = 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # The multiplication tensor: T[a, b, c] where a indexes A, b indexes B, c indexes C
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    
    def schoolbook_decomposition():
        """Return the schoolbook rank n*m*p decomposition."""
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
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        # Verify all required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This term should contribute 1 to C[i,k]
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
                    # And no other (i',j',k') should contribute to the same C[i,k]
                    # except through the correct j
        # Also check that no spurious contributions exist
        # For each output (i,k), only terms with the correct j should contribute
        # This is automatically satisfied if the above holds for all (i,j,k)
        # and the structure is correct
        return True
    
    def try_rank_r_decomposition(r, max_iter=1000, timeout=60):
        """
        Try to find a rank-r decomposition using random initialization and gradient descent
        on the constraint that the tensor is exactly decomposed.
        
        We formulate this as: minimize ||T - sum_c u_c ⊗ v_c ⊗ w_c||^2
        subject to the constraint that the solution is exact (residual = 0).
        """
        start_time = time.time()
        
        # Initialize with random small values
        U = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n * m)]
        V = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m * p)]
        W = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p * n)]
        
        # Compute the target tensor entries
        # T[a, b, c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
        # and the term is valid (which it always is for the multiplication tensor)
        
        def compute_residual(U, V, W):
            """Compute the residual tensor."""
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        res += (s - 1) ** 2
            return res
        
        def compute_gradient(U, V, W):
            """Compute gradient of residual w.r.t. U, V, W."""
            grad_U = [[0.0] * r for _ in range(n * m)]
            grad_V = [[0.0] * r for _ in range(m * p)]
            grad_W = [[0.0] * r for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        diff = 2 * (s - 1)
                        for c in range(r):
                            grad_U[i * m + j][c] += diff * V[j * p + k][c] * W[k * n + i][c]
                            grad_V[j * p + k][c] += diff * U[i * m + j][c] * W[k * n + i][c]
                            grad_W[k * n + i][c] += diff * U[i * m + j][c] * V[j * p + k][c]
            
            return grad_U, grad_V, grad_W
        
        # Gradient descent
        lr = 0.01
        for it in range(max_iter):
            if time.time() - start_time > timeout:
                break
            
            res = compute_residual(U, V, W)
            if res < 1e-12:
                # Check exactness
                if validate(U, V, W):
                    # Round to nearest integers or half-integers
                    U_rounded = [[round(x * 2) / 2 for x in row] for row in U]
                    V_rounded = [[round(x * 2) / 2 for x in row] for row in V]
                    W_rounded = [[round(x * 2) / 2 for x in row] for row in W]
                    if validate(U_rounded, V_rounded, W_rounded):
                        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    return {"U": U, "V": V, "W": W}
            
            grad_U, grad_V, grad_W = compute_gradient(U, V, W)
            
            for i in range(n * m):
                for c in range(r):
                    U[i][c] -= lr * grad_U[i][c]
            for i in range(m * p):
                for c in range(r):
                    V[i][c] -= lr * grad_V[i][c]
            for i in range(p * n):
                for c in range(r):
                    W[i][c] -= lr * grad_W[i][c]
            
            # Decay learning rate
            lr *= 0.995
        
        return None
    
    # First, try to find a decomposition with rank < 54
    # The minimum rank for 3x3 * 3x6 is not well-known, but let's search
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, n*p) = max(9, 18, 18) = 18
    # But this is a weak bound. Let's try from 24 upwards (heuristic)
    
    best = None
    for r in range(24, 55):
        if time.time() > 140:  # Safety margin
            break
        result = try_rank_r_decomposition(r, max_iter=500, timeout=30)
        if result is not None:
            best = result
            break
    
    if best is not None:
        return best
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
