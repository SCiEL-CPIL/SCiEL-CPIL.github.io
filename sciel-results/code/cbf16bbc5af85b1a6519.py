"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 42
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition via random search / structured approach
    # The minimum rank for matrix multiplication (n x m) * (m x p) is at least max(n*m, m*p, p*n)
    # For 2x3x7: max(6, 21, 14) = 21
    # Actually the lower bound is max(n*m, m*p, p*n) / something, but let's use known results.
    # For 2x3x7, the schoolbook is 42. We know that rank < 42 is possible.
    # Let's try a structured approach: exploit the small dimensions.
    
    # Actually, let's try a simple random search for a valid decomposition at various ranks.
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    
    # This is a tensor decomposition problem. Let's try a few ranks.
    # Known lower bound: rank >= max(n*m, m*p, p*n) / something
    # Actually, the trivial lower bound is max(n*m, m*p, p*n) is not right.
    # The rank must be at least max(n*m, m*p, p*n) / min(n,m,p) ... no.
    
    # Let's just try to find a decomposition with rank < 42.
    # We'll use a simple iterative approach: start with random matrices and try to minimize the error.
    
    deadline = time.time() + 140  # Leave 10s margin for 150s limit
    best_rank = rank_school
    best_result = None
    
    # First, let's try the schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a lower rank decomposition
    # The minimum rank for 2x3x7 matrix multiplication is known to be less than 42.
    # Let's try ranks from a lower bound up to 41.
    
    # A simple lower bound: the rank must be at least max(n*m, m*p, p*n) / max(n,m,p)
    # Actually, let's use the fact that rank >= max(n*m, m*p, p*n) / min(n,m,p) is not standard.
    # Let's just try a few candidate ranks.
    
    # Known result: for 2x3x7, rank can be as low as 27 or so. Let's try to find one.
    # We'll use a simple optimization approach.
    
    def check_decomposition(U, V, W, r):
        """Check if U, V, W form a valid decomposition."""
        # For each (i, j, k), check: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 0.0
                    rhs = 0.0
                    for c in range(r):
                        lhs += U[i * m + j][c] * V[j * p + k][c]
                        rhs += W[k * n + i][c]
                    # Actually, the condition is:
                    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
                    # Tensor form: T_{i,j,k} = delta (indicator)
                    # Decomposition: T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
                    # Wait, let me re-read the convention.
                    # W[k*n+i, r] - so W is indexed by (k, i)
                    # The tensor for matrix multiplication (A is n x m, B is m x p, C is n x p)
                    # C_{i,k} = sum_j A_{i,j} B_{j,k}
                    # The multiplication tensor has entries T_{i,j,k} = 1 for all i,j,k (since each term contributes)
                    # No, the tensor is: T_{i,j,k} where the decomposition is:
                    # sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c} should give the coefficient of A_{i,j} B_{j,k} in C_{i,k}
                    # Which is 1 for all i,j,k.
                    # So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
                    pass
        
        # Let me redo this properly
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition using random search
    # This is a hard problem, so let's try a few structured approaches.
    
    # Approach: try rank = 21 (which is m*p = 3*7)
    # Actually, let's just return the schoolbook for now since finding a better one
    # in the time limit is challenging without a proper optimization.
    
    # But let's try a simple random search for rank < 42
    import random
    
    for target_rank in range(30, 42):
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            if check_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result
# EVOLVE-BLOCK-END
