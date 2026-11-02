"""Rewritten program for 3x4x6 matrix multiplication rank minimization."""
import time

DIMENSIONS = [3, 4, 6]

def _validate(U, V, W, n, m, p, rank):
    """Check that U, V, W exactly decompose the multiplication tensor."""
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
                    # This is the tensor condition.
                    pass
    # More efficient: build the 3D tensor and check
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 4, 6
    start_time = time.time()
    
    # Known lower bounds and upper bounds for R(3,4,6):
    # Schoolbook: 72
    # We can try to find a better decomposition.
    # For 3x4x6, we can think of it as blocks.
    # Actually, let's try to use the fact that we can sometimes reduce rank
    # by using non-integer or exploiting structure.
    
    # One known approach: R(n,m,p) <= R(n,m,p') * (p/p') if p' divides p... not exactly.
    # Let's try a direct search or known constructions.
    
    # For small sizes, the rank is often close to n*m*p.
    # Let's try to see if we can do better than 72.
    
    # Actually, for 3x4x6, we can use the following:
    # Multiply 3x4 by 4x6. We can view the 4x6 as two 4x3 blocks.
    # So C = A * [B1, B2] = [A*B1, A*B2]
    # R(3,4,3) is at most 36 (schoolbook).
    # So total rank at most 72.
    # But maybe R(3,4,3) < 36?
    # R(3,4,3): 3*4*3 = 36.
    # Is there a known better rank?
    
    # Let's try to construct a rank-71 or lower decomposition if possible.
    # For now, let's implement the schoolbook and see if we can prune.
    
    # Actually, let's try a randomized approach to see if we can find a lower rank.
    # But the task requires exact decomposition with integer/half-integer entries.
    # So we need an exact algebraic construction.
    
    # For now, return the schoolbook decomposition which is valid.
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
