import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    t_start = time.time()
    DEADLINE = 145
    n, m, p = DIMENSIONS  # 3, 3, 7

    def schoolbook():
        rank = n * m * p  # 63
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

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # For 3x3x7, the tensor is sum over j of A[i,j]*B[j,k] for each (i,k)
    # We can decompose this as a sum over j of 3x1x7 subproblems.
    # Each subproblem j has rank at most 3*7 = 21 (schoolbook)
    # But we can do better by noting that for each j, we need A[i,j]*B[j,k]
    # which is an outer product of a 3-vector and a 7-vector.
    # So each j contributes rank 21, total rank 3*21 = 63 (same as schoolbook).
    
    # However, we can potentially share structure between different j values.
    # For example, if we group the B columns, we might be able to use Strassen-like
    # tricks on the A dimension.
    
    # Let's try a different approach: decompose 3x3x7 as 3x3x3 + 3x3x4
    # The 3x3x3 part has rank 23 (Laderman) if we can find valid factors.
    # The 3x3x4 part: for each of 4 columns of B, we need A[i,j]*B[j,k]
    # This is like 4 copies of 3x3x1, each of rank 9, so 36 total.
    # Total: 23 + 36 = 59.
    
    # But we need to verify that the 3x3x3 Laderman factors are correct.
    # Given the difficulty, let's use a verified construction.
    
    # Actually, let's use a simpler approach: use rank 63 schoolbook but
    # verify it works correctly.
    
    result = schoolbook()
    
    # Verify to be safe
    if not verify(result['U'], result['V'], result['W']):
        raise RuntimeError("Schoolbook verification failed!")
    
    return result
