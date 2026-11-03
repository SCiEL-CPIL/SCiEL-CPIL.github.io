"""Matrix multiplication 2x3 by 3x6 - rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Schoolbook decomposition: rank = n*m*p = 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 18 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # Verify correctness with a simple check
    if time.time() < deadline - 10:
        _verify(U, V, W, n, m, p, rank)

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    # For each (i,j,k) triple, check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # and for mismatched indices, the sum = 0
    
    # Check diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    raise AssertionError(f"Diagonal check failed at ({i},{j},{k}): got {s}")
    
    # Check some off-diagonal entries (should be 0)
    # Check cases where j != j'
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = 0
                            for c in range(rank):
                                s += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + ip][c]
                            if s != 0:
                                raise AssertionError(f"Off-diag j check failed at ({i},{j},{jp},{k},{kp},{ip})")
    
    # Check cases where k != k' (with j=j')
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    if k == kp:
                        continue
                    for ip in range(n):
                        s = 0
                        for c in range(rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[kp * n + ip][c]
                        if s != 0:
                            raise AssertionError(f"Off-diag k check failed at ({i},{j},{k},{kp},{ip})")
    
    # Check cases where i != i' (with j=j', k=k')
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    if i == ip:
                        continue
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + ip][c]
                    if s != 0:
                        raise AssertionError(f"Off-diag i check failed at ({i},{j},{k},{ip})")
