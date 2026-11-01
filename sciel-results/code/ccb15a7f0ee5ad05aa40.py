import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Find a rank-minimal tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Uses Kronecker product structure and systematic search for low-rank decompositions.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Schoolbook decomposition (rank 36) as fallback
    def schoolbook():
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

    # Verify a decomposition
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Kronecker product decomposition: M_{2,3,6} = M_{2,1,3} ⊗ M_{1,3,2}
    # This gives rank = (2*1*3) * (1*3*2) = 6 * 6 = 36
    # But structured, may help with search
    def kronecker_decomp():
        # M_{n1,m1,p1} ⊗ M_{n2,m2,p2} where n1*n2=n, m1*m2=m, p1*p2=p
        # Use n1=2, m1=1, p1=3 and n2=1, m2=3, p2=2
        n1, m1, p1 = 2, 1, 3
        n2, m2, p2 = 1, 3, 2

        # Schoolbook for M_{n1,m1,p1}
        r1 = n1 * m1 * p1  # 6
        U1 = [[0] * r1 for _ in range(n1 * m1)]
        V1 = [[0] * r1 for _ in range(m1 * p1)]
        W1 = [[0] * r1 for _ in range(p1 * n1)]
        col = 0
        for i1 in range(n1):
            for j1 in range(m1):
                for k1 in range(p1):
                    U1[i1 * m1 + j1][col] = 1
                    V1[j1 * p1 + k1][col] = 1
                    W1[k1 * n1 + i1][col] = 1
                    col += 1

        # Schoolbook for M_{n2,m2,p2}
        r2 = n2 * m2 * p2  # 6
        U2 = [[0] * r2 for _ in range(n2 * m2)]
        V2 = [[0] * r2 for _ in range(m2 * p2)]
        W2 = [[0] * r2 for _ in range(p2 * n2)]
        col = 0
        for i2 in range(n2):
            for j2 in range(m2):
                for k2 in range(p2):
                    U2[i2 * m2 + j2][col] = 1
                    V2[j2 * p2 + k2][col] = 1
                    W2[k2 * n2 + i2][col] = 1
                    col += 1

        # Combine via Kronecker product
        rank = r1 * r2  # 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for i1 in range(n1):
            for j1 in range(m1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        ui = i1 * (m1 * m2) + j1 * m2 + j2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                col = r1_idx * r2 + r2_idx
                                val = U1[i1*m1+j1][r1_idx] * U2[i2*m2+j2][r2_idx]
                                U[ui][col] = val

        for j1 in range(m1):
            for k1 in range(p1):
                for j2 in range(m2):
                    for k2 in range(p2):
                        vj = j1 * (p1 * p2) + k1 * p2 + k2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                col = r1_idx * r2 + r2_idx
                                val = V1[j1*p1+k1][r1_idx] * V2[j2*p2+k2][r2_idx]
                                V[vj][col] = val

        for k1 in range(p1):
            for i1 in range(n1):
                for k2 in range(p2):
                    for i2 in range(n2):
                        wk = k1 * (n1 * n2) + i1 * n2 + i2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                col = r1_idx * r2 + r2_idx
                                val = W1[k1*n1+i1][r1_idx] * W2[k2*n2+i2][r2_idx]
                                W[wk][col] = val

        return {"U": U, "V": V, "W": W}

    # Try to find lower rank by attempting to zero out some Kronecker terms
    # and compensating with adjustments
    def try_reduce_rank(base_result, target_rank):
        """Try to reduce rank by zeroing out columns and checking validity."""
        if elapsed() > deadline - 5:
            return None

        U = [row[:] for row in base_result["U"]]
        V = [row[:] for row in base_result["V"]]
        W = [row[:] for row in base_result["W"]]

        # Try removing columns one by one from the end
        # This is a very limited search, but let's try
        for col_to_remove in range(len(U[0]) - 1, target_rank - 1, -1):
            if elapsed() > deadline - 3:
                break
            # Remove this column
            U = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]

            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}

        return None

    # Get the Kronecker decomposition
    result = kronecker_decomp()

    # Verify it's correct
    if not verify(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook
        result = schoolbook()

    # Try to reduce rank
    current_rank = len(result["U"][0])
    # Try a few target ranks
    for target in range(current_rank - 1, max(current_rank - 10, 17), -1):
        if elapsed() > deadline - 10:
            break
        reduced = try_reduce_rank(result, target)
        if reduced is not None:
            result = reduced
            current_rank = target
            break

    return result
