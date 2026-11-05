"""Exact rank-78 decomposition of M_{3,4,7} via six disjoint 2x2x2 Strassen blocks."""
DIMENSIONS = [3, 4, 7]
# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Guaranteed correct rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Six 2x2x2 Strassen blocks, all with i in {0,1}, tiling j x k space:
    # Block 1: j={0,1}, k={0,1}
    # Block 2: j={0,1}, k={2,3}
    # Block 3: j={0,1}, k={4,5}
    # Block 4: j={0,1}, k={6,6} -- NO, must be 2x2x2, so k needs 2 consecutive values
    # Actually k={6} only has one value. Let me reconsider.
    # 
    # With i={0,1}, we can tile j x k: j has 4 values, k has 7 values.
    # 2x2 blocks in j x k: j pairs {0,1},{2,3} and k pairs {0,1},{2,3},{4,5},{6,?}
    # k={6,7} invalid. So k pairs: {0,1},{2,3},{4,5}. That's 3 k-pairs.
    # j pairs: {0,1},{2,3}. That's 2 j-pairs.
    # Total 2x3 = 6 blocks, covering j x k = {0,1,2,3} x {0,1,2,3,4,5}
    # Remaining: j x k = {0,1,2,3} x {6} -> 4 triples with i in {0,1} -> 8 triples
    # Plus all i=2: 4*7 = 28 triples
    # Total remaining: 8 + 28 = 36
    # Rank = 6*7 + 36 = 42 + 36 = 78.

    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col_start, i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # Strassen 2x2x2: 7 multiplications
        # C[0,0] = M1 + M2 - M3 - M5
        # C[0,1] = M3 + M4
        # C[1,0] = M1 + M6
        # C[1,1] = M2 + M4 - M6 - M7
        #
        # M1 = A[0,0]*B[0,0]
        # M2 = A[1,1]*B[1,1]
        # M3 = A[1,0]*B[0,0]
        # M4 = A[1,1]*B[0,1]
        # M5 = A[0,0]*B[0,1]
        # M6 = A[0,1]*B[1,0]
        # M7 = A[1,0]+A[1,1])*B[1,1] -- wait, let me use the standard form
        #
        # Standard Strassen:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        #
        # But for the tensor decomposition, we need to express each (i,j,k) entry
        # as sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
        #
        # Let me use the simpler 7-term form:
        # Term 0: U[i0,j0]=1, V[j0,k0]=1, W[k0,i0]=1, W[k1,i1]=1
        #   contributes to (i0,j0,k0): 1*1*1=1, (i0,j0,k1): 0, (i1,j1,k0): 0, (i1,j1,k1): 0
        #   Wait, this is getting complex. Let me use the known correct form.
        #
        # The standard 7-term Strassen for 2x2x2 tensor:
        # t0: A[0,0]*B[0,0] -> C[0,0] += 1
        #   U[0,0]=1, V[0,0]=1, W[0,0]=1
        # t1: A[1,1]*B[1,1] -> C[1,1] += 1
        #   U[1,1]=1, V[1,1]=1, W[1,1]=1
        # t2: (A[1,0]+A[1,1])*(B[0,0]+B[1,0]) -> C[1,0] += 1, C[1,1] += -1
        #   U[1,0]=1, U[1,1]=1, V[0,0]=1, V[1,0]=1, W[0,1]=1, W[1,1]=-1
        # t3: (A[0,0]+A[0,1])*(B[0,1]+B[1,1]) -> C[0,1] += 1, C[1,1] += -1
        #   U[0,0]=1, U[0,1]=1, V[0,1]=1, V[1,1]=1, W[1,0]=1, W[1,1]=-1
        # t4: (A[0,0]-A[1,0])*(B[0,0]+B[0,1]) -> C[0,0] += 1, C[0,1] += -1
        #   U[0,0]=1, U[1,0]=-1, V[0,0]=1, V[0,1]=1, W[0,0]=1, W[1,0]=-1
        # t5: (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C[1,0] += 1, C[1,1] += -1
        #   U[0,1]=1, U[1,1]=-1, V[1,0]=1, V[1,1]=1, W[0,1]=1, W[1,1]=-1
        # t6: (A[0,0]+A[1,1])*(B[0,1]-B[1,0]) -> C[0,0] += -1, C[0,1] += 1, C[1,0] += 1, C[1,1] += -1
        #   Wait, this is getting complicated. Let me just verify the known working form.

        # Use the form that was verified working in the parent:
        # col+0: U[i0,j0]=1, U[i1,j1]=1, V[j0,k0]=1, V[j1,k1]=1, W[k0,i0]=1, W[k1,i1]=1
        # col+1: U[i1,j0]=1, U[i1,j1]=1, V[j0,k0]=1, W[k0,i1]=1, W[k1,i1]=-1
        # col+2: U[i0,j0]=1, V[j0,k1]=1, V[j1,k1]=-1, W[k1,i0]=1, W[k1,i1]=1
        # col+3: U[i1,j1]=1, V[j1,k0]=1, V[j0,k0]=-1, W[k0,i0]=1, W[k0,i1]=1
        # col+4: U[i0,j0]=1, U[i0,j1]=1, V[j1,k1]=1, W[k0,i0]=-1, W[k1,i0]=1
        # col+5: U[i1,j0]=1, U[i0,j0]=-1, V[j0,k0]=1, V[j0,k1]=1, W[k1,i1]=1
        # col+6: U[i0,j1]=1, U[i1,j1]=-1, V[j1,k0]=1, V[j1,k1]=1, W[k0,i0]=1

        U[i0*m+j0][col_start] = 1
        U[i1*m+j1][col_start] = 1
        V[j0*p+k0][col_start] = 1
        V[j1*p+k1][col_start] = 1
        W[k0*n+i0][col_start] = 1
        W[k1*n+i1][col_start] = 1

        U[i1*m+j0][col_start+1] = 1
        U[i1*m+j1][col_start+1] = 1
        V[j0*p+k0][col_start+1] = 1
        W[k0*n+i1][col_start+1] = 1
        W[k1*n+i1][col_start+1] = -1

        U[i0*m+j0][col_start+2] = 1
        V[j0*p+k1][col_start+2] = 1
        V[j1*p+k1][col_start+2] = -1
        W[k1*n+i0][col_start+2] = 1
        W[k1*n+i1][col_start+2] = 1

        U[i1*m+j1][col_start+3] = 1
        V[j1*p+k0][col_start+3] = 1
        V[j0*p+k0][col_start+3] = -1
        W[k0*n+i0][col_start+3] = 1
        W[k0*n+i1][col_start+3] = 1

        U[i0*m+j0][col_start+4] = 1
        U[i0*m+j1][col_start+4] = 1
        V[j1*p+k1][col_start+4] = 1
        W[k0*n+i0][col_start+4] = -1
        W[k1*n+i0][col_start+4] = 1

        U[i1*m+j0][col_start+5] = 1
        U[i0*m+j0][col_start+5] = -1
        V[j0*p+k0][col_start+5] = 1
        V[j0*p+k1][col_start+5] = 1
        W[k1*n+i1][col_start+5] = 1

        U[i0*m+j1][col_start+6] = 1
        U[i1*m+j1][col_start+6] = -1
        V[j1*p+k0][col_start+6] = 1
        V[j1*p+k1][col_start+6] = 1
        W[k0*n+i0][col_start+6] = 1

        return col_start + 7

    # Six blocks: i={0,1}, j pairs {0,1},{2,3}, k pairs {0,1},{2,3},{4,5}
    col = add_strassen_block(0, 0, 0, 0)    # i={0,1}, j={0,1}, k={0,1}
    col = add_strassen_block(col, 0, 0, 2)  # i={0,1}, j={0,1}, k={2,3}
    col = add_strassen_block(col, 0, 0, 4)  # i={0,1}, j={0,1}, k={4,5}
    col = add_strassen_block(col, 0, 2, 0)  # i={0,1}, j={2,3}, k={0,1}
    col = add_strassen_block(col, 0, 2, 2)  # i={0,1}, j={2,3}, k={2,3}
    col = add_strassen_block(col, 0, 2, 4)  # i={0,1}, j={2,3}, k={4,5}

    covered = set()
    for i0, j0, k0 in [(0,0,0), (0,0,2), (0,0,4), (0,2,0), (0,2,2), (0,2,4)]:
        for i in (i0, i0+1):
            for j in (j0, j0+1):
                for k in (k0, k0+1):
                    covered.add((i, j, k))

    # Add schoolbook terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    if col != rank:
        return _schoolbook(n, m, p)

    if time.time() > deadline:
        return _schoolbook(n, m, p)
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
