"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0

    n, m, p = 2, 3, 6
    rank = 33  # 3*7 (Strassen on 2x2x2 blocks) + 12 (schoolbook 2x1x6)

    U = [[0] * rank for _ in range(n * m)]   # (6, 33)
    V = [[0] * rank for _ in range(m * p)]   # (18, 33)
    W = [[0] * rank for _ in range(p * n)]   # (12, 33)

    def urow(i, j):
        return i * m + j

    def vrow(j, k):
        return j * p + k

    def wrow(k, i):
        return k * n + i

    col = 0

    # Strassen 2x2x2 rank-7 decomposition
    # For multiplying A_sub (2x2) by B_sub (2x2) to get C_sub (2x2)
    # A_sub[i1, j1] maps to A[i1, j1] (rows 0,1 of A; cols j1, j1+1)
    # B_sub[j1, k1] maps to B[j1, k1] (rows j1, j1+1 of B; cols k1, k1+1)
    # C_sub[i1, k1] maps to C[i1, k1] (rows i1 of C; cols k1, k1+1)
    #
    # Strassen 7 products:
    # m1 = (a+d)(e+h)  -> C[0,0]+=1, C[1,1]+=1
    # m2 = (c+d)(e)    -> C[1,0]+=1
    # m3 = (a)(f-h)    -> C[0,1]+=1, C[1,1]+=1
    # m4 = (d)(g-e)    -> C[0,0]+=1, C[1,0]+=1
    # m5 = (a+b)(h)    -> C[0,1]+=1, C[1,1]+=-1
    # m6 = (c-a)(e+g)  -> C[1,1]+=-1
    # m7 = (b-d)(f+h)  -> C[0,0]+=-1, C[1,1]+=1

    # Each Strassen product is a sum of terms like (sum of A terms)(sum of B terms)
    # In tensor decomposition, each product corresponds to ONE rank-1 term
    # where U encodes the A-side linear combination, V encodes the B-side,
    # and W encodes the C-side.

    # For block b (b=0,1,2), the 2x2 sub-block of B is at columns 2b, 2b+1
    # The A sub-block uses columns j1=0 and j1=1 of A (i.e., A[*,0] and A[*,1])
    # Wait - A is 2x3, so A[i,j] for i in {0,1}, j in {0,1,2}
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:3] (2x1)
    # B1 = B[0:2, :] (2x6), B2 = B[2:3, :] (1x6)
    # B1 blocks: B1[:, 0:2], B1[:, 2:4], B1[:, 4:6]

    # For Strassen on block b:
    # A_sub[i1, j1] = A[i1, j1]  (j1 in {0,1}, so A cols 0,1)
    # B_sub[j1, k1] = B[j1, 2*b + k1]  (j1 in {0,1}, k1 in {0,1})
    # C_sub[i1, k1] = C[i1, 2*b + k1]  (i1 in {0,1}, k1 in {0,1})

    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # m1 = (a+d)(e+h): A[0,0]=1, A[1,1]=1; B[0,0]=1, B[1,1]=1; C[0,0]=1, C[1,1]=1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # m2 = (c+d)(e): A[0,1]=1, A[1,1]=1; B[0,0]=1; C[1,0]=1
        ([(0,1,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # m3 = (a)(f-h): A[0,0]=1; B[0,1]=1, B[1,1]=-1; C[0,1]=1, C[1,1]=1
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # m4 = (d)(g-e): A[1,1]=1; B[1,0]=1, B[0,0]=-1; C[0,0]=1, C[1,0]=1
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # m5 = (a+b)(h): A[0,0]=1, A[0,1]=1; B[1,1]=1; C[0,1]=1, C[1,1]=-1
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (1,1,-1)]),
        # m6 = (c-a)(e+g): A[0,1]=1, A[0,0]=-1; B[0,0]=1, B[1,0]=1; C[1,1]=-1
        ([(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,-1)]),
        # m7 = (b-d)(f+h): A[0,1]=1, A[1,1]=-1; B[0,1]=1, B[1,1]=1; C[0,0]=-1, C[1,1]=1
        ([(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], [(0,0,-1), (1,1,1)]),
    ]

    for b in range(3):
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # A-side: A[i, j] where i,j in {0,1}
            for (i1, j1, val) in a_coeffs:
                U[urow(i1, j1)][col] += val
            # B-side: B[j, 2*b + k] where j in {0,1}, k in {0,1}
            for (j1, k1, val) in b_coeffs:
                V[vrow(j1, 2 * b + k1)][col] += val
            # C-side: C[i, 2*b + k] where i in {0,1}, k in {0,1}
            for (i1, k1, val) in c_coeffs:
                W[wrow(2 * b + k1, i1)][col] += val
            col += 1

    # Schoolbook for A2 * B2: A2 is 2x1 (A[i,2]), B2 is 1x6 (B[2,k])
    # C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0,1,2,3,4,5}
    for i in range(n):
        for k in range(p):
            U[urow(i, 2)][col] = 1
            V[vrow(2, k)][col] = 1
            W[wrow(k, i)][col] = 1
            col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
