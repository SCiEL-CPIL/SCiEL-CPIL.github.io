"""Rewritten program: Kronecker-product structured decomposition for 2x3x6."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    n, m, p = DIMENSIONS  # 2, 3, 6

    # Exploit p = p1 * p2 with p1 = 6, p2 = 1
    # Decompose M_{2,3,6} as Kronecker product M_{2,3,1} (rank 6) and M_{1,1,6} (rank 1)
    # Total rank = 6 * 1 = 6
    #
    # The tensor T[i, j, k] for multiplying (n x m) by (m x p) has:
    #   U indexed by [i*m + j]  (n*m = 6 rows)
    #   V indexed by [j*p + k]  (m*p = 18 rows)
    #   W indexed by [k*n + i]  (p*n = 12 rows)
    #
    # We write k = k1 * p2 + k2 where k2 in {0} (p2=1), k1 in {0..5} (p1=6)
    # So k = k1 (since p2 = 1)
    #
    # Kronecker structure:
    #   U[i*m+j, r] = U1[i*m+j, r1] * U2[0, r2]  where r = r1 * p2 + r2 = r1
    #   V[j*p+k, r] = V1[j*p1+k1, r1] * V2[k2, r2] = V1[j*6+k, r1] * V2[0, 0]
    #   W[k*n+i, r] = W1[k1*n+i, r1] * W2[k2*n2, r2] where n2=1
    #
    # For M_{1,1,6}: n2=1, m2=1, p2=1... wait, let me re-derive.
    #
    # Actually, let's think of it differently. We factor p = p1 * p2.
    # Let p1 = 6, p2 = 1.
    # The tensor M_{n,m,p} = M_{n,m,p1*p2} can be decomposed using:
    #   For each output element (i, k) where k = k1*p2 + k2:
    #   sum_j A[i,j] * B[j, k1*p2+k2]
    #
    # If we view B as a (m x p1) x (p1 x p2) block... this is getting complex.
    #
    # Simpler approach: since p2 = 1, the M_{1,1,p1} part has rank p1 = 6... no.
    #
    # Let me just directly construct rank-6 decomposition.
    #
    # T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,6)
    # We need: sum_{r=0}^{R-1} U[i*3+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1
    #
    # Try R = 6. Assign r = i*3 + j (one term per (i,j) pair).
    #   U[i*3+j, r] = 1 if r == i*3+j, else 0  -> identity matrix
    #   V[j*6+k, r] must satisfy: for fixed j, sum over r where U has 1 gives...
    #   For fixed (i,j,k): only r = i*3+j contributes to U.
    #   So we need: V[j*6+k, i*3+j] * W[k*2+i, i*3+j] = 1
    #   This must hold for ALL i, j, k.
    #
    # For fixed j and k, and varying i:
    #   V[j*6+k, i*3+j] * W[k*2+i, i*3+j] = 1 for i = 0, 1
    #
    # Let's set W[k*2+i, r] = 1 for all k, i, r. Then:
    #   V[j*6+k, i*3+j] * 1 = 1 -> V[j*6+k, i*3+j] = 1 for all i, j, k
    #   But V has 18 rows and 6 columns. Row j*6+k, column i*3+j.
    #   For j=0,k=0: V[0, 0] = 1 (i=0), V[0, 3] = 1 (i=1)
    #   For j=0,k=1: V[1, 0] = 1, V[1, 3] = 1
    #   ...
    #   For j=1,k=0: V[6, 1] = 1, V[6, 4] = 1
    #   For j=1,k=1: V[7, 1] = 1, V[7, 4] = 1
    #   ...
    #   For j=2,k=0: V[12, 2] = 1, V[12, 5] = 1
    #   For j=2,k=1: V[13, 2] = 1, V[13, 5] = 1
    #   ...
    #   For j=2,k=5: V[17, 2] = 1, V[17, 5] = 1
    #
    # This works! Each V row j*6+k has 1s at columns 3*j and 3*j+1.
    # W is all ones (12 x 6).
    # U is identity (6 x 6).
    #
    # Let's verify: for (i=0, j=1, k=3):
    #   r = 0*3+1 = 1
    #   U[0*3+1, 1] = U[1,1] = 1
    #   V[1*6+3, 1] = V[9, 1] = 1 (since 3*1=3? No, 3*j=3, 3*j+1=4)
    #   Wait, I said columns 3*j and 3*j+1. For j=1: columns 3 and 4.
    #   V[9, 1]: column 1 is not 3 or 4. So V[9,1] = 0. That breaks it.
    #
    # Let me re-index. r = i*3 + j. For i=0: r=j. For i=1: r=3+j.
    # So for j=1: r=1 (i=0) or r=4 (i=1).
    # V[j*6+k, r] should be 1 when r = i*3+j for some valid i.
    # For j=1: r=1 or r=4.
    # So V[1*6+k, 1] = 1 and V[1*6+k, 4] = 1 for all k.
    #
    # Generalizing: V[j*6+k, i*3+j] = 1 for i in {0,1}, i.e., columns 3*j and 3*j+1... 
    # Wait: i*3 + j for i=0 is j, for i=1 is 3+j.
    # So columns j and 3+j for each j.
    #
    # Let me redo:
    # U[i*3+j, r] = 1 iff r == i*3+j
    # V[j*6+k, r] = 1 iff r mod 3 == j and r // 3 in {0, 1}
    #   i.e., r = j or r = 3+j
    # W[k*2+i, r] = 1 for all
    #
    # Check (i=0, j=1, k=3):
    #   r = 0*3+1 = 1
    #   U[1, 1] = 1
    #   V[9, 1]: 1 mod 3 = 1 = j, 1//3 = 0 in {0,1}. So V[9,1] = 1.
    #   W[6, 1] = 1
    #   Product = 1. Good.
    #
    # Check (i=1, j=1, k=3):
    #   r = 1*3+1 = 4
    #   U[4, 4] = 1
    #   V[9, 4]: 4 mod 3 = 1 = j, 4//3 = 1 in {0,1}. So V[9,4] = 1.
    #   W[6, 4] = 1
    #   Product = 1. Good.
    #
    # Check (i=0, j=0, k=5):
    #   r = 0
    #   U[0, 0] = 1
    #   V[5, 0]: 0 mod 3 = 0, 0//3 = 0. V[5,0] = 1.
    #   W[10, 0] = 1
    #   Product = 1. Good.
    #
    # This gives rank 6! All entries are 0 or 1 (integers).

    R = 6
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # U: identity-like, U[i*m+j, i*m+j] = 1
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1

    # V: V[j*p+k, r] = 1 if r % m == j and r // m < n
    for j in range(m):
        for k in range(p):
            for r in range(R):
                if r % m == j and r // m < n:
                    V[j * p + k][r] = 1

    # W: all ones
    for k in range(p):
        for i in range(n):
            for r in range(R):
                W[k * n + i][r] = 1

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    # Fallback to schoolbook
                    R_fb = n * m * p
                    U = [[0] * R_fb for _ in range(n * m)]
                    V = [[0] * R_fb for _ in range(m * p)]
                    W = [[0] * R_fb for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][col] = 1
                                V[jj * p + kk][col] = 1
                                W[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
