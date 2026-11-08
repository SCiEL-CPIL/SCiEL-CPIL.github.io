"""Matrix multiplication 4x5, 5x6 with Kronecker/block decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]
DEADLINE_S = 115  # safety margin before 120s kill


def _small_decomp(n, m, p):
    """Return (U, V, W) of shapes (n*m, r), (m*p, r), (p*n, r) for small n,m,p.
    Uses schoolbook for very small, or known Strassen-like for 2x2x2."""
    if n == 1:
        # T_{0,j,k} = 1 for all j,k
        # rank = m (one per j, W all ones)
        r = m
        U = np.ones((1 * m, r))
        V = np.eye(m, r)  # V[j, j'] = delta_{j,j'}
        W = np.ones((p * 1, r))
        return U, V, W
    if p == 1:
        # T_{i,j,0} = 1
        r = n
        U = np.eye(n, r).reshape(n * m, r)  # U[i*m+j, i] = delta
        # Actually U[i*m+j, i] = 1
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                U[i * m + j, i] = 1
        V = np.ones((m * 1, r))
        W = np.ones((1 * n, r))
        return U, V, W
    if n == 2 and m == 2 and p == 2:
        # Strassen: rank 7
        # Standard Strassen for 2x2 * 2x2
        # M1 = (A11-A22)(B22-B11)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A22)(B11+B22)
        # M6 = (A12-A22)(B21+B22)
        # M7 = (A21-A11)(B11+B12)
        # C11 = M5 + M4 - M2 + M6
        # C12 = M1 + M3
        # C21 = M2 + M4
        # C22 = M5 - M1 + M3 - M7
        r = 7
        # Indexing: A_{i,j}, B_{j,k}, C_{i,k}
        # U indexed by (i,j), V by (j,k), W by (k,i) per trace convention W[k*n+i]
        # For 2x2, n=2, m=2, p=2
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W indices: (k,i): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

        # M1: (A11-A22)(B22-B11) -> A[0]-A[3], B[3]-B[0]
        U[0, 0] = 1; U[3, 0] = -1
        V[3, 0] = 1; V[0, 0] = -1
        W[1, 0] = 1  # C12
        W[3, 0] = -1  # C22

        # M2: (A21+A22)B11 -> A[2]+A[3], B[0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = -1  # C11
        W[2, 1] = 1  # C21

        # M3: A11(B12-B22) -> A[0], B[1]-B[3]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C12
        W[3, 2] = 1  # C22

        # M4: A22(B21-B11) -> A[3], B[2]-B[0]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21

        # M5: (A11+A22)(B11+B22) -> A[0]+A[3], B[0]+B[3]
        U[0, 4] = 1; U[3, 4] = 1
        V[0, 4] = 1; V[3, 4] = 1
        W[0, 4] = 1  # C11
        W[3, 4] = 1  # C22

        # M6: (A12-A22)(B21+B22) -> A[1]-A[3], B[2]+B[3]
        U[1, 5] = 1; U[3, 5] = -1
        V[2, 5] = 1; V[3, 5] = 1
        W[0, 5] = 1  # C11

        # M7: (A21-A11)(B11+B12) -> A[2]-A[0], B[0]+B[1]
        U[2, 6] = 1; U[0, 6] = -1
        V[0, 6] = 1; V[1, 6] = 1
        W[3, 6] = -1  # C22

        return U, V, W
    else:
        # Fallback: schoolbook
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W


def _kron_decomp(n1, m1, p1, n2, m2, p2):
    """Kronecker product of two decompositions.
    Combines (n1,m1,p1) and (n2,m2,p2) into (n1*n2, m1*m2, p1*p2).
    Rank = r1 * r2.
    """
    U1, V1, W1 = _small_decomp(n1, m1, p1)
    U2, V2, W2 = _small_decomp(n2, m2, p2)
    r1, r2 = U1.shape[1], U2.shape[1]
    r = r1 * r2

    # For Kronecker:
    # A is (n1*n2) x (m1*m2), viewed as block matrix
    # A_{(i1,i2),(j1,j2)} = A1_{i1,j1} * A2_{i2,j2}
    # B_{(j1,j2),(k1,k2)} = B1_{j1,k1} * B2_{j2,k2}
    # C_{(i1,i2),(k1,k2)} = C1_{i1,k1} * C2_{i2,k2}
    #
    # U index: (i1*n2+i2, j1*m2+j2) -> r1*r2
    # V index: (j1*m2+j2, k1*p2+k2) -> r1*r2
    # W index: (k1*p2+k2, i1*n2+i2) -> r1*r2 (trace convention)
    #
    # U[i1*n2+i2, j1*m2+j2, (r1_idx, r2_idx)] = U1[i1*n1+m1... wait
    # Actually U1 is (n1*m1, r1), indexed by (i1,j1)
    # U2 is (n2*m2, r2), indexed by (i2,j2)
    # U = U1 kron U2 in a specific way
    # U[(i1*n2+i2), (j1*m2+j2), (a,b)] = U1[(i1,j1), a] * U2[(i2,j2), b]

    U = np.zeros((n1 * n2 * m1 * m2, r))
    V = np.zeros((m1 * m2 * p1 * p2, r))
    W = np.zeros((p1 * p2 * n1 * n2, r))

    for a in range(r1):
        for b in range(r2):
            ab = a * r2 + b
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U1[i1 * m1 + j1, a]
                    if u1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2 = U2[i2 * m2 + j2, b]
                            if u2 == 0:
                                continue
                            # U index
                            ui = i1 * n2 + i2
                            uj = j1 * m2 + j2
                            U[ui * (m1 * m2) + uj, ab] = u1 * u2

                            # V index: (j1,j2,k1,k2)
                            for k1 in range(p1):
                                v1 = V1[j1 * p1 + k1, a]
                                if v1 == 0:
                                    continue
                                for k2 in range(p2):
                                    v2 = V2[j2 * p2 + k2, b]
                                    if v2 == 0:
                                        continue
                                    vi = (j1 * p1 + k1) * (m2 * p2) + (j2 * p2 + k2)
                                    # Wait, V is (m*p, r) where m = m1*m2, p = p1*p2
                                    # V index: (j, k) where j is in [0, m1*m2), k in [0, p1*p2)
                                    # j = j1*m2 + j2, k = k1*p2 + k2
                                    vi = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                                    V[vi, ab] = v1 * v2

                                    # W index: (k, i) where k in [0, p1*p2), i in [0, n1*n2)
                                    # W index: (k1*p2+k2, i1*n2+i2)
                                    for _ in range(0):  # placeholder, W depends on W1, W2
                                        pass
                                    wi = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                                    w1 = W1[k1 * n1 + i1, a]
                                    w2 = W2[k2 * n2 + i2, b]
                                    if w1 == 0 or w2 == 0:
                                        continue
                                    W[wi, ab] = w1 * w2
    return U, V, W


def _add_decomps(U1, V1, W1, U2, V2, W2, n, m, p):
    """Add two decompositions for same (n,m,p). Rank adds."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    return U, V, W


def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 6

    # Strategy: Split n=4 into 2+2, p=6 into 3+3
    # C = A * B, A is 4x5, B is 5x6
    # Split rows of A into top-2 and bottom-2
    # Split cols of B into left-3 and right-3
    # C = [A1; A2] * [B1, B2] = [[A1*B1, A1*B2], [A2*B1, A2*B2]]
    #
    # Block (1,1): A1 (2x5) * B1 (5x3) -> 2x3
    # Block (1,2): A1 (2x5) * B2 (5x3) -> 2x3
    # Block (2,1): A2 (2x5) * B1 (5x3) -> 2x3
    # Block (2,2): A2 (2x5) * B2 (5x3) -> 2x3
    #
    # Each 2x5 * 5x3 can be further decomposed.
    # 2x5 * 5x3: split 5 into 2+3
    # = [A1a, A1b] * [B1a; B1b] where A1a is 2x2, A1b is 2x3, B1a is 2x3, B1b is 3x3
    # C = A1a*B1a + A1b*B1b
    #
    # A1a*B1a: 2x2 * 2x3 -> rank? 
    #   2x2 * 2x3: split 3 into 1+2
    #   = A1a*[B1a1, B1a2] where B1a1 is 2x1, B1a2 is 2x2
    #   = A1a*B1a1 + A1a*B1a2
    #   A1a*B1a1: 2x2 * 2x1 -> rank 4 (schoolbook) or can do better?
    #   Actually 2x2 * 2x1: C = A*B, A is 2x2, B is 2x1. 
    #   This is just 2x2 matrix times 2x1 vector. Rank = 4? No.
    #   T_{i,j,k} where k only has 1 value. T_{i,j,0} = 1.
    #   This is rank = n*m = 4? No, rank = m*p = 2? 
    #   Let me think: C_i = sum_j A_{ij} B_{j,0}
    #   T_{i,j,0} = 1 for all i,j.
    #   Decomposition: r=2, U_{i,j,0}=1, U_{i,j,1}=0? No.
    #   r=1: U_{i,j,0}=1, V_{j,0,0}=1, W_{0,i,0}=1. Then sum = 1*1*1 = 1. Yes! Rank 1.
    #   Wait: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}. For p=1, k=0.
    #   U_{i,r} V_{j,r} W_{0,r} = 1 for all i,j.
    #   r=1: U_{i,0}=1, V_{j,0}=1, W_{0,0}=1. Works. Rank 1.
    #
    # A1a*B1a2: 2x2 * 2x2 -> Strassen rank 7
    #
    # A1b*B1b: 2x3 * 3x3 -> 
    #   Split 3 (cols of B1b) into 1+2
    #   = A1b*[B1b1, B1b2] where B1b1 is 3x1, B1b2 is 3x2
    #   = A1b*B1b1 + A1b*B1b2
    #   A1b*B1b1: 2x3 * 3x1 -> rank? T_{i,j,k=0} = 1 for all i in [0,2), j in [0,3).
    #   r=1: U=1, V=1, W=1. Rank 1.
    #   A1b*B1b2: 2x3 * 3x2 -> 
    #     Split 3 (rows of B1b2 / cols of A1b) into 1+2
    #     = [A1b_top; A1b_bot] * [B1b2_top; B1b2_bot]
    #     = A1b_top*B1b2_top + A1b_bot*B1b2_bot
    #     A1b_top: 2x1, B1b2_top: 1x2 -> 2x1 * 1x2 = 2x2. Rank 4? 
    #       T_{i,j,k} for 2x1*1x2: i in [0,2), j in [0,1), k in [0,2)
    #       T_{i,0,k} = 1 for all i,k. Rank = 2 (one per k? or per i?)
    #       r=2: r=0: U_{i,0,r0}=1, V_{0,0,r0}=1, W_{k,i,r0}=1. Sum = 1. But we need sum over r.
    #       Actually T_{i,0,k} = 1. Set r=1: U=1, V=1, W=1. Rank 1.
    #     A1b_bot: 2x2, B1b2_bot: 2x2 -> Strassen rank 7
    #
    # This is getting complex. Let me just use the Kronecker approach for 4x5x6.
    # 4x5x6 = (2x2) * (2x? ) ... 
    # 
    # Alternative: 4x5x6, use 2x2x2 Strassen blocks.
    # Split 4=2+2, 5=2+3, 6=3+3
    # This gives 2*2*2 = 8 blocks, each being a product of sub-matrices.
    # 
    # Let me try a direct construction with known ranks:
    # 2x2x2: rank 7 (Strassen)
    # 2x2x3: 
    #   2x2 * 2x3: split 3 into 1+2
    #   = 2x2*2x1 (rank 1) + 2x2*2x2 (rank 7) = rank 8
    # 2x3x3:
    #   2x3 * 3x3: split 3 into 1+2
    #   = 2x3*3x1 (rank 1) + 2x3*3x2
    #   2x3*3x2: split 3 into 1+2
    #   = 2x1*1x2 (rank 1) + 2x2*2x2 (rank 7) = rank 8
    #   So 2x3x3 = 1 + 8 = 9
    # 3x2x3:
    #   3x2 * 2x3: split 3 into 1+2
    #   = 3x2*2x1 (rank 1) + 3x2*2x2
    #   3x2*2x2: split 3 into 1+2
    #   = 1x2*2x2 (rank 4) + 2x2*2x2 (rank 7) = rank 11
    #   So 3x2x3 = 1 + 11 = 12
    # 3x3x3:
    #   3x3 * 3x3: rank 21 (best known) or 23 (Landsberg)
    #   Use 21 for now? We don't have an explicit 21.
    #   Use schoolbook 27 or block: split 3=1+2
    #   = 1x3*3x1 (1) + 1x3*3x2 (6) + 2x3*3x1 (6) + 2x3*3x2 (8) = 21
    #   Wait: 1x3*3x2: 1x3 * 3x2. T_{0,j,k} = 1. Rank = 3? 
    #     r=3: one per j. U_{0,j,j}=1, V_{j,k,j}=1, W_{k,0,j}=1. Yes rank 3.
    #   2x3*3x1: rank 1 (as shown above for any m: 2xm*x1 rank 1)
    #   2x3*3x2: rank 8 as computed
    #   So 3x3x3 = 1 + 3 + 1 + 8 = 13? That can't be right, it's too low.
    #   
    # I think I'm making errors. Let me just use the Kronecker product of 2x2x2 and 2x3x3 or similar.
    #
    # Let me use: 4x5x6 = (2x2x2) kron (2x3x3) but that gives 4x6x6, not 4x5x6.
    #
    # Let me just implement a clean block decomposition.
    # Split n=4 into 2+2, m=5 into 2+3, p=6 into 3+3
    # 8 blocks:
    # (2x2)*(2x3): rank 8
    # (2x2)*(3x3): rank 9
    # (2x3)*(2x3): rank ? 2x3*2x3... wait m is shared.
    # 
    # Actually for matrix multiplication A(n x m) * B(m x p) = C(n x p),
    # when we split, the m dimension must be split consistently.
    # n=4=2+2, m=5=2+3, p=6=3+3
    # Blocks of C:
    # C11 = A11*B11 (2x2 * 2x3) -> 2x3
    # C12 = A11*B12 (2x2 * 3x3) -> 2x3
    # C21 = A21*B11 (2x3 * 2x3) -> 2x3
    # C22 = A21*B12 (2x3 * 3x3) -> 2x3
    # C31 = A31*B31 (2x2 * 2x3) -> 2x3  [A31 is bottom-left 2x2 of A]
    # C32 = A31*B32 (2x2 * 3x3) -> 2x3
    # C41 = A41*B31 (2x3 * 2x3) -> 2x3
    # C42 = A41*B32 (2x3 * 3x3) -> 2x3

    # Ranks:
    # 2x2 * 2x3: 8
    # 2x2 * 3x3: 9
    # 2x3 * 2x3: ? Let's compute. 2x3 * 3x... wait B is m x p.
    # A21 is 2x3 (rows 2-3 of A, cols 0-1 of A... no.
    # A is 4x5. Split rows 0-1 and 2-3. Split cols 0-1 and 2-4.
    # A11: 2x2, A12: 2x3, A21: 2x2, A22: 2x3
    # B is 5x6. Split rows 0-1 and 2-4. Split cols 0-2 and 3-5.
    # B11: 2x3, B12: 2x3, B21: 3x3, B22: 3x3
    # C = A*B:
    # C11 = A11*B11 + A12*B21 (2x2*2x3 + 2x3*3x3)
    # C12 = A11*B12 + A12*B22 (2x2*2x3 + 2x3*3x3)
    # C21 = A21*B11 + A22*B21 (2x2*2x3 + 2x3*3x3)
    # C22 = A21*B12 + A22*B22 (2x2*2x3 + 2x3*3x3)

    # So we need:
    # 2x2 * 2x3: rank 8
    # 2x3 * 3x3: rank 9
    # Total per C block: 8+9 = 17, 4 blocks: 68 total.
    # Plus the additive structure... actually each C block is a sum of two products,
    # so the total rank is sum of all individual product ranks:
    # 4 * (8 + 9) = 68

    # Let's verify 2x2*2x3 rank:
    # 2x2 * 2x3: split p=3 into 1+2
    # = 2x2*2x1 (rank 1) + 2x2*2x2 (rank 7) = 8. Correct.
    # 2x3 * 3x3: split p=3 into 1+2
    # = 2x3*3x1 (rank 1) + 2x3*3x2
    # 2x3*3x2: split m=3 into 1+2
    # = 2x1*1x2 (rank 1) + 2x2*2x2 (rank 7) = 8
    # So 2x3*3x3 = 1 + 8 = 9. Correct.
    # Total: 4 * (8 + 9) = 68.

    # Let's construct this.
    U = np.zeros((n * m, 68))
    V = np.zeros((m * p, 68))
    W = np.zeros((p * n, 68))

    col = 0
    for ci in range(2):  # C row block: 0=top, 1=bottom
        for ck in range(2):  # C col block: 0=left, 1=right
            # C_block(ci, ck) = A_block(ci, 0) * B_block(0, ck) + A_block(ci, 1) * B_block(1, ck)
            # Term 1: A(ci,0) * B(0,ck): 2x2 * 2x3, rank 8
            U1, V1, W1 = _small_decomp(2, 2, 3)
            # Map to global indices
            # A(ci,0): rows ci*2 to ci*2+1, cols 0 to 1
            # B(0,ck): rows 0 to 1, cols ck*3 to ck*3+2
            # C(ci,ck): rows ci*2 to ci*2+1, cols ck*3 to ck*3+2
            for r in range(8):
                for i1 in range(2):
                    for j1 in range(2):
                        if U1[i1 * 2 + j1, r] != 0:
                            gi = ci * 2 + i1
                            gj = j1  # cols 0-1 of A
                            U[gi * m + gj, col + r] = U1[i1 * 2 + j1, r]
                for j1 in range(2):
                    for k1 in range(3):
                        if V1[j1 * 3 + k1, r] != 0:
                            gj = j1  # rows 0-1 of B
                            gk = ck * 3 + k1
                            V[gj * p + gk, col + r] = V1[j1 * 3 + k1, r]
                for k1 in range(3):
                    for i1 in range(2):
                        if W1[k1 * 2 + i1, r] != 0:
                            gk = ck * 3 + k1
                            gi = ci * 2 + i1
                            W[gk * n + gi, col + r] = W1[k1 * 2 + i1, r]
            col += 8

            # Term 2: A(ci,1) * B(1,ck): 2x3 * 3x3, rank 9
            U2, V2, W2 = _small_decomp(2, 3, 3)
            for r in range(9):
                for i1 in range(2):
                    for j1 in range(3):
                        if U2[i1 * 3 + j1, r] != 0:
                            gi = ci * 2 + i1
                            gj = 2 + j1  # cols 2-4 of A
                            U[gi * m + gj, col + r] = U2[i1 * 3 + j1, r]
                for j1 in range(3):
                    for k1 in range(3):
                        if V2[j1 * 3 + k1, r] != 0:
                            gj = 2 + j1  # rows 2-4 of B
                            gk = ck * 3 + k1
                            V[gj * p + gk, col + r] = V2[j1 * 3 + k1, r]
                for k1 in range(3):
                    for i1 in range(2):
                        if W2[k1 * 2 + i1, r] != 0:
                            gk = ck * 3 + k1
                            gi = ci * 2 + i1
                            W[gk * n + gi, col + r] = W2[k1 * 2 + i1, r]
            col += 9

    # Verify
    if time.time() - start > DEADLINE_S:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Quick verification with random matrices
    rng = np.random.RandomState(42)
    A = rng.rand(n, m)
    B = rng.rand(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(68):
        C_computed += U[:, r] @ V[:, r] * W[:, r]  # This is wrong indexing
        # Actually C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let me just do a spot check
    break

    # The verification loop above is incomplete. Let's just return.
    # For safety, let me redo verification properly.
    C_computed = np.zeros((n, p))
    for r in range(68):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_computed[i, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    # This is O(68*4*5*6) = 8160 ops, fine.
    # But wait, this should equal A@B. Let me check with a simple test.
    # Actually the tensor decomposition T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # should satisfy sum_j T_{i,j,k} * A[i,j] * B[j,k] ... no.
    # The decomposition is for the multiplication tensor:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # T[i,j,k] = 1
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # So C[i,k] = sum_j A[i,j] B[j,k] * sum_r ... no.
    # Actually C[i,k] = sum_{j} A[i,j] B[j,k], and we want to compute this using the factors.
    # The standard identity: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * ... no.
    # It's: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * ... 
    # No, the factors are such that:
    # sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) (V... ) 
    # I think the correct verification is:
    # For each r, compute M_r where M_r[i,j] = U[i,j,r], N_r[j,k] = V[j,k,r], P_r[k,i] = W[k,i,r]
    # Then C = sum_r (A * M_r) * N_r * P_r? No.
    # 
    # The tensor decomposition T = sum_r U_r kron V_r kron W_r
    # means T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # And matrix multiplication uses T[i,j,k]=1, so:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]
    # = sum_r W[k,i,r] sum_j (U[i,j,r] A[i,j]) (V[j,k,r] B[j,k])
    # 
    # So to verify: for random A, B, compute C_expected = A@B
    # and C_decomp[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # This is O(rank * n * m + rank * m * p + rank * n * p) per verification, fine.

    C_decomp = np.zeros((n, p))
    for r in range(68):
        # Compute intermediate
        tmp = np.zeros((n, p))
        for i in range(n):
            s1 = 0
            for j in range(m):
                s1 += U[i * m + j, r] * A[i, j]
            for k in range(p):
                s2 = 0
                for j in range(m):
                    s2 += V[j * p + k, r] * B[j, k]
                tmp[i, k] = W[k * n + i, r] * s1 * s2
        C_decomp += tmp

    if not np.allclose(C_expected, C_decomp, atol=1e-10):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
