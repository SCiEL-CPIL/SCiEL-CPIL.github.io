"""Matrix multiplication 4x4x6 - Kronecker structured decomposition with dependency elimination."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """Return (U, V, W) each of shape (4, 7) for the 2x2x2 Strassen decomposition.
    U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # M1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    # M3 = a00(b01-b11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    # M4 = a11(b10-b00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1
    # M5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    # M6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    # M7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    return U, V, W

def _strassen_2x2x3():
    """Return (U, V, W) for 2x2x3: Strassen for cols 0,1 + schoolbook for col 2.
    U shape (4, 11), V shape (6, 11), W shape (6, 11).
    U[i*2+j], V[j*3+k], W[k*2+i]."""
    U1, V1, W1 = _strassen_2x2x2()
    R = 11
    U = np.zeros((4, R), dtype=int)
    V = np.zeros((6, R), dtype=int)
    W = np.zeros((6, R), dtype=int)
    for c in range(7):
        U[:, c] = U1[:, c]
        # Map V1[j*2+k] to V[j*3+k]
        for j in range(2):
            for k in range(2):
                V[j*3+k, c] = V1[j*2+k, c]
        # Map W1[k*2+i] to W[k*2+i] (same indexing for first 2 cols)
        for k in range(2):
            for i in range(2):
                W[k*2+i, c] = W1[k*2+i, c]
    # Schoolbook for column 2 (4 terms)
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # Term: U[i*2+j, r] * V[j*3+2, r] * W[2*2+i, r] = 1
    U[0,7]=1; V[0*3+2,7]=1; W[2*2+0,7]=1   # i=0,j=0: U[0], V[2], W[4]
    U[1,8]=1; V[1*3+2,8]=1; W[2*2+0,8]=1   # i=0,j=1: U[1], V[5], W[4]
    U[2,9]=1; V[0*3+2,9]=1; W[2*2+1,9]=1   # i=1,j=0: U[2], V[2], W[5]
    U[3,10]=1; V[1*3+2,10]=1; W[2*2+1,10]=1 # i=1,j=1: U[3], V[5], W[5]
    return U, V, W

def _build_kronecker():
    """Build the full 4x4x6 decomposition via Kronecker product of 2x2x2 and 2x2x3.
    Returns (U, V, W) with shapes (16, 77), (24, 77), (24, 77).
    
    The 4x4x6 problem is a Kronecker product of 2x2x2 (for the 2x2 block structure
    in both A and B's first dimension pair) and 2x2x3 (for the remaining).
    
    A is 4x4, B is 4x6, C is 4x6.
    We view A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x3, C as 2x2 blocks of 2x3.
    
    Index conventions:
    U: shape (16, R), U[i*4+j, r] where i,j in 0..3
    V: shape (24, R), V[j*6+k, r] where j in 0..3, k in 0..5
    W: shape (24, R), W[k*4+i, r] where k in 0..5, i in 0..3
    
    Kronecker: i = i1*2 + i2, j = j1*2 + j2, k = k1*3 + k2
    where i1,j1 in {0,1} (outer 2x2), i2,j2 in {0,1} (inner 2x2), k1 in {0,1}, k2 in {0,1,2}
    
    U[i*4+j, r] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
    V[j*6+k, r] = V1[j1*2+k1, c1] * V2[j2*3+k2, c2]
    W[k*4+i, r] = W1[k1*2+i1, c1] * W2[k2*2+i2, c2]
    """
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _strassen_2x2x3()
    R = 7 * 11
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            # U: i = i1*2+i2, j = j1*2+j2 -> row = i*4+j = (i1*2+i2)*4 + (j1*2+j2)
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U1[i1*2+j1, c1]
                    if u1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2+j2, c2]
                            if u2 == 0:
                                continue
                            row = (i1*2+i2)*4 + (j1*2+j2)
                            U[row, c] = u1 * u2
            # V: j = j1*2+j2, k = k1*3+k2 -> row = j*6+k = (j1*2+j2)*6 + (k1*3+k2)
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V1[j1*2+k1, c1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v2 = V2[j2*3+k2, c2]
                            if v2 == 0:
                                continue
                            row = (j1*2+j2)*6 + (k1*3+k2)
                            V[row, c] = v1 * v2
            # W: k = k1*3+k2, i = i1*2+i2 -> row = k*4+i = (k1*3+k2)*4 + (i1*2+i2)
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W1[k1*2+i1, c1]
                    if w1 == 0:
                        continue
                    for k2 in range(3):
                        for i2 in range(2):
                            w2 = W2[k2*2+i2, c2]
                            if w2 == 0:
                                continue
                            row = (k1*3+k2)*4 + (i1*2+i2)
                            W[row, c] = w1 * w2
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=6):
    """Verify decomposition with random test."""
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha = np.sum(A * U[:, r].reshape(n, m))
        beta = np.sum(B * V[:, r].reshape(m, p))
        # W is (p*n, rank), W[k*n+i, r], so W[:, r].reshape(p, n) gives W[k,i]
        # C[i,k] should get contribution alpha * beta * W[k*n+i, r]
        # So C_computed[i,k] += alpha * beta * W[k*n+i, r]
        # W[:, r].reshape(p, n) has element [k,i] = W[k*n+i, r]
        C_computed += alpha * beta * W[:, r].reshape(p, n).T
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _verify_all_ones(U, V, W, n=4, m=4, p=6, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n_eq = n * m * p
    R = U.shape[1]
    # Vectorized check
    # M[eq, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # eq = i*m*p + j*p + k
    M = np.zeros((n_eq, R), dtype=float)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    M[eq, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                eq += 1
    target = np.ones(n_eq)
    return np.allclose(M @ np.ones(R), target, atol=tol)

def _try_reduce_rank(U, V, W, deadline):
    """Try to reduce rank by detecting linear dependencies among terms.
    Build the 96 x R constraint matrix and find minimum support solution."""
    n, m, p = 4, 4, 6
    n_eq = n * m * p
    R = U.shape[1]
    # Build constraint matrix M of shape (n_eq, R)
    M = np.zeros((n_eq, R), dtype=float)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    M[eq, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                eq += 1
    target = np.ones(n_eq)
    # Verify
    if not np.allclose(M @ np.ones(R), target, atol=1e-9):
        return U, V, W  # invalid
    
    # Use SVD to find the null space and identify redundant columns
    # First, find the rank of M
    rank_M = np.linalg.matrix_rank(M)
    
    # Try greedy removal: for each column, check if it's in the span of others
    active = list(range(R))
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for t in list(active):
            if len(active) <= 1:
                break
            remaining = [x for x in active if x != t]
            M_sub = M[:, remaining]
            # Check if target is in the column space of M_sub
            # i.e., does M_sub @ x = target have a solution?
            # Use least squares residual
            try:
                sol, residuals, rank_sub, sv = np.linalg.lstsq(M_sub, target, rcond=None)
                residual = np.linalg.norm(M_sub @ sol - target)
                if residual < 1e-8:
                    active = remaining
                    improved = True
                    break
            except np.linalg.LinAlgError:
                continue
    # Extract reduced matrices
    R_new = len(active)
    U_new = U[:, active]
    V_new = V[:, active]
    W_new = W[:, active]
    return U_new, V_new, W_new

def construct():
    deadline = time.time() + 140
    # Build Kronecker decomposition
    U, V, W = _build_kronecker()
    # Verify using tensor identity
    if not _verify_all_ones(U, V, W):
        # Fallback schoolbook
        R = 4*4*6
        U = np.zeros((16, R), dtype=int)
        V = np.zeros((24, R), dtype=int)
        W = np.zeros((24, R), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # Try rank reduction
    if time.time() < deadline - 10:
        U, V, W = _try_reduce_rank(U, V, W, deadline)
    # Final verification
    if not _verify_all_ones(U, V, W):
        R = 4*4*6
        U = np.zeros((16, R), dtype=int)
        V = np.zeros((24, R), dtype=int)
        W = np.zeros((24, R), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
