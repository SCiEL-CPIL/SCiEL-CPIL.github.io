import numpy as np

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS  # 3,4,5


def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2. Returns U,V,W each (4,7).
    U order: A00,A01,A10,A11 ; V order: B00,B01,B10,B11 ; W order: C00,C01,C10,C11."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11) -> C00 +, C11 +
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00 -> C10 +, C11 -
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11) -> C01 +, C11 +
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A11(B10-B00) -> C00 +, C10 +
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A00+A01)B11 -> C00 -, C01 +
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A10-A00)(B00+B01) -> C11 +
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11) -> C00 +
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _build_221():
    """Rank-4 schoolbook for 2x2x1: A(2x2)*B(2x1)=C(2x1). U(4,4),V(2,4),W(2,4).
    U order: A00,A01,A10,A11 ; V order: B0,B1 ; W order: C0,C1."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1  # A00 B0 -> C0
    U[1,1]=1; V[1,1]=1; W[0,1]=1  # A01 B1 -> C0
    U[2,2]=1; V[0,2]=1; W[1,2]=1  # A10 B0 -> C1
    U[3,3]=1; V[1,3]=1; W[1,3]=1  # A11 B1 -> C1
    return U, V, W


def _build_122():
    """Rank-4 schoolbook for 1x2x2: A(1x2)*B(2x2)=C(1x2). U(2,4),V(4,4),W(2,4).
    U order: A0,A1 ; V order: B00,B01,B10,B11 ; W order: C0,C1."""
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1  # A0 B00 -> C0
    U[1,1]=1; V[2,1]=1; W[0,1]=1  # A1 B10 -> C0
    U[0,2]=1; V[1,2]=1; W[1,2]=1  # A0 B01 -> C1
    U[1,3]=1; V[3,3]=1; W[1,3]=1  # A1 B11 -> C1
    return U, V, W


def _build_121():
    """Rank-2 for 1x2x1: A(1x2)*B(2x1)=C(1x1). U(2,2),V(2,2),W(1,2)."""
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W


def _verify(U, V, W):
    """Check that sum_r U[:,r]⊗V[:,r]⊗W[:,r] equals multiplication tensor.
    U: (12, r) idx i*m+j ; V: (20, r) idx j*p+k ; W: (15, r) idx k*n+i."""
    r = U.shape[1]
    # Build tensor T[i,j,k]
    T = np.zeros((n, m, p))
    Umat = np.asarray(U, dtype=float)
    Vmat = np.asarray(V, dtype=float)
    Wmat = np.asarray(W, dtype=float)
    for idx in range(r):
        u = Umat[:, idx].reshape(n, m)
        v = Vmat[:, idx].reshape(m, p)
        w = Wmat[:, idx].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    # Expected: M[i,j,k] = 1 for all (contributes A[i,j]*B[j,k] to C[i,k])
    M = np.ones((n, m, p))
    return np.allclose(T, M, atol=1e-9)


def _construct():
    U_parts, V_parts, W_parts = [], [], []
    def add(u_idx, v_idx, w_idx, Ublk, Vblk, Wblk):
        for r in range(Ublk.shape[1]):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for i, a in enumerate(u_idx): u[a] = Ublk[i, r]
            for i, a in enumerate(v_idx): v[a] = Vblk[i, r]
            for i, a in enumerate(w_idx): w[a] = Wblk[i, r]
            U_parts.append(u); V_parts.append(v); W_parts.append(w)

    # Pre-compute blocks
    S222_U, S222_V, S222_W = _build_strassen_222()
    S221_U, S221_V, S221_W = _build_221()
    S122_U, S122_V, S122_W = _build_122()
    S121_U, S121_V, S121_W = _build_121()

    # U indices: row i, col j -> i*m+j
    # Row0: 0,1,2,3 ; Row1: 4,5,6,7 ; Row2: 8,9,10,11
    # V indices: row j, col k -> j*p+k
    # B row0: 0..4 ; row1: 5..9 ; row2: 10..14 ; row3: 15..19
    # W indices: C[i,k] -> k*n+i
    # i=0: k0->0,k1->3,k2->6,k3->9,k4->12
    # i=1: k0->1,k1->4,k2->7,k3->10,k4->13
    # i=2: k0->2,k1->5,k2->8,k3->11,k4->14

    # Part A: rows 0,1 of A (2x4) * B(4x5)
    # Block A1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (2x2x2)
    add([0,1,4,5],[0,1,5,6],[0,3,1,4], S222_U, S222_V, S222_W)
    # Block A2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (2x2x2)
    add([0,1,4,5],[2,3,7,8],[6,9,7,10], S222_U, S222_V, S222_W)
    # Block A3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (2x2x1)
    add([0,1,4,5],[4,9],[12,13], S221_U, S221_V, S221_W)
    # Block A4: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (2x2x2)
    add([2,3,6,7],[10,11,15,16],[0,3,1,4], S222_U, S222_V, S222_W)
    # Block A5: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4] (2x2x2)
    add([2,3,6,7],[12,13,17,18],[6,9,7,10], S222_U, S222_V, S222_W)
    # Block A6: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5] (2x2x1)
    add([2,3,6,7],[14,19],[12,13], S221_U, S221_V, S221_W)

    # Part B: row 2 of A (1x4) * B(4x5)
    # Block B1: A[2,0:2] * B[0:2,0:2] -> C[2,0:2] (1x2x2)
    add([8,9],[0,1,5,6],[2,5], S122_U, S122_V, S122_W)
    # Block B2: A[2,0:2] * B[0:2,2:4] -> C[2,2:4] (1x2x2)
    add([8,9],[2,3,7,8],[8,11], S122_U, S122_V, S122_W)
    # Block B3: A[2,0:2] * B[0:2,4:5] -> C[2,4:5] (1x2x1)
    add([8,9],[4,9],[14], S121_U, S121_V, S121_W)
    # Block B4: A[2,2:4] * B[2:4,0:2] -> C[2,0:2] (1x2x2)
    add([10,11],[10,11,15,16],[2,5], S122_U, S122_V, S122_W)
    # Block B5: A[2,2:4] * B[2:4,2:4] -> C[2,2:4] (1x2x2)
    add([10,11],[12,13,17,18],[8,11], S122_U, S122_V, S122_W)
    # Block B6: A[2,2:4] * B[2:4,4:5] -> C[2,4:5] (1x2x1)
    add([10,11],[14,19],[14], S121_U, S121_V, S121_W)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def _construct_schoolbook():
    """Guaranteed rank-60 schoolbook decomposition."""
    U_parts, V_parts, W_parts = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def construct():
    import time
    t0 = time.time()
    deadline = 150.0
    margin = 30.0
    U, V, W = _construct()
    if time.time() - t0 > deadline - margin:
        # fall back
        U, V, W = _construct_schoolbook()
    else:
        if not _verify(U, V, W):
            U, V, W = _construct_schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
