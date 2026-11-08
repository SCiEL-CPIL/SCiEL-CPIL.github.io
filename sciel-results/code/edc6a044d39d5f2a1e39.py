import numpy as np
import time
from itertools import product

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
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W


def _build_122():
    """Rank-4 schoolbook for 1x2x2: A(1x2)*B(2x2)=C(1x2). U(2,4),V(4,4),W(2,4).
    U order: A0,A1 ; V order: B00,B01,B10,B11 ; W order: C0,C1."""
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    U[0,2]=1; V[1,2]=1; W[1,2]=1
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    return U, V, W


def _build_121():
    """Rank-2 for 1x2x1: A(1x2)*B(2x1)=C(1x1). U(2,2),V(2,2),W(1,2)."""
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W


def _verify(U, V, W):
    """Check that sum_r U[:,r]⊗V[:,r]⊗W[:,r] equals multiplication tensor."""
    r = U.shape[1]
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
    M = np.ones((n, m, p))
    return np.allclose(T, M, atol=1e-9)


def _construct_block_decomposition():
    """Block decomposition using Strassen for 2x2x2 blocks."""
    U_parts, V_parts, W_parts = [], [], []
    def add(u_idx, v_idx, w_idx, Ublk, Vblk, Wblk):
        for r in range(Ublk.shape[1]):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for i, a in enumerate(u_idx): u[a] = Ublk[i, r]
            for i, a in enumerate(v_idx): v[a] = Vblk[i, r]
            for i, a in enumerate(w_idx): w[a] = Wblk[i, r]
            U_parts.append(u); V_parts.append(v); W_parts.append(w)

    # Part A: rows 0,1 of A (2x4) * B(4x5)
    Uu, Vv, Ww = _build_strassen_222()
    add([0,1,4,5],[0,1,5,6],[0,3,1,4], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([0,1,4,5],[2,3,7,8],[6,9,7,10], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_221()
    add([0,1,4,5],[4,9],[12,13], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([2,3,6,7],[10,11,15,16],[0,3,1,4], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([2,3,6,7],[12,13,17,18],[6,9,7,10], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_221()
    add([2,3,6,7],[14,19],[12,13], Uu, Vv, Ww)

    # Part B: row 2 of A (1x4) * B(4x5)
    Uu, Vv, Ww = _build_122()
    add([8,9],[0,1,5,6],[2,5], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([8,9],[2,3,7,8],[8,11], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_121()
    add([8,9],[4,9],[14], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([10,11],[10,11,15,16],[2,5], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([10,11],[12,13,17,18],[8,11], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_121()
    add([10,11],[14,19],[14], Uu, Vv, Ww)

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


def _try_compress(U, V, W, max_iter=5000, deadline=None):
    """
    Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
    If we have R terms, we look for a set of R' < R terms that span the same tensor.
    This is equivalent to finding a lower-rank decomposition.
    
    Strategy: 
    1. Flatten the tensor representation.
    2. Use a greedy algorithm to remove redundant terms.
    """
    R = U.shape[1]
    U_mat = np.asarray(U, dtype=float)
    V_mat = np.asarray(V, dtype=float)
    W_mat = np.asarray(W, dtype=float)
    
    # Each term r defines a rank-1 tensor T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We want to see if some terms are linear combinations of others.
    # This is hard to do directly. Instead, we can try to merge terms.
    
    # A simpler approach: check if any two terms are identical or negatives.
    # Or, use a randomized approach to find a lower rank decomposition.
    
    # Let's try a different approach: 
    # The tensor is T = sum_r U_r ⊗ V_r ⊗ W_r.
    # We can try to eliminate terms by checking if removing a term can be compensated by adjusting others.
    # This is essentially a sparse recovery problem.
    
    # For now, let's just return the original if we don't find a compression.
    # A more sophisticated approach would use alternating least squares or similar.
    
    # Let's try to merge identical terms.
    terms = {}
    for r in range(R):
        u = U_mat[:, r]
        v = V_mat[:, r]
        w = W_mat[:, r]
        # Create a canonical form for the term
        # We can use the fact that (alpha * u) ⊗ (beta * v) ⊗ (gamma * w) = (alpha*beta*gamma) * (u ⊗ v ⊗ w)
        # So we can normalize each term.
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        norm_w = np.linalg.norm(w)
        if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
            scale = norm_u * norm_v * norm_w
            u_norm = u / norm_u
            v_norm = v / norm_v
            w_norm = w / norm_w
            # Check sign consistency
            if u_norm[0] < 0:
                u_norm = -u_norm
                scale = -scale
            if v_norm[0] < 0:
                v_norm = -v_norm
                scale = -scale
            if w_norm[0] < 0:
                w_norm = -w_norm
                scale = -scale
            key = (np.round(u_norm, 6).tobytes(), np.round(v_norm, 6).tobytes(), np.round(w_norm, 6).tobytes())
            if key in terms:
                terms[key][0] += scale
                terms[key][1] = r
            else:
                terms[key] = [scale, r]
        else:
            # Handle zero terms
            key = (u.tobytes(), v.tobytes(), w.tobytes())
            if key in terms:
                terms[key][0] += 1
                terms[key][1] = r
            else:
                terms[key] = [1, r]
    
    # Reconstruct with merged terms
    U_new = []
    V_new = []
    W_new = []
    for scale, r in terms.values():
        if abs(scale) > 1e-10:
            U_new.append(U_mat[:, r] * scale)
            V_new.append(V_mat[:, r])
            W_new.append(W_mat[:, r])
    
    if len(U_new) < R:
        U_c = np.array(U_new).T
        V_c = np.array(V_new).T
        W_c = np.array(W_new).T
        if _verify(U_c, V_c, W_c):
            return U_c, V_c, W_c
    
    return None


def construct():
    import time
    t0 = time.time()
    deadline = 150.0
    margin = 30.0
    
    # First, get the block decomposition
    U, V, W = _construct_block_decomposition()
    
    # Try to compress
    if time.time() - t0 < deadline - margin:
        compressed = _try_compress(U, V, W)
        if compressed is not None:
            U, V, W = compressed
    
    # Verify and fall back if needed
    if time.time() - t0 < deadline - margin:
        if not _verify(U, V, W):
            U, V, W = _construct_schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
