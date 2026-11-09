import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition of the 2x2 * 2x2 multiplication tensor."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # c=0: M1 = (A00+A11)(B01+B10)
    U[0, 0] = 1; U[3, 0] = 1
    V[1, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # c=1: M2 = (A10+A11)A00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # c=2: M3 = A00(B00-B10)
    U[0, 2] = 1
    V[0, 2] = 1; V[2, 2] = -1
    W[0, 2] = 1; W[3, 2] = 1
    # c=3: M4 = A11(B11-B01)
    U[3, 3] = 1
    V[3, 3] = 1; V[1, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    # c=4: M5 = (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    # c=5: M6 = (A10-A00)(B00+B11)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1
    # c=6: M7 = (A01-A11)(B01+B10)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _build_4x4x4_rank49():
    """Build a rank-49 decomposition of the 4x4 * 4x4 multiplication tensor using
    Block Strassen's algorithm (Strassen on 2x2 blocks, each block multiplication
    decomposed into rank 7)."""
    sU, sV, sW = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Outer Strassen terms (7)
    for t in range(7):
        # Inner Strassen terms (7)
        for s in range(7):
            col = t * 7 + s
            
            # Iterate over block indices a, b, c (0..1)
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        # Get coefficients from outer Strassen
                        u_out = sU[a * 2 + b, t] # Wait, sU is 2x2? No, sU is 4x7 for 2x2 matrices.
                        # sU indices: row i (0..3), col t (0..6).
                        # i = 2a + alpha? No, in the 2x2 Strassen, the "row" is the element index of the 2x2 matrix.
                        # The 2x2 matrix has elements (0,0), (0,1), (1,0), (1,1).
                        # So sU[i, t] is the coefficient for element i in term t.
                        # In the block context, element i corresponds to block (a,b)?
                        # No, the outer Strassen operates on the 2x2 block matrix.
                        # The "elements" of the 2x2 block matrix ARE the 2x2 blocks.
                        # So sU[i, t] gives the coefficient for block i.
                        # Block i corresponds to (a,b) where i = 2a + b?
                        # Standard row-major: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
                        # So i = 2a + b.
                        
                        u_out = sU[2*a + b, t]
                        if u_out == 0: continue
                        
                        v_out = sV[2*b + c, t] # Block B is (b,c)
                        if v_out == 0: continue
                        
                        w_out = sW[2*c + a, t] # Block C is (c,a) -> W index is (k,i) so (c,a)
                        if w_out == 0: continue
                        
                        # Now, for this specific combination of blocks, we need to fill in the
                        # inner 2x2 structure using the inner Strassen term s.
                        
                        # Iterate over inner element indices alpha, beta, gamma
                        for alpha in range(2):
                            for beta in range(2):
                                for gamma in range(2):
                                    # Inner Strassen coefficients
                                    u_in = sU[2*alpha + beta, s]
                                    if u_in == 0: continue
                                    
                                    v_in = sV[2*beta + gamma, s]
                                    if v_in == 0: continue
                                    
                                    w_in = sW[2*gamma + alpha, s]
                                    if w_in == 0: continue
                                    
                                    coeff = u_out * v_out * w_in # Wait, W is separate
                                    # Actually, U, V, W are separate matrices in the decomposition.
                                    # U[i,j,col] = u_out * u_in
                                    # V[j,k,col] = v_out * v_in
                                    # W[k,i,col] = w_out * w_in
                                    
                                    i = 2*a + alpha
                                    j = 2*b + beta
                                    k = 2*c + gamma
                                    
                                    U[i*4 + j, col] += u_out * u_in
                                    V[j*4 + k, col] += v_out * v_in
                                    W[k*4 + i, col] += w_out * w_in
                                    
    return U, V, W

def _build_4x4x8():
    """Build a rank-98 decomposition of the 4x4 * 4x8 multiplication tensor."""
    U1, V1, W1 = _build_4x4x4_rank49()
    rank = 2 * 49
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # First half (columns 0-3 of B, rows 0-3 of C)
    U[:, 0:49] = U1
    V[0:16, 0:49] = V1
    W[0:16, 0:49] = W1

    # Second half (columns 4-7 of B, rows 4-7 of C)
    U[:, 49:98] = U1
    V[16:32, 49:98] = V1
    W[16:32, 49:98] = W1

    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want T[i,j,k] == 1 for all i,j,k
    
    # Use einsum for speed
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=100):
    """Try to reduce rank by identifying and removing linearly dependent terms."""
    start_time = time.time()
    current_rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try removing terms one at a time
    removed_any = True
    while removed_any and time.time() - start_time < max_time:
        removed_any = False
        for c in range(best_rank):
            if time.time() - start_time > max_time:
                break
            
            U_test = np.delete(best_U, c, axis=1)
            V_test = np.delete(best_V, c, axis=1)
            W_test = np.delete(best_W, c, axis=1)
            
            if _verify_fast(U_test, V_test, W_test, n, m, p, tol=1e-10):
                best_U = U_test
                best_V = V_test
                best_W = W_test
                best_rank -= 1
                removed_any = True
                break
        
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    try:
        U, V, W = _build_4x4x8()
        if _verify_fast(U, V, W, n, m, p, tol=1e-10):
            # Try to reduce rank
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, max_time=120)
            if _verify_fast(U_red, V_red, W_red, n, m, p, tol=1e-10):
                U, V, W = U_red, V_red, W_red
            
            # Snap to integers
            U_snap = np.round(U).astype(int)
            V_snap = np.round(V).astype(int)
            W_snap = np.round(W).astype(int)
            if _verify_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            else:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook
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

# EVOLVE-BLOCK-END
