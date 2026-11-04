import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and compare against the standard tensor M[i,j,k] = 1 if j==j else 0 (actually M[i,j,k] = delta_{j,?})
    # The matrix multiplication tensor: T[i,j,k] = 1 if the (i,k)-th entry of A*B depends on A[i,j]*B[j,k]
    # Actually the tensor is: sum_j A[i,j]*B[j,k] = C[i,k]
    # So the 3-tensor T[i,j,k] should equal 1 if k is free... no.
    # The tensor T is defined by T[i,j,k] = 1 if j==j (always 1 for valid indices) -- no.
    
    # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    # The rank-r decomposition means:
    # M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Compute T from the factorization
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]  # (n*m,)
        v = V[:, r]  # (m*p,)
        w = W[:, r]  # (p*n,)
        # Reshape
        u3 = u.reshape(n, m)
        v3 = v.reshape(m, p)
        w3 = w.reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u3, v3, w3)
    
    # Target: M[i,j,k] = 1 for all i,j,k
    M = np.ones((n, m, p))
    
    err = np.max(np.abs(T - M))
    return err < tol

def construct():
    """
    Returns a rank-98 exact tensor decomposition for 4x4x8 matrix multiplication.
    
    Uses two-level Strassen for 4x4x4 (rank 49) applied to two halves of the 8-wide output.
    Total rank: 98.
    """
    n, m, p = 4, 4, 8
    
    # --- Strassen 2x2x2 (rank 7) ---
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[0,3]=-1; V2[2,3]=1; W2[0,3]=1; W2[1,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    U2[0,5]=-1; U2[2,5]=1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    
    # --- Build 4x4x4 from two-level Strassen (rank 49) ---
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            for uo in range(4):
                uo_val = U2[uo, ro]
                if uo_val != 0:
                    Io = uo // 2; Jo = uo % 2
                    for ui in range(4):
                        ui_val = U2[ui, ri]
                        if ui_val != 0:
                            ip = ui // 2; jp = ui % 2
                            i = 2*Io + ip; j = 2*Jo + jp
                            U4[i*4+j, r] = uo_val * ui_val
            
            for vo in range(4):
                vo_val = V2[vo, ro]
                if vo_val != 0:
                    Jo = vo // 2; Ko = vo % 2
                    for vi in range(4):
                        vi_val = V2[vi, ri]
                        if vi_val != 0:
                            jp = vi // 2; kp = vi % 2
                            j = 2*Jo + jp; k = 2*Ko + kp
                            V4[j*4+k, r] = vo_val * vi_val
            
            for wo in range(4):
                wo_val = W2[wo, ro]
                if wo_val != 0:
                    Ko = wo // 2; Io = wo % 2
                    for wi in range(4):
                        wi_val = W2[wi, ri]
                        if wi_val != 0:
                            kp = wi // 2; ip = wi % 2
                            k = 2*Ko + kp; i = 2*Io + ip
                            W4[k*4+i, r] = wo_val * wi_val
    
    # --- Assemble 4x4x8 (rank 98) ---
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left half: B columns 0..3 -> C columns 0..3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
            W[k*4 + j, 0:49] = W4[k*4 + j, :]
    
    # Right half: B columns 4..7 -> C columns 4..7
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + 4 + k, 49:98] = V4[j*4 + k, :]
            W[(4+k)*4 + j, 49:98] = W4[k*4 + j, :]
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Tensor decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
