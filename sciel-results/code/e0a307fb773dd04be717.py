"""Rank-44 decomposition for 2x3 * 3x8 matrix multiplication via Strassen + outer product."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _check_time():
    if time.time() - _START_TIME > _TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")

def _strassen_2x2_decomp():
    """Rank-7 Strassen for 2x2 * 2x2.
    Returns U(4x7), V(4x7), W(4x7) for tensor T[a,b,d] with a=i*2+j, b=j*2+k, d=k*2+i.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a11+a22)(b11+b22)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1  # c11, c22
    
    # m2 = (a21+a22)b11
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1  # c21, -c22
    
    # m3 = a11(b12-b22)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1  # c12, c22
    
    # m4 = a22(b21-b11)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1  # c11, c21
    
    # m5 = (a11+a12)b22
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1  # -c11, c12
    
    # m6 = (a21-a11)(b11+b12)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1  # c22
    
    # m7 = (a12-a22)(b21+b22)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1  # c11
    
    return U, V, W

def _outer_2x1_1x2_decomp():
    """Rank-4 standard for 2x1 * 1x2 (outer product).
    A is 2x1, B is 1x2, C is 2x2. C[i,k] = A[i,0]*B[0,k].
    """
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    for i in range(2):
        for k in range(2):
            t = i * 2 + k
            U[i, t] = 1
            V[k, t] = 1
            W[k*2+i, t] = 1
    return U, V, W

def _build_2x3_3x2_decomp():
    """Build a rank-11 decomposition for 2x3 * 3x2 = 2x2.
    Split A(2x3) = [A_left(2x2), A_right(2x1)], B(3x2) = [B_top(2x2); B_bottom(1x2)].
    C = A_left * B_top + A_right * B_bottom.
    Rank = 7 (Strassen for 2x2*2x2) + 4 (standard for 2x1*1x2) = 11.
    
    Returns U(6x11), V(6x11), W(4x11) where:
    a = i*3+j (i∈{0,1}, j∈{0,1,2}), b = j*2+k (j∈{0,1,2}, k∈{0,1}), d = k*2+i (k∈{0,1}, i∈{0,1})
    """
    # Strassen part for A_left(2x2) * B_top(2x2):
    Us, Vs, Ws = _strassen_2x2_decomp()
    
    U_strassen = np.zeros((6, 7))
    V_strassen = np.zeros((6, 7))
    W_strassen = np.zeros((4, 7))
    
    for i in range(2):
        for j in range(2):
            a_str = i*2 + j
            a_glob = i*3 + j
            U_strassen[a_glob, :] = Us[a_str, :]
    
    for j in range(2):
        for k in range(2):
            b_str = j*2 + k
            b_glob = j*2 + k
            V_strassen[b_glob, :] = Vs[b_str, :]
    
    for k in range(2):
        for i in range(2):
            d_str = k*2 + i
            d_glob = k*2 + i
            W_strassen[d_glob, :] = Ws[d_str, :]
    
    # Outer product part for A_right(2x1) * B_bottom(1x2):
    Uo, Vo, Wo = _outer_2x1_1x2_decomp()
    
    U_outer = np.zeros((6, 4))
    V_outer = np.zeros((6, 4))
    W_outer = np.zeros((4, 4))
    
    for i in range(2):
        a_glob = i*3 + 2
        U_outer[a_glob, :] = Uo[i, :]
    
    for k in range(2):
        b_glob = 2*2 + k
        V_outer[b_glob, :] = Vo[k, :]
    
    W_outer = Wo
    
    # Combine
    U = np.hstack([U_strassen, U_outer])
    V = np.hstack([V_strassen, V_outer])
    W = np.hstack([W_strassen, W_outer])
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition for n x m * m x p."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(T_decomp, T, atol=tol, rtol=0)

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build the 2x3 * 3x2 sub-decomposition (rank 11)
    U_sub, V_sub, W_sub = _build_2x3_3x2_decomp()
    
    # Verify sub-decomposition
    assert _verify_decomposition(U_sub, V_sub, W_sub, 2, 3, 2), "Sub-decomposition verification failed"
    
    r_sub = U_sub.shape[1]  # 11
    
    # Construct full decomposition: 4 blocks
    p_sub = 2
    num_blocks = p // p_sub  # 4
    total_rank = num_blocks * r_sub  # 44
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for t in range(num_blocks):
        U_full[:, t*r_sub:(t+1)*r_sub] = U_sub
        
        for j in range(m):
            for k_prime in range(p_sub):
                b_global = j * p + (t * p_sub + k_prime)
                b_local = j * p_sub + k_prime
                V_full[b_global, t*r_sub:(t+1)*r_sub] = V_sub[b_local]
        
        for k_prime in range(p_sub):
            for i in range(n):
                d_global = (t * p_sub + k_prime) * n + i
                d_local = k_prime * n + i
                W_full[d_global, t*r_sub:(t+1)*r_sub] = W_sub[d_local]
    
    # Verify full decomposition
    assert _verify_decomposition(U_full, V_full, W_full, n, m, p), "Full decomposition verification failed"
    
    _check_time()
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
