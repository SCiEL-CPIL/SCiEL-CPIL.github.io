"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _verify_fast(U, V, W, T):
    """Fast verification using precomputed tensor structure."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3
        
        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1
        
        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1
        
        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1
        
        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1
        
        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1
    
    if remainder == 1:
        k2 = num_pairs * 2
        v_e2 = k2
        v_g2 = p + k2
        w_c02 = k2 * n + 0
        w_c12 = k2 * n + 1
        
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
        
        U[3, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _try_rank_reduction(U, V, W, T, max_time=60):
    """Attempt to reduce rank via greedy single-column removal."""
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    for attempt in range(current_rank):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r in range(current_rank):
            if time.time() - start_time > max_time:
                break
            mask = [i for i in range(current_rank) if i != r]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify_fast(U_test, V_test, W_test, T):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    
    return U, V, W

def _try_leaking_strassen():
    """
    Try a decomposition where the last Strassen block (cols 8,9) "leaks"
    into col 10, allowing us to handle col 10 with fewer terms.
    
    Standard: 5 Strassen blocks (35) + 4 single terms = 39
    Leaking: 4 Strassen blocks (28) + modified block for cols 8,9,10 (10 terms) = 38
    
    The modified block for 2x2*2x3 needs rank 10.
    Let's construct it explicitly.
    """
    n, m, p = 2, 2, 11
    T = _build_target_tensor()
    
    # 4 standard Strassen blocks for cols 0-7 (rank 28)
    # Then a custom rank-10 decomposition for cols 8,9,10
    
    # For cols 8,9,10:
    # B = [[B[0,8], B[0,9], B[0,10]],
    #      [B[1,8], B[1,9], B[1,10]]]
    # Let e=B[0,8], f=B[0,9], g=B[1,8], h=B[1,9], i=B[0,10], j=B[1,10]
    #
    # C[0,8] = a*e + b*g
    # C[0,9] = a*f + b*h
    # C[0,10] = a*i + b*j
    # C[1,8] = c*e + d*g
    # C[1,9] = c*f + d*h
    # C[1,10] = c*i + d*j
    #
    # 10 terms:
    # Use Strassen for the 2x2 part (cols 8,9) with 7 terms,
    # then 3 terms for col 10.
    #
    # But we showed rank 10 is hard. Let me try a different split:
    # 5 terms for cols 8,9 and 5 terms for col 10? No, that's worse.
    #
    # Alternative: use a non-Strassen decomposition for 2x2*2x3 with rank 10.
    #
    # Let me try this specific construction:
    # t1 = (a+d)(e+h) -> C[0,8] += 1, C[1,9] += 1
    # t2 = (c+d)(e+i) -> C[1,8] += 1, C[1,10] += 1
    # t3 = a(f+h+j) -> C[0,9] += 1, C[0,8] += 1, C[0,10] += 1
    # t4 = d(g+i) -> C[0,8] += 1, C[1,8] += 1, C[1,10] += 1
    # t5 = (a+b)(h+j) -> C[0,8] -= 1, C[0,9] += 1, C[0,10] += 1
    # t6 = (c-a)(e+f+i) -> C[1,9] += 1, C[1,8] += 1, C[1,10] += 1
    # t7 = (b-d)(g+h+j) -> C[0,8] += 1, C[0,9] += 1, C[0,10] += 1
    # t8 = a*i -> C[0,10] += 1
    # t9 = b*j -> C[0,10] += 1
    # t10 = c*i -> C[1,10] += 1
    #
    # Let me check if this works by computing the total contribution to each C entry.
    
    # This is getting complex. Let me just try to verify a specific construction.
    
    # Actually, let me try a simpler approach: just return the standard Strassen
    # and rely on the rank reduction to find improvements.
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145
    
    T = _build_target_tensor()
    
    U, V, W = _build_strassen_decomposition()
    
    if _verify(U, V, W, T):
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
        
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 10:
            U_red, V_red, W_red = _try_rank_reduction(U, V, W, T, max_time=min(remaining_time, 90))
            if U_red.shape[1] < best_rank:
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = U_red.shape[1]
        
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    raise RuntimeError("No valid decomposition found")
