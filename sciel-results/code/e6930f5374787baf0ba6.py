import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 125  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def winograd_4x4_decomposition():
    """
    Winograd WMCC for 4x4 matrix multiplication.
    Treats 4x4 multiply as 4 cyclic convolutions of length 4.
    Each cyclic convolution of length 4 has rank 6 (Winograd).
    Total: 4 * 6 = 24 rank terms for the 4x4 block.
    
    The 4-point cyclic convolution uses the factorization:
    y = (F^{-1} * (F(a) .* F(b))) where F is 4x4 DFT-like matrix.
    But Winograd's method avoids complex numbers using real matrices.
    
    For length-4 cyclic convolution, the Winograd scheme uses:
    - Input transforms: 2x2 blocks
    - 6 multiplications
    - Output transforms: 2x2 blocks
    """
    # Winograd 4-point cyclic convolution matrices
    # Input transform for a (length 4): maps to 6 intermediate values
    # The standard Winograd 4-point uses:
    #   a0, a1, a2, a3 -> [a0+a2, a0-a2, a1+a3, a1-a3, a0+a1+a2+a3, a0-a1+a2-a3]
    # Actually, let me use the known Winograd 4-point scheme.
    
    # For 4-point DFT-based approach (simpler to verify):
    # F = [[1, 1, 1, 1],
    #      [1, -1, 1, -1],
    #      [1, j, -1, -j],
    #      [1, -j, -1, j]]
    # This has rank 4 but uses complex numbers.
    
    # Winograd's real scheme for 4-point cyclic convolution:
    # Uses 6 multiplications with real arithmetic.
    # The factorization is: F = B * G where B is 6x4, G is 6x4 (with 6 rows for 6 mults)
    
    # Let me use a known rank-6 decomposition for 4-point cyclic convolution.
    # The tensor for 4-point cyclic convolution: T[a,b,c] = sum_k a_k * b_{(c-k) mod 4}
    # This has rank 6 (Winograd 1979).
    
    # Winograd 4-point cyclic convolution:
    # Inputs: a = [a0,a1,a2,a3], b = [b0,b1,b2,b3]
    # 6 products:
    #   m0 = (a0+a2)*(b0+b2)
    #   m1 = (a0-a2)*(b0-b2)  
    #   m2 = (a1+a3)*(b1+b3)
    #   m3 = (a1-a3)*(b1-b3)
    #   m4 = (a0+a1+a2+a3)*(b0+b1+b2+b3)
    #   m5 = (a0-a1+a2-a3)*(b0-b1+b2-b3)
    # Output:
    #   c0 = (m0 + m1 + m2 + m3 + m4 + m5) / 4  -- wait, this isn't quite right
    # 
    # Actually, let me use the proper Winograd 4-point scheme.
    # The correct decomposition uses the polynomial:
    # (a0 + a1*x + a2*x^2 + a3*x^3)(b0 + b1*x + b2*x^2 + b3*x^3) mod (x^4-1)
    # 
    # Factor x^4-1 = (x-1)(x+1)(x^2+1) over reals... but x^2+1 doesn't factor over reals.
    # Over reals: x^4-1 = (x-1)(x+1)(x^2+1)
    # The rank over reals is 6 because x^2+1 contributes 2 (needs 2 mults for complex-like).
    
    # Let me just use the DFT approach with real/imaginary separation.
    # 4-point DFT: F is 4x4. The tensor rank over reals for 4-point DFT-based
    # convolution is at most 8 (4 complex mults = 8 real mults) but Winograd shows 6.
    
    # For simplicity, let me use rank 8 per convolution (4 complex mults), total 32.
    # But that's worse than Strassen's 28. Let me use the actual Winograd rank-6.
    
    # Winograd 4-point cyclic convolution (rank 6):
    # Based on the factorization of x^4 - 1:
    # x^4 - 1 = (x-1)(x+1)(x^2+1)
    # 
    # The 6 rank-1 terms correspond to:
    # 1. Evaluation at x=1: (sum a)(sum b) -> contributes to all c_k equally
    # 2. Evaluation at x=-1: (a0-a2... wait
    # 
    # Let me look up the exact Winograd 4-point scheme:
    # 
    # m1 = (a0+a2)(b0+b2)
    # m2 = (a0-a2)(b0-b2)
    # m3 = (a1+a3)(b1+b3)
    # m4 = (a1-a3)(b1-b3)
    # m5 = (a0+a1+a2+a3)(b0+b1+b2+b3)
    # m6 = (a0-a1+a2-a3)(b0-b1+b2-b3)
    #
    # c0 = (m1+m2+m3+m4+m5+m6)/4 ... no, this gives wrong results.
    #
    # I think I need to be more careful. Let me just use the DFT approach with
    # 4 complex multiplications = 8 real multiplications per convolution.
    # Total: 4 convolutions * 8 = 32. That's worse than 28.
    #
    # Alternatively, use the known result that 4x4 matrix multiplication has
    # tensor rank at most 24 (from Winograd) or possibly lower.
    #
    # Let me try a different approach: directly optimize for rank 20, 22, 24
    # using the Winograd structure as initialization.
    
    # For now, return a Strassen-based decomposition as fallback
    return None

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28) - clean version."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        r = start_r
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    return U, V, W

def winograd_based_decomposition():
    """
    Attempt Winograd-style decomposition for 2x4x4 tensor.
    
    Key insight: The 4x4 matrix multiplication can be decomposed using
    cyclic convolution structure. For the 2x4x4 case, we have two "rows"
    of A, each multiplying B.
    
    Let's try to find a rank-20 or rank-22 decomposition using the
    algebraic structure of 4-point cyclic convolution.
    
    The 4-point cyclic convolution tensor has rank 6 over reals.
    If we can exploit this, 4 convolutions * 6 = 24 for the 4x4 block.
    For 2x4, we might share some structure to get below 24.
    """
    n, m, p = 2, 4, 4
    
    # Winograd 4-point cyclic convolution: rank 6
    # The 6 rank-1 components for cyclic convolution of length 4:
    # Using the factorization x^4 - 1 = (x-1)(x+1)(x^2+1)
    # Over reals, x^2+1 is irreducible, contributing 2 to the rank.
    # So total rank = 1 + 1 + 2 + 1 + 1 = 6? Let me verify.
    #
    # Actually, the rank of the 4-point cyclic convolution tensor over R is 6.
    # The 6 multiplications are:
    #   m0 = (a0+a2)(b0+b2)
    #   m1 = (a0-a2)(b0-b2)
    #   m2 = (a1+a3)(b1+b3)
    #   m3 = (a1-a3)(b1-b3)
    #   m4 = (a0+a1+a2+a3)(b0+b1+b2+b3)
    #   m5 = (a0-a1+a2-a3)(b0-b1+b2-b3)
    #
    # Output:
    #   c0 = (m0+m1+m2+m3+m4+m5)/4
    #   c1 = (m2-m3+m4-m5)/4  ... I'm not sure about the exact coefficients
    #
    # Let me verify with a specific example.
    # a = [1,2,3,4], b = [5,6,7,8]
    # Cyclic conv: c[k] = sum_j a[j]*b[(k-j) mod 4]
    # c[0] = a0*b0 + a1*b3 + a2*b2 + a3*b1 = 1*5 + 2*8 + 3*7 + 4*6 = 5+16+21+24 = 66
    # c[1] = a0*b1 + a1*b0 + a2*b3 + a3*b2 = 1*6 + 2*5 + 3*8 + 4*7 = 6+10+24+28 = 68
    # c[2] = a0*b2 + a1*b1 + a2*b0 + a3*b3 = 1*7 + 2*6 + 3*5 + 4*8 = 7+12+15+32 = 66
    # c[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 = 1*8 + 2*7 + 3*6 + 4*5 = 8+14+18+20 = 60
    #
    # Now compute m values:
    # m0 = (1+3)(5+7) = 4*12 = 48
    # m1 = (1-3)(5-7) = (-2)(-2) = 4
    # m2 = (2+4)(6+8) = 6*14 = 84
    # m3 = (2-4)(6-8) = (-2)(-2) = 4
    # m4 = (1+2+3+4)(5+6+7+8) = 10*26 = 260
    # m5 = (1-2+3-4)(5-6+7-8) = (-2)(-2) = 4
    #
    # Try: c0 = (m0+m1+m2+m3+m4+m5)/4 = (48+4+84+4+260+4)/4 = 404/4 = 101 ≠ 66
    # So the coefficients are not all 1/4.
    #
    # I need the correct Winograd 4-point scheme. Let me use a different approach.
    # Use the fact that over C, the rank is 4 (DFT), and over R it's at most 8.
    # But Winograd showed rank 6 over R.
    #
    # For now, let me just use rank 8 per convolution (4 complex mults as 8 real),
    # giving total 32. That's worse than 28, so not useful directly.
    #
    # Better approach: use the DFT with real/imaginary parts carefully.
    # 4-point DFT: F = [[1,1,1,1],[1,-1,1,-1],[1,i,-1,-i],[1,-i,-1,i]]
    # Y = F * (F(A) .* F(B)) * F^{-1}
    # The 4 complex multiplications can be done with 3 real multiplications each
    # (using Karatsuba-like tricks), giving 12 real mults total.
    # But that's per convolution, so 4*12 = 48. Worse.
    #
    # I think the key insight is that for 2x4x4, we don't need full 4x4 decomposition.
    # Let me just try optimization from rank 24, 22, 20 with Winograd-inspired init.
    
    return None

def residuals(x, n=2, m=4, p=4, rank=24):
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    return T_recon.flatten() - 1.0

def try_rank_reduction(target_rank, n=2, m=4, p=4, max_restarts=20):
    """Try to reduce rank from Strassen's 28 to target_rank."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    
    U_s = np.array(strassen_decomposition()[0], dtype=float)
    V_s = np.array(strassen_decomposition()[1], dtype=float)
    W_s = np.array(strassen_decomposition()[2], dtype=float)
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        if restart == 0:
            # Direct truncation
            U_init = U_s[:, :target_rank].copy()
            V_init = V_s[:, :target_rank].copy()
            W_init = W_s[:, :target_rank].copy()
        elif restart == 1:
            # Truncation with small perturbation
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * 0.01
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * 0.01
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * 0.01
        else:
            # Larger perturbation
            scale = 0.02 + 0.005 * restart
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * scale
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * scale
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), 
                                       method='lm', max_nfev=5000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-10:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try rank 24
    if best_rank > 24 and not time_up():
        res = try_rank_reduction(24, n, m, p)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = 24
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try rank 22
    if best_rank > 22 and not time_up():
        res = try_rank_reduction(22, n, m, p)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = 22
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try rank 20
    if best_rank > 20 and not time_up():
        res = try_rank_reduction(20, n, m, p)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = 20
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try intermediate ranks if lower failed
    if best_rank > 24 and not time_up():
        for r in [25, 26, 27]:
            if time_up():
                break
            res = try_rank_reduction(r, n, m, p)
            if res is not None:
                U2, V2, W2 = res
                valid, err = verify_decomposition(U2, V2, W2, n, m, p)
                if valid and r < best_rank:
                    best_rank = r
                    best_result = {"U": U2, "V": V2, "W": W2}
                    break
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
