import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R
    # V: 12 x R
    # W: 12 x R
    
    # Construct target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_based():
        """
        Strassen-based construction for 3x3 x 3x4.
        
        Split the 3x3 matrix into blocks:
        A = [[A11, A12, A13],
             [A21, A22, A23],
             [A31, A32, A33]]
        
        B = [[B11, B12, B13, B14],
             [B21, B22, B23, B24],
             [B31, B32, B33, B34]]
        
        We can use Strassen on 2x2 subblocks and handle the rest.
        
        Strategy: Group columns of B into pairs to use Strassen.
        Let's group B's 4 columns as (col0,col1), (col2,col3)
        
        For the 2x2 Strassen on the top-left 2x2 block of A with 
        corresponding 2x2 blocks of B:
        
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B12-B22)
        M4 = A22(B21-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        Then:
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        For the 3rd row/col, we need additional terms.
        
        Let me construct this more carefully.
        
        Actually, let's use a simpler approach: use known low-rank 
        constructions for small matrix multiplication.
        
        For 3x3 x 3x3, the best known is rank 23 (or similar).
        For 3x3 x 3x4, we can extend.
        
        Let me try a direct construction with rank 27 or so.
        """
        # Try a structured approach: use Strassen for 2x2 parts
        # and handle the 3rd dimension separately
        
        rank = 27  # Start with a guess, we'll refine
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # This is a placeholder - the real work is in the optimization
        # below. We'll use this as an initial guess structure.
        
        # Actually, let's just return None and rely on the optimizer
        return None

    def rationalize_matrix(M, tol=1e-6):
        rows = M.shape[0]
        cols = M.shape[1]
        result = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                found = False
                for d in range(1, 21):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    if abs(float(frac) - val) < tol:
                        result[i][j] = frac
                        found = True
                        break
                if not found:
                    return None
        return result

    def is_half_integer(frac):
        return (frac * 2).denominator == 1

    def verify(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_half_integer(val):
                        return False
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0, 1)
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                if j1 != j2:
                    for c in range(p*n):
                        k1 = c // n
                        i2 = c % n
                        s = Fraction(0, 1)
                        for r in range(rank):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if s != 0:
                            return False
        return True

    def compute_residual(U, V, W, T):
        T_calc = np.zeros_like(T)
        R = U.shape[1]
        for r in range(R):
            T_calc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(T_calc - T)

    def als_decompose(T, R, max_iters=80, tol=1e-10, init_type='random', seed=None):
        nm, mp, pn = n*m, m*p, p*n
        
        if seed is not None:
            np.random.seed(seed)
        
        if init_type == 'random':
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
        elif init_type == 'structured':
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            # Strassen-inspired initialization
            # Use 7 Strassen terms for 2x2 blocks, extended
            for r in range(R):
                i = r % n
                j = (r // n) % m
                k = (r // (n*m)) % p
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
            U += 0.1 * np.random.randn(nm, R)
            V += 0.1 * np.random.randn(mp, R)
            W += 0.1 * np.random.randn(pn, R)
        elif init_type == 'sparse_random':
            # Sparse random with half-integer values
            half_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            density = min(0.6, 15.0 / R)
            for r in range(R):
                u_nz = np.random.choice(nm, size=max(1, int(nm*density)), replace=False)
                v_nz = np.random.choice(mp, size=max(1, int(mp*density)), replace=False)
                w_nz = np.random.choice(pn, size=max(1, int(pn*density)), replace=False)
                for idx in u_nz:
                    U[idx, r] = np.random.choice(half_vals)
                for idx in v_nz:
                    V[idx, r] = np.random.choice(half_vals)
                for idx in w_nz:
                    W[idx, r] = np.random.choice(half_vals)
        else:
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
                
            for a in range(nm):
                A = np.einsum('br,cr->bc r', V, W).reshape(mp*pn, R)
                y = T[a].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
            
            for b in range(mp):
                A = np.einsum('ar,cr->ac r', U, W).reshape(nm*pn, R)
                y = T[:, b, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
            
            for c in range(pn):
                A = np.einsum('ar,br->ab r', U, V).reshape(nm*mp, R)
                y = T[:, :, c].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
            
            err = compute_residual(U, V, W, T)
            if err < tol:
                return U, V, W, err
            if abs(prev_err - err) < 1e-12 and it > 10:
                return U, V, W, err
            prev_err = err
                
        return U, V, W, err

    def gradient_polish(U, V, W, T, max_iters=200, tol=1e-12):
        nm, mp, pn = n*m, m*p, p*n
        R = U.shape[1]
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            T_calc = np.zeros((nm, mp, pn))
            for r in range(R):
                T_calc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            diff = T_calc - T
            
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            err = np.linalg.norm(diff)
            if err < tol:
                break
            
            if it % 20 == 0:
                lr = max(0.001, lr * 0.9)
        
        final_err = compute_residual(U, V, W, T)
        return U, V, W, final_err

    def snap_to_half_int(U, V, W):
        """Snap to nearest half-integers and verify."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Verify
        R = U_s.shape[1]
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            T_calc += np.einsum('a,b,c->abc', U_s[:, r], V_s[:, r], W_s[:, r])
        
        if np.allclose(T_calc, T, atol=1e-9):
            # Check half-integer
            if np.allclose(U_s * 2, np.round(U_s * 2), atol=1e-9) and \
               np.allclose(V_s * 2, np.round(V_s * 2), atol=1e-9) and \
               np.allclose(W_s * 2, np.round(W_s * 2), atol=1e-9):
                return U_s, V_s, W_s
        return None

    def try_rational_and_verify(U, V, W, tol=1e-5):
        U_frac = rationalize_matrix(U, tol=tol)
        V_frac = rationalize_matrix(V, tol=tol)
        W_frac = rationalize_matrix(W, tol=tol)
        if U_frac and V_frac and W_frac:
            if verify(U_frac, V_frac, W_frac):
                U_out = [[float(x) for x in row] for row in U_frac]
                V_out = [[float(x) for x in row] for row in V_frac]
                W_out = [[float(x) for x in row] for row in W_frac]
                return {"U": U_out, "V": V_out, "W": W_out}
        return None

    # Search for minimal R
    # Known results: 3x3x3 has rank at most 23, 3x3x4 likely around 27-30
    # Start from lower ranks and work up
    
    best_result = None
    
    # Try ranks from 20 to 36
    for R in range(20, 37):
        if time.time() > deadline - 20:
            break
        
        # Multiple restarts with different strategies
        seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]
        init_types = ['random', 'structured', 'sparse_random', 'random', 'structured',
                      'sparse_random', 'random', 'structured', 'sparse_random', 'random']
        
        for restart in range(min(len(seeds), max(3, 15 - (R - 20)))):
            if time.time() > deadline - 15:
                break
            
            seed = seeds[restart % len(seeds)]
            init_type = init_types[restart % len(init_types)]
            
            U, V, W, err = als_decompose(T, R, max_iters=40, tol=1e-10, 
                                          init_type=init_type, seed=seed)
            
            if err < 1e-6:
                U, V, W, err = gradient_polish(U, V, W, T, max_iters=80, tol=1e-12)
                
                if err < 1e-8:
                    # Try snap to half-integers
                    snapped = snap_to_half_int(U, V, W)
                    if snapped is not None:
                        U_s, V_s, W_s = snapped
                        result = try_rational_and_verify(U_s, V_s, W_s, tol=1e-5)
                        if result:
                            return result
                    
                    # Try rationalization directly
                    result = try_rational_and_verify(U, V, W, tol=1e-5)
                    if result:
                        return result
                    result = try_rational_and_verify(U, V, W, tol=1e-6)
                    if result:
                        return result
        
        # If we get close but can't rationalize, try more restarts for lower ranks
        if R <= 25 and time.time() < deadline - 30:
            for extra_restart in range(8):
                if time.time() > deadline - 20:
                    break
                seed = 1000 + extra_restart * 100
                U, V, W, err = als_decompose(T, R, max_iters=60, tol=1e-10, 
                                              init_type='random', seed=seed)
                if err < 1e-8:
                    U, V, W, err = gradient_polish(U, V, W, T, max_iters=120, tol=1e-12)
                    if err < 1e-9:
                        snapped = snap_to_half_int(U, V, W)
                        if snapped is not None:
                            U_s, V_s, W_s = snapped
                            result = try_rational_and_verify(U_s, V_s, W_s, tol=1e-5)
                            if result:
                                return result
                        result = try_rational_and_verify(U, V, W, tol=1e-6)
                        if result:
                            return result
    
    # Fallback to schoolbook
    return schoolbook()
