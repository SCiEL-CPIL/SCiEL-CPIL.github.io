import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 145

    def verify_decomposition(U, V, W, n, m, p, rank):
        """
        Verify if U, V, W form a valid decomposition for n x m x p.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        # Create random test matrices
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                sum_val = 0.0
                for r in range(rank):
                    u_val = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                    v_val = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                    sum_val += W[k*n+i, r] * u_val * v_val
                C_decomp[i, k] = sum_val
                
        return np.allclose(C_expected, C_decomp, atol=1e-8)

    def try_find_rank_r(n, m, p, r, max_iters=50, timeout_per_iter=2.0):
        """
        Attempt to find a rank-r decomposition for n x m x p.
        Returns (U, V, W) if found, else None.
        """
        if r < 1:
            return None
            
        # Initialize with random values
        for attempt in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U_init = np.random.randn(n*m, r)
            V_init = np.random.randn(m*p, r)
            W_init = np.random.randn(p*n, r)
            
            # Define objective function
            def objective(params):
                U = params[:n*m*r].reshape(n*m, r)
                V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Compute error
                error = 0.0
                # We need to match the tensor T_{(i,j), (j',k), (k',i')} = delta_{jj'} delta_{kk'} delta_{ii'}
                # This is equivalent to checking C = A @ B for random A, B
                # But for optimization, it's better to check the tensor equation directly
                # or use a few random matrices to approximate the error.
                
                # Use 5 random matrices for speed
                err_sum = 0.0
                for _ in range(5):
                    A_rand = np.random.rand(n, m)
                    B_rand = np.random.rand(m, p)
                    C_exp = A_rand @ B_rand
                    C_dec = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                u_val = sum(U[i*m+j, rr] * A_rand[i, j] for j in range(m))
                                v_val = sum(V[j*p+k, rr] * B_rand[j, k] for j in range(m))
                                s += W[k*n+i, rr] * u_val * v_val
                            C_dec[i, k] = s
                    err_sum += np.sum((C_exp - C_dec) ** 2)
                return err_sum / 5.0

            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            result = minimize(objective, params_init, method='L-BFGS-B', 
                              options={'maxiter': 200, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U_opt = result.x[:n*m*r].reshape(n*m, r)
                V_opt = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify rounded version
                if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, r):
                    return U_rounded, V_rounded, W_rounded
                    
        return None

    # Strategy: Try to find a rank-10 decomposition for 2x2x3
    # If found, tile it 3 times for 2x2x9 (rank 30)
    # If not, try rank-11 (rank 33, worse than 32)
    # If not, fall back to current rank-32
    
    # First, try to find rank 10 for 2x2x3
    U3, V3, W3 = try_find_rank_r(2, 2, 3, 10, max_iters=20)
    
    if U3 is not None:
        # Tile it 3 times
        rank_total = 30
        U = np.zeros((2*2, rank_total))
        V = np.zeros((2*9, rank_total))
        W = np.zeros((9*2, rank_total))
        
        for block in range(3):
            k_start = block * 3
            r_start = block * 10
            # Copy U
            U[:, r_start:r_start+10] = U3
            # Copy V (map j*p+k)
            for j in range(2):
                for k in range(3):
                    v_idx = j*9 + (k_start + k)
                    v_idx_3 = j*3 + k
                    V[v_idx, r_start:r_start+10] = V3[v_idx_3, :]
            # Copy W (map k*n+i)
            for k in range(3):
                for i in range(2):
                    w_idx = (k_start + k)*2 + i
                    w_idx_3 = k*2 + i
                    W[w_idx, r_start:r_start+10] = W3[w_idx_3, :]
                    
        if verify_decomposition(U, V, W, 2, 2, 9, 30):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Current rank-32 construction
    def build_rank32():
        rank = 32
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0

        # Verify Strassen numerically first
        np.random.seed(42)
        A_test = np.random.rand(2, 2)
        B_test = np.random.rand(2, 2)
        C_test = A_test @ B_test

        a, b = A_test[0,0], A_test[0,1]
        c, d = A_test[1,0], A_test[1,1]
        x, y = B_test[0,0], B_test[0,1]
        z, w = B_test[1,0], B_test[1,1]

        m1 = (a+d)*(x+w)
        m2 = (c+d)*x
        m3 = (a-b)*(y+w)
        m4 = (c+d)*z
        m5 = (a+b)*x
        m6 = (c-a)*(x+z)
        m7 = (b-d)*(y+w)

        C00 = m1 - m4 + m6 + m7
        C01 = m3 + m4
        C10 = m2 + m5
        C11 = m1 - m3 + m5 - m2

        if not (np.isclose(C00, C_test[0,0]) and np.isclose(C01, C_test[0,1]) and
                np.isclose(C10, C_test[1,0]) and np.isclose(C11, C_test[1,1])):
            return None

        # Build decomposition using verified Strassen
        for pair_idx in range(4):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1

            U[0, col] = 1; U[3, col] = 1
            V[0*p+k1, col] = 1; V[1*p+k2, col] = 1
            W[k1*n+0, col] = 1
            W[k2*n+1, col] = 1
            col += 1

            U[2, col] = 1; U[3, col] = 1
            V[0*p+k1, col] = 1
            W[k1*n+1, col] = 1
            W[k2*n+1, col] = -1
            col += 1

            U[0, col] = 1; U[1, col] = -1
            V[0*p+k2, col] = 1; V[1*p+k2, col] = 1
            W[k2*n+0, col] = 1
            W[k2*n+1, col] = -1
            col += 1

            U[2, col] = 1; U[3, col] = 1
            V[1*p+k1, col] = 1
            W[k1*n+0, col] = -1
            W[k2*n+0, col] = 1
            col += 1

            U[0, col] = 1; U[1, col] = 1
            V[0*p+k1, col] = 1
            W[k1*n+1, col] = 1
            W[k2*n+1, col] = 1
            col += 1

            U[2, col] = 1; U[0, col] = -1
            V[0*p+k1, col] = 1; V[1*p+k1, col] = 1
            W[k1*n+0, col] = 1
            col += 1

            U[1, col] = 1; U[3, col] = -1
            V[0*p+k2, col] = 1; V[1*p+k2, col] = 1
            W[k1*n+0, col] = 1
            col += 1

        # Last single column k=8: rank 4
        k = 8
        U[0, col] = 1; V[0*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[1, col] = 1; V[1*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[2, col] = 1; V[0*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
        U[3, col] = 1; V[1*p+k, col] = 1; W[k*n+1, col] = 1; col += 1

        assert col == rank, f"Expected {rank}, got {col}"

        # Full numerical verification
        np.random.seed(123)
        A_rand = np.random.rand(n, m)
        B_rand = np.random.rand(m, p)
        C_expected = A_rand @ B_rand

        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    u_val = sum(U[i*m+j, r] * A_rand[i,j] for j in range(m))
                    v_val = sum(V[j*p+k, r] * B_rand[j,k] for j in range(m))
                    C_decomp[i,k] += W[k*n+i, r] * u_val * v_val

        if not np.allclose(C_expected, C_decomp, atol=1e-10):
            return None

        return U.tolist(), V.tolist(), W.tolist()

    result = build_rank32()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 36
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
