import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Naive decomposition has rank n*m*p = 64
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for a in range(m):
            for j in range(p):
                # U extracts A[i, a]
                # Index in U: i * m + a
                U[i * m + a, r] = 1.0
                
                # V extracts B[a, j]
                # Index in V: a * p + j
                V[a * p + j, r] = 1.0
                
                # W places result into C[i, j]
                # Convention: W[k * n + i, r] where k is likely j (column) and i is row
                # So index is j * n + i
                W[j * n + i, r] = 1.0
                
                r += 1
                
    # Verification to be safe (though naive is analytically correct)
    def verify(U, V, W):
        """Verify tensor decomposition for 2x4x8 matrix multiplication."""
        rank = U.shape[1]
        np.random.seed(123)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r_idx in range(rank):
            u_r = U[:, r_idx].reshape(n, m)   # u_r[i,a]
            v_r = V[:, r_idx].reshape(m, p)   # v_r[a,j]
            w_r = W[:, r_idx].reshape(p, n)   # w_r[j,i]
            for i in range(n):
                for j in range(p):
                    ua = np.dot(u_r[i, :], A_test[i, :])
                    vb = np.dot(v_r[:, j], B_test[:, j])
                    C_computed[i, j] += w_r[j, i] * ua * vb
        
        return np.allclose(C_computed, C_expected, atol=1e-10)

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # If the W convention was actually i*p+j, try that
        W_alt = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for a in range(m):
                for j in range(p):
                    W_alt[i * p + j, r] = 1.0
                    r += 1
        
        def verify_alt(U, V, W):
            """Verify tensor decomposition for 2x4x8 matrix multiplication."""
            rank = U.shape[1]
            np.random.seed(123)
            A_test = np.random.rand(n, m)
            B_test = np.random.rand(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p))
            for r_idx in range(rank):
                u_r = U[:, r_idx].reshape(n, m)
                v_r = V[:, r_idx].reshape(m, p)
                w_r = W[:, r_idx].reshape(n, p)   # w_r[i,j]
                for i in range(n):
                    for j in range(p):
                        ua = np.dot(u_r[i, :], A_test[i, :])
                        vb = np.dot(v_r[:, j], B_test[:, j])
                        C_computed[i, j] += w_r[i, j] * ua * vb
            
            return np.allclose(C_computed, C_expected, atol=1e-10)

        if verify_alt(U, V, W_alt):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W_alt.tolist()
            }
        
        # If both fail, return the first one as a fallback (should not happen for naive)
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
