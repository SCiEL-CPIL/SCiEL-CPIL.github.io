import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140

    def schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 80"""
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
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def verify(U, V, W, num_tests=10):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B  # shape (n, p)
            
            C_decomp = np.zeros((n, p))
            for r in range(rank):
                u_r = U[:, r]
                v_r = V[:, r]
                w_r = W[:, r]
                
                Au = u_r.reshape(n, m)
                Bv = v_r.reshape(m, p)
                Cw = w_r.reshape(p, n)  # Cw[k, i] = W[k*n+i, r]
                
                A_masked = A * Au
                B_masked = B * Bv
                C_decomp += (A_masked @ B_masked) * Cw.T
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                return False
        return True

    def strassen_2x2_terms():
        """
        Return the 7 rank-1 terms for 2x2 matrix multiplication.
        Each term is (u, v, w) where:
          u: 4-vector (coefficients for A flattened row-major)
          v: 4-vector (coefficients for B flattened row-major)
          w: 4-vector (coefficients for C flattened row-major)
        """
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append((np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])))
        # M2 = (A10+A11)B00
        terms.append((np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])))
        # M3 = A00(B10-B11)
        terms.append((np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1]), np.array([0, 1, 0, 0])))
        # M4 = A11(B01-B00)
        terms.append((np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0]), np.array([0, 1, 0, -1])))
        # M5 = (A00+A01)B11
        terms.append((np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1])))
        # M6 = (A10-A00)(B00+B01)
        terms.append((np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([-1, 0, 0, 0])))
        # M7 = (A01-A11)(B10+B11)
        terms.append((np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])))
        return terms

    def build_decomposition():
        """
        Build a rank-65 decomposition:
        - First 49 terms: double Strassen for the 4x4 x 4x4 part (first 4 columns of B)
        - Next 16 terms: schoolbook for the 4x4 x 1 part (5th column of B)
        """
        s2 = strassen_2x2_terms()
        
        U_full = np.zeros((n * m, 65))
        V_full = np.zeros((m * p, 65))
        W_full = np.zeros((p * n, 65))
        
        # First 49 columns: double Strassen for B[:, 0:4] -> C[:, 0:4]
        for oi, (u_o, v_o, w_o) in enumerate(s2):
            for ii, (u_i, v_i, w_i) in enumerate(s2):
                col = oi * 7 + ii
                Uo = u_o.reshape(2, 2)
                Ui = u_i.reshape(2, 2)
                Vo = v_o.reshape(2, 2)
                Vi = v_i.reshape(2, 2)
                Wo = w_o.reshape(2, 2)
                Wi = w_i.reshape(2, 2)
                
                # U: 4x4 matrix A, flattened row-major (i*m + j)
                for a in range(4):
                    for b in range(4):
                        ar = a // 2
                        ai = a % 2
                        bc = b // 2
                        bi = b % 2
                        U_full[a * m + b, col] = Uo[ar, bc] * Ui[ai, bi]
                
                # V: 4x5 matrix B, flattened (j*p + k), only first 4 columns
                for j in range(4):
                    for k in range(4):
                        jr = j // 2
                        ji = j % 2
                        kc = k // 2
                        ki = k % 2
                        V_full[j * p + k, col] = Vo[jr, kc] * Vi[ji, ki]
                
                # W: 5x4 matrix C, flattened (k*n + i), only first 4 rows
                for k in range(4):
                    for i in range(4):
                        kr = k // 2
                        ki_ = k % 2
                        ic = i // 2
                        ii_ = i % 2
                        W_full[k * n + i, col] = Wo[kr, ic] * Wi[ki_, ii_]
        
        # Next 16 columns: schoolbook for B[:, 4] -> C[:, 4]
        for i in range(n):
            for j in range(m):
                col = 49 + i * m + j
                U_full[i * m + j, col] = 1
                V_full[j * p + 4, col] = 1
                W_full[4 * n + i, col] = 1
        
        return {'U': U_full.tolist(), 'V': V_full.tolist(), 'W': W_full.tolist()}

    # Try the Strassen-based decomposition
    try:
        result = build_decomposition()
        if verify(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    result = schoolbook()
    return result
# EVOLVE-BLOCK-END
