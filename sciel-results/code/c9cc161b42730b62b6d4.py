import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def build_strassen(i1, i2, j1, j2, k1, k2):
        """Return list of 7 factor dicts for Strassen 2x2x2 block."""
        factors = []
        # M1 = (A00+A11)(B00+B11)
        factors.append({'U': [(i1,j1,1),(i2,j2,1)], 'V': [(j1,k1,1),(j2,k2,1)], 'W': [(k1,i1,1),(k2,i2,1)]})
        # M2 = (A10+A11)B00
        factors.append({'U': [(i2,j1,1),(i2,j2,1)], 'V': [(j1,k1,1)], 'W': [(k1,i2,1),(k2,i2,-1)]})
        # M3 = A00(B01-B11)
        factors.append({'U': [(i1,j1,1)], 'V': [(j1,k2,1),(j2,k2,-1)], 'W': [(k2,i1,1),(k2,i2,1)]})
        # M4 = A11(B10-B00)
        factors.append({'U': [(i2,j2,1)], 'V': [(j2,k1,1),(j1,k1,-1)], 'W': [(k1,i1,1),(k1,i2,1)]})
        # M5 = (A00+A01)B11
        factors.append({'U': [(i1,j1,1),(i1,j2,1)], 'V': [(j2,k2,1)], 'W': [(k1,i1,-1),(k2,i1,1)]})
        # M6 = (A10-A00)(B00+B01)
        factors.append({'U': [(i2,j1,1),(i1,j1,-1)], 'V': [(j1,k1,1),(j1,k2,1)], 'W': [(k2,i2,1)]})
        # M7 = (A01-A11)(B10+B11)
        factors.append({'U': [(i1,j2,1),(i2,j2,-1)], 'V': [(j2,k1,1),(j2,k2,1)], 'W': [(k1,i1,1)]})
        return factors

    def build_outer(i_list, j, k_list):
        """Return list of factor dicts for outer product terms."""
        factors = []
        for i in i_list:
            for k in k_list:
                factors.append({'U': [(i,j,1)], 'V': [(j,k,1)], 'W': [(k,i,1)]})
        return factors

    def assemble(factors):
        rank = len(factors)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c, f in enumerate(factors):
            for (i, j, val) in f['U']:
                U[i*m+j, c] = val
            for (j, k, val) in f['V']:
                V[j*p+k, c] = val
            for (k, i, val) in f['W']:
                W[k*n+i, c] = val
        return U, V, W, rank

    def verify(U, V, W, rank):
        res = np.zeros((n, m, p))
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]
        return np.allclose(res, 1.0, atol=1e-9)

    # Primary construction: two Strassen blocks + outer products (rank 28)
    factors = []
    factors += build_strassen(0, 1, 0, 1, 0, 1)       # A[:,0:2] x B[0:2,0:2] -> C[:,0:2]
    factors += build_outer([0, 1], 2, [0, 1])          # A[:,2] x B[2,0:2] -> C[:,0:2]
    factors += build_strassen(0, 1, 0, 1, 2, 3)        # A[:,0:2] x B[0:2,2:4] -> C[:,2:4]
    factors += build_outer([0, 1], 0, [4])             # A[:,0] x B[0,4] -> C[:,4]
    factors += build_outer([0, 1], 1, [4])             # A[:,1] x B[1,4] -> C[:,4]
    factors += build_outer([0, 1], 2, [2, 3, 4])       # A[:,2] x B[2,2:5] -> C[:,2:5]

    U, V, W, rank = assemble(factors)

    if not verify(U, V, W, rank):
        # Fallback to schoolbook (rank 30)
        fb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    fb.append({'U': [(i,j,1)], 'V': [(j,k,1)], 'W': [(k,i,1)]})
        U, V, W, rank = assemble(fb)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
