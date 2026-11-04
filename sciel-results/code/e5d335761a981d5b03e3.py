"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Check all valid (i,j,k) positions: should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero positions: for all (a,b,c) not in the support, should be 0
        # Support: (i*m+j, j*p+k, k*n+i) for i in [0,n), j in [0,m), k in [0,p)
        # We need to check that for any (a, b, c) where the indices don't align, the sum is 0
        # a = i*m + j, b = j'*p + k, c = k'*n + i'
        # The tensor entry T[a,b,c] = 1 iff j==j' and k==k' and i==i'
        # So for all (i,j,k,j',k',i') with not (j==j' and k==k' and i==i'), sum should be 0
        # But that's too many checks. Instead, we can check a smaller set:
        # For each (i,j,k), check that U[i*m+j,:] * V[j'*p+k,:] * W[k*n+i,:] = 0 for j'!=j
        # and U[i*m+j,:] * V[j*p+k'] * W[k*n+i,:] = 0 for k'!=k
        # and U[i*m+j,:] * V[j*p+k] * W[k*n+i'] = 0 for i'!=i
        # Actually, the full check is: for all a in [0,nm), b in [0,mp), c in [0,pn),
        # sum_r U[a,r]*V[b,r]*W[c,r] = 1 if (a,m+b,p+c) is in support, else 0
        # Let's do a more efficient check:
        # For each i in [0,n):
        #   For each j in [0,m):
        #     For each j2 in [0,m):
        #       If j2 != j:
        #         For each k in [0,p):
        #           val = dot(U[i*m+j,:], V[j2*p+k,:] * W[k*n+i,:]) should be 0
        #     For each k in [0,p):
        #       For each k2 in [0,p):
        #         If k2 != k:
        #           val = dot(U[i*m+j,:], V[j*p+k2,:] * W[k*n+i,:]) should be 0
        #       For each i2 in [0,n):
        #         If i2 != i:
        #           val = dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i2,:]) should be 0
        # This is O(n*m*m*p + n*m*p*p + n*m*p*n) = O(n*m*(m*p + p*p + p*n))
        # For n=2, m=4, p=5: 2*4*(4*5 + 5*5 + 5*2) = 8*(20+25+10) = 8*55 = 440 checks
        # Each check is a dot product of length r, so total O(440*r)
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j2 * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        if abs(val) > tol:
                            return False
                for k in range(p):
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        idx_u = i * m + j
                        idx_v = j * p + k2
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        if abs(val) > tol:
                            return False
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i2
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        if abs(val) > tol:
                            return False
        return True

    def schoolbook():
        rank0 = n * m * p
        U = np.zeros((nm, rank0))
        V = np.zeros((mp, rank0))
        W = np.zeros((pn, rank0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def get_strassen_222(j_offset, k_offset):
        """Strassen 2x2x2 decomposition for sub-blocks with j in {j_offset, j_offset+1} and k in {k_offset, k_offset+1}."""
        U = np.zeros((nm, 7))
        V = np.zeros((mp, 7))
        W = np.zeros((pn, 7))
        
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Global: U[i*4 + (j_offset+j)], V[(j_offset+j)*5 + (k_offset+k)], W[(k_offset+k)*2 + i]
        
        # M1 = (A00+A11)(B00+B11)
        U[0*4 + (0 + j_offset), 0] = 1.0
        U[1*4 + (1 + j_offset), 0] = 1.0
        V[(0 + j_offset)*5 + (0 + k_offset), 0] = 1.0
        V[(1 + j_offset)*5 + (1 + k_offset), 0] = 1.0
        W[(0 + k_offset)*2 + 0, 0] = 1.0
        W[(1 + k_offset)*2 + 1, 0] = 1.0

        # M2 = (A10+A11)B00
        U[1*4 + (0 + j_offset), 1] = 1.0
        U[1*4 + (1 + j_offset), 1] = 1.0
        V[(0 + j_offset)*5 + (0 + k_offset), 1] = 1.0
        W[(0 + k_offset)*2 + 1, 1] = 1.0
        W[(1 + k_offset)*2 + 1, 1] = -1.0

        # M3 = A00(B01-B11)
        U[0*4 + (0 + j_offset), 2] = 1.0
        V[(0 + j_offset)*5 + (1 + k_offset), 2] = 1.0
        V[(1 + j_offset)*5 + (1 + k_offset), 2] = -1.0
        W[(1 + k_offset)*2 + 0, 2] = 1.0
        W[(1 + k_offset)*2 + 1, 2] = 1.0

        # M4 = A11(B10-B00)
        U[1*4 + (1 + j_offset), 3] = 1.0
        V[(1 + j_offset)*5 + (0 + k_offset), 3] = 1.0
        V[(0 + j_offset)*5 + (0 + k_offset), 3] = -1.0
        W[(0 + k_offset)*2 + 0, 3] = 1.0
        W[(0 + k_offset)*2 + 1, 3] = 1.0

        # M5 = (A00+A01)B11
        U[0*4 + (0 + j_offset), 4] = 1.0
        U[0*4 + (1 + j_offset), 4] = 1.0
        V[(1 + j_offset)*5 + (1 + k_offset), 4] = 1.0
        W[(0 + k_offset)*2 + 0, 4] = -1.0
        W[(1 + k_offset)*2 + 0, 4] = 1.0

        # M6 = (A10-A00)(B00+B01)
        U[1*4 + (0 + j_offset), 5] = 1.0
        U[0*4 + (0 + j_offset), 5] = -1.0
        V[(0 + j_offset)*5 + (0 + k_offset), 5] = 1.0
        V[(0 + j_offset)*5 + (1 + k_offset), 5] = 1.0
        W[(1 + k_offset)*2 + 1, 5] = 1.0

        # M7 = (A01-A11)(B10+B11)
        U[0*4 + (1 + j_offset), 6] = 1.0
        U[1*4 + (1 + j_offset), 6] = -1.0
        V[(1 + j_offset)*5 + (0 + k_offset), 6] = 1.0
        V[(1 + j_offset)*5 + (1 + k_offset), 6] = 1.0
        W[(0 + k_offset)*2 + 0, 6] = 1.0

        return U, V, W

    def get_trivial_221(j_offset, k_offset):
        """Trivial 2x2x1 decomposition for sub-blocks with j in {j_offset, j_offset+1} and k = k_offset."""
        U = np.zeros((nm, 4))
        V = np.zeros((mp, 4))
        W = np.zeros((pn, 4))
        
        col = 0
        for i in range(2):
            for j in range(2):
                U[i*4 + (j_offset + j), col] = 1.0
                V[(j_offset + j)*5 + k_offset, col] = 1.0
                W[k_offset*2 + i, col] = 1.0
                col += 1
        return U, V, W

    # Build rank-36 decomposition using Strassen for 2x2x2 blocks and trivial for 2x2x1
    # Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}
    # Split p=5 into: k in {0,1}, k in {2,3}, k in {4}
    
    # Block 1: j in {0, 1}
    U1_s1, V1_s1, W1_s1 = get_strassen_222(0, 0)   # k in {0, 1}
    U1_s2, V1_s2, W1_s2 = get_strassen_222(0, 2)   # k in {2, 3}
    U1_t1, V1_t1, W1_t1 = get_trivial_221(0, 4)    # k in {4}
    
    U_block1 = np.hstack([U1_s1, U1_s2, U1_t1])
    V_block1 = np.hstack([V1_s1, V1_s2, V1_t1])
    W_block1 = np.hstack([W1_s1, W1_s2, W1_t1])

    # Block 2: j in {2, 3}
    U2_s1, V2_s1, W2_s1 = get_strassen_222(2, 0)   # k in {0, 1}
    U2_s2, V2_s2, W2_s2 = get_strassen_222(2, 2)   # k in {2, 3}
    U2_t1, V2_t1, W2_t1 = get_trivial_221(2, 4)    # k in {4}
    
    U_block2 = np.hstack([U2_s1, U2_s2, U2_t1])
    V_block2 = np.hstack([V2_s1, V2_s2, V2_t1])
    W_block2 = np.hstack([W2_s1, W2_s2, W2_t1])

    U_full = np.hstack([U_block1, U_block2])
    V_full = np.hstack([V_block1, V_block2])
    W_full = np.hstack([W_block1, W_block2])

    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback to schoolbook
    U_s, V_s, W_s = schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
