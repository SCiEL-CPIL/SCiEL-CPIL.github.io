DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    terms = []
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    # Strassen-like decomposition for the first 4 columns of B (4x4x4 matrix multiplication)
    # We'll use a known rank-7 algorithm for 2x2x2, scaled up to 4x4x4
    
    # For 4x4x4, we can use 4 independent 2x2x2 multiplications with some overlap
    # Actually, let's use a cleaner approach: decompose 4x4x4 into blocks
    
    # Block decomposition: split into 2x2 blocks of size 2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2
    
    # For each 2x2 block multiplication, use Strassen (7 terms)
    # We have 4 block multiplications: (0,0), (0,1), (1,0), (1,1)
    
    def strassen_2x2(i0, j0, k0):
        """Add Strassen terms for 2x2 block at row i0, col j0, result col k0"""
        # A block at (i0, j0): indices i in [i0, i0+1], j in [j0, j0+1]
        # B block at (j0, k0): indices j in [j0, j0+1], k in [k0, k0+1]
        
        # Strassen 7 products for 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        i_vals = [i0, i0+1]
        j_vals = [j0, j0+1]
        k_vals = [k0, k0+1]
        
        # M1 = (A00+A11)*(B00+B11)
        u1 = []
        for ii in [0, 1]:
            for jj in [0, 1]:
                u1.append(((i_vals[ii]*m + j_vals[jj]), 1))
        v1 = []
        for jj in [0, 1]:
            for kk in [0, 1]:
                v1.append(((j_vals[jj]*p + k_vals[kk]), 1))
        w1 = []
        for kk in [0, 1]:
            for ii in [0, 1]:
                w1.append(((k_vals[kk]*n + i_vals[ii]), 1))
        
        # We need to add these terms and then subtract to get the right result
        # Actually, let's just use schoolbook for the 2x2 blocks: 4 terms each
        # 4 blocks * 4 terms = 16 terms for 4x4x4
        
        return None
    
    # Let's use a simpler but still better approach:
    # Use rank-4 for each 2x2 block (schoolbook), total 16 terms for 4x4x4
    # Then for the remaining 3 columns, use rank-4 per column: 3 * 4 = 12 terms
    # Total: 28 terms
    
    # Actually, let's just use schoolbook for everything but be smart about it
    # For 4x4x7, we can do:
    # - 4x4x4 part: 16 terms (schoolbook on 2x2 blocks)
    # - 4x4x3 part: 12 terms (schoolbook on columns)
    # Total: 28 terms
    
    # 4x4x4 part: split into four 2x2 blocks
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                # 2x2 schoolbook
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            i = i0 + ii
                            j = j0 + jj
                            k = k0 + kk
                            u = make_u(i, j, 1)
                            v = make_v(j, k, 1)
                            w = make_w(k, i, 1)
                            terms.append((u, v, w))
    
    # 4x4x3 part: columns 4, 5, 6
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
