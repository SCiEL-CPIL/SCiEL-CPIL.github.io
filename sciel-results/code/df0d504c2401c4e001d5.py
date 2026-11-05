"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # We'll build the decomposition by summing rank-1 terms.
    # Each term is (U_vec, V_vec, W_vec) where:
    #   U_vec has length n*m = 12
    #   V_vec has length m*p = 20  
    #   W_vec has length p*n = 15
    # and T[x,y,z] = sum_c U[x,c]*V[y,c]*W[z,c]
    
    terms = []  # list of (U_sparse, V_sparse, W_sparse) as dicts or arrays
    
    def add_term(U_dict, V_dict, W_dict):
        """Add a rank-1 term. Dicts map index -> value (nonzero only)."""
        terms.append((U_dict, V_dict, W_dict))
    
    # Strassen's algorithm for 2x2x2: 7 multiplications
    # Multiplies M1(2x2) * M2(2x2) = M3(2x2)
    # M1 = [[a,b],[c,d]], M2 = [[e,f],[g,h]], M3 = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen:
    #   p1 = (a+d)*(e+h)
    #   p2 = (c+d)*e
    #   p3 = a*(f-h)
    #   p4 = d*(g-e)
    #   p5 = (a+b)*h
    #   p6 = (c-a)*(e+f)
    #   p7 = (b-d)*(g+h)
    #   M3[0,0] = p1+p4-p5+p7
    #   M3[0,1] = p3+p5
    #   M3[1,0] = p2+p4
    #   M3[1,1] = p1-p2+p3+p6
    #
    # For tensor decomposition, each "multiplication" p_i corresponds to a rank-1 term
    # where U picks up the first factor's entries, V picks up the second factor's entries,
    # and W distributes to output entries.
    
    def add_strassen_2x2x2(i0, j0, k0, i1, j1, k1):
        """Add Strassen terms for M1[i0:i1, j0:j1] * M2[j0:j1, k0:k1] = M3[i0:i1, k0:k1]
        where i0,i1 are row indices in A (0-based, i1 exclusive),
              j0,j1 are col/row indices (the shared dimension),
              k0,k1 are col indices in B.
        Requires i1-i0==2, j1-j0==2, k1-k0==2.
        """
        # A indices: (i,j) -> i*m + j for i in [i0,i1), j in [j0,j1)
        # B indices: (j,k) -> j*p + k for j in [j0,j1), k in [k0,k1)
        # C indices: (i,k) -> k*n + i for i in [i0,i1), k in [k0,k1)
        
        # p1 = (a+d)*(e+h): 
        #   a = A[i0,j0] -> U[i0*m+j0]=1, d = A[i1-1,j1-1] -> U[i1-1*m+j1-1]=1
        #   e = B[j0,k0] -> V[j0*p+k0]=1, h = B[j1-1,k1-1] -> V[j1-1*p+k1-1]=1
        #   Output: C[i0,k0]=p1, C[i1-1,k0]=-p1, C[i0,k1-1]=-p1, C[i1-1,k1-1]=p1
        U = {(i0)*m+j0: 1, (i1-1)*m+(j1-1): 1}
        V = {j0*p+k0: 1, (j1-1)*p+(k1-1): 1}
        W = {k0*n+i0: 1, k0*n+(i1-1): -1, (k1-1)*n+i0: -1, (k1-1)*n+(i1-1): 1}
        add_term(U, V, W)
        
        # p2 = (c+d)*e
        #   c = A[i1-1,j0], d = A[i1-1,j1-1], e = B[j0,k0]
        #   Output: C[i1-1,k0]=p2, C[i1-1,k1-1]=-p2
        U = {(i1-1)*m+j0: 1, (i1-1)*m+(j1-1): 1}
        V = {j0*p+k0: 1}
        W = {k0*n+(i1-1): 1, (k1-1)*n+(i1-1): -1}
        add_term(U, V, W)
        
        # p3 = a*(f-h)
        #   a = A[i0,j0], f = B[j0,k1-1], h = B[j1-1,k1-1]
        #   Output: C[i0,k1-1]=p3
        U = {i0*m+j0: 1}
        V = {j0*p+(k1-1): 1, (j1-1)*p+(k1-1): -1}
        W = {(k1-1)*n+i0: 1}
        add_term(U, V, W)
        
        # p4 = d*(g-e)
        #   d = A[i1-1,j1-1], g = B[j1-1,k0], e = B[j0,k0]
        #   Output: C[i1-1,k0]=p4
        U = {(i1-1)*m+(j1-1): 1}
        V = {(j1-1)*p+k0: 1, j0*p+k0: -1}
        W = {k0*n+(i1-1): 1}
        add_term(U, V, W)
        
        # p5 = (a+b)*h
        #   a = A[i0,j0], b = A[i0,j1-1], h = B[j1-1,k1-1]
        #   Output: C[i0,k0]=-p5, C[i1-1,k0]=p5
        U = {i0*m+j0: 1, i0*m+(j1-1): 1}
        V = {(j1-1)*p+(k1-1): 1}
        W = {k0*n+i0: -1, k0*n+(i1-1): 1}
        add_term(U, V, W)
        
        # p6 = (c-a)*(e+f)
        #   c = A[i1-1,j0], a = A[i0,j0], e = B[j0,k0], f = B[j0,k1-1]
        #   Output: C[i1-1,k1-1]=p6
        U = {(i1-1)*m+j0: 1, i0*m+j0: -1}
        V = {j0*p+k0: 1, j0*p+(k1-1): 1}
        W = {(k1-1)*n+(i1-1): 1}
        add_term(U, V, W)
        
        # p7 = (b-d)*(g+h)
        #   b = A[i0,j1-1], d = A[i1-1,j1-1], g = B[j1-1,k0], h = B[j1-1,k1-1]
        #   Output: C[i0,k0]=p7
        U = {i0*m+(j1-1): 1, (i1-1)*m+(j1-1): -1}
        V = {(j1-1)*p+k0: 1, (j1-1)*p+(k1-1): 1}
        W = {k0*n+i0: 1}
        add_term(U, V, W)
    
    def add_schoolbook_block(i0, j0, k0, nr, nc):
        """Add schoolbook terms for A[i0:i0+nr, j0:j0+nc] * B[j0:j0+nc, k0:k0+nc] = C[i0:i0+nr, k0:k0+nc]
        All rectangular, schoolbook.
        """
        for i in range(nr):
            for j in range(nc):
                for k in range(nc):
                    U = {(i0+i)*m + (j0+j): 1}
                    V = {(j0+j)*p + (k0+k): 1}
                    W = {(k0+k)*n + (i0+i): 1}
                    add_term(U, V, W)
    
    # Decomposition strategy for 3x4x5:
    # Split rows of A: top 2 rows (i=0,1), bottom 1 row (i=2)
    # Split cols of A / rows of B: cols 0-1 (j=0,1), cols 2-3 (j=2,3)
    # Split cols of B: cols 0-1 (k=0,1), cols 2-4 (k=2,3,4)
    
    # Block (0,0,0): A[0:2,0:2] * B[0:2,0:2] = C[0:2,0:2] -> 2x2x2, use Strassen
    add_strassen_2x2x2(0, 0, 0, 2, 2, 2)
    
    # Block (0,1,0): A[0:2,2:4] * B[2:4,0:2] = C[0:2,0:2] -> 2x2x2, use Strassen
    add_strassen_2x2x2(0, 2, 0, 2, 4, 2)
    
    # Block (0,0,1): A[0:2,0:2] * B[0:2,2:4] = C[0:2,2:4] -> 2x2x2, use Strassen
    add_strassen_2x2x2(0, 0, 2, 2, 2, 4)
    
    # Block (0,1,1): A[0:2,2:4] * B[2:4,2:4] = C[0:2,2:4] -> 2x2x2, use Strassen
    add_strassen_2x2x2(0, 2, 2, 2, 4, 4)
    
    # Block (1,0,0): A[2:3,0:2] * B[0:2,0:2] = C[2:3,0:2] -> 1x2x2, schoolbook: 1*2*2=4
    add_schoolbook_block(2, 0, 0, 1, 2)
    
    # Block (1,1,0): A[2:3,2:4] * B[2:4,0:2] = C[2:3,0:2] -> 1x2x2, schoolbook: 4
    add_schoolbook_block(2, 2, 0, 1, 2)
    
    # Block (1,0,1): A[2:3,0:2] * B[0:2,2:4] = C[2:3,2:4] -> 1x2x2, schoolbook: 4
    add_schoolbook_block(2, 0, 2, 1, 2)
    
    # Block (1,1,1): A[2:3,2:4] * B[2:4,2:4] = C[2:3,2:4] -> 1x2x2, schoolbook: 4
    add_schoolbook_block(2, 2, 2, 1, 2)
    
    # Total rank: 4*7 + 4*4 = 28 + 16 = 44
    
    rank = len(terms)
    
    # Check deadline
    if time.time() > deadline - 5:
        # Convert to arrays and return
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx, c] = val
            for idx, val in Vd.items():
                V[idx, c] = val
            for idx, val in Wd.items():
                W[idx, c] = val
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify correctness with random test
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, (n, m))
    B_test = np.random.randint(-5, 6, (m, p))
    C_expected = A_test @ B_test
    
    # Build U, V, W matrices
    U_mat = np.zeros((n*m, rank))
    V_mat = np.zeros((m*p, rank))
    W_mat = np.zeros((p*n, rank))
    for c, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U_mat[idx, c] = val
        for idx, val in Vd.items():
            V_mat[idx, c] = val
        for idx, val in Wd.items():
            W_mat[idx, c] = val
    
    # Verify: C[i,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over j
    # Actually the tensor equation is: T[x,y,z] = sum_c U[x,c]*V[y,c]*W[z,c]
    # where x=i*m+j, y=j*p+k, z=k*n+i
    # And C[i,k] = sum_j T[i*m+j, j*p+k, k*n+i]
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                x = i*m + j
                y = j*p + k
                z = k*n + i
                t_val = 0.0
                for c in range(rank):
                    t_val += U_mat[x, c] * V_mat[y, c] * W_mat[z, c]
                s += t_val
            C_computed[i, k] = s
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        terms.clear()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U = {i*m+j: 1}
                    V = {j*p+k: 1}
                    W = {k*n+i: 1}
                    add_term(U, V, W)
        rank = len(terms)
        U_mat = np.zeros((n*m, rank))
        V_mat = np.zeros((m*p, rank))
        W_mat = np.zeros((p*n, rank))
        for c, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U_mat[idx, c] = val
            for idx, val in Vd.items():
                V_mat[idx, c] = val
            for idx, val in Wd.items():
                W_mat[idx, c] = val
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
