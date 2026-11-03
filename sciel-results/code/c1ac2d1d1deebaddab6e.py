"""Rewritten program for 3x5x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    def build_2x2x3_strassen():
        """
        Build rank-8 decomposition for 2x2 @ 2x3 = 2x3.
        Local indices: i in {0,1}, j in {0,1}, k in {0,1,2}
        Returns (U_local, V_local, W_local) as lists of dicts or arrays.
        """
        # We'll return the rank-1 terms as (u_idx, v_idx, w_idx, coeff)
        # where u_idx is (i,j), v_idx is (j,k), w_idx is (k,i)
        # But we need to map to global indices later.
        
        # For now, let's define the 8 multiplications with their contributions
        # to U (row i*m+j), V (col j*p+k), W (out k*n+i)
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # M3 = A[0,0] * (B[1,0]-B[0,0])
        # M4 = A[1,1] * (B[1,1]-B[0,1])
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        # M8 = (A[0,0]+A[1,1]) * (B[0,1]-B[1,0])
        
        # C[0,0] = M1 + M4 + M5
        # C[0,1] = M3 + M5 + M8
        # C[0,2] = ? need to check
        # C[1,0] = M2 + M4 + M7
        # C[1,1] = M1 + M3 + M6
        # C[1,2] = ?
        
        # Let me verify M1:
        # (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # M4: A11*(B11-B01) = A11*B11 - A11*B01
        # M5: (A00+A01)*B11 = A00*B11 + A01*B11
        # M1+M4+M5 = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B11 - A11*B01 + A00*B11 + A01*B11
        #           = A00*B00 + 2*A00*B11 + A11*B00 + 2*A11*B11 - A11*B01 + A01*B11
        # But C[0,0] = A00*B00 + A01*B10
        # This doesn't match. My formula is wrong.
        
        # Let me use a correct 2x2x3 algorithm.
        # Reference: "Fast rectangular matrix multiplication" 
        # For 2x2 * 2x3, rank 8:
        
        # Let me just use the known rank-8 algorithm for 2x2x3.
        # I'll define it as a list of (u_factors, v_factors, w_factors)
        
        # Actually, let me be more careful and derive it.
        # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
        # C = [[a*e+b*h, a*f+b*i, a*g+b*j], [c*e+d*h, c*f+d*i, c*g+d*j]]
        
        # 8 multiplications:
        # M1 = (a+d)(e+i)
        # M2 = (c+d)e
        # M3 = a(h-e)
        # M4 = d(i-f)
        # M5 = (a+b)i
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(h+i)
        # M8 = (a+d)(f-h)
        
        # C[0,0] = M1 + M4 + M5 = (a+d)(e+i) + d(i-f) + (a+b)i
        #         = ae + ai + de + di + di - df + ai + bi
        #         = ae + 2ai + de + 2di - df + bi
        # Should be: ae + bh. Not matching.
        
        # I think I need to look this up properly. Let me use a different approach.
        # For now, let me just use 2x2x2 Strassen (rank 7) for k-pairs and handle
        # the extra k with schoolbook, which gives 99.
        
        # To beat 99, let me try a different tiling.
        return None
    
    # Let me try a cleaner approach: just use 2x2x2 Strassen blocks
    # and see if I can find a better tiling than the current 99.
    
    def build_decomposition(tiling_strategy):
        """
        tiling_strategy: list of (i_range, j_range, k_range, method)
        method: 'strassen' for 2x2x2, 'schoolbook' for any size
        """
        U = np.zeros((n * m, 200))
        V = np.zeros((m * p, 200))
        W = np.zeros((p * n, 200))
        
        col = 0
        
        def add_strassen_2x2x2(i0, j0, k0):
            nonlocal col
            # Standard Strassen 2x2x2
            # M1 = (A11+A22)(B11+B22)
            U[i0 * m + j0, col] += 1
            U[(i0+1) * m + j0+1, col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0+1) * p + k0+1, col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0+1) * n + i0+1, col] += 1
            col += 1
            
            # M2 = (A21+A22)B11
            U[(i0+1) * m + j0, col] += 1
            U[(i0+1) * m + j0+1, col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + i0+1, col] += 1
            W[(k0+1) * n + i0+1, col] -= 1
            col += 1
            
            # M3 = A11(B21-B11)
            U[i0 * m + j0, col] += 1
            V[j0 * p + k0+1, col] += 1
            V[(j0+1) * p + k0+1, col] -= 1
            W[(k0+1) * n + i0, col] += 1
            W[(k0+1) * n + i0+1, col] += 1
            col += 1
            
            # M4 = A22(B22-B12)
            U[(i0+1) * m + j0+1, col] += 1
            V[(j0+1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + i0+1, col] += 1
            col += 1
            
            # M5 = (A11+A12)B22
            U[i0 * m + j0, col] += 1
            U[i0 * m + j0+1, col] += 1
            V[(j0+1) * p + k0+1, col] += 1
            W[(k0+1) * n + i0, col] += 1
            W[k0 * n + i0, col] -= 1
            col += 1
            
            # M6 = (A21-A11)(B11+B12)
            U[(i0+1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + k0+1, col] += 1
            W[(k0+1) * n + i0+1, col] += 1
            col += 1
            
            # M7 = (A12-A22)(B21+B22)
            U[i0 * m + j0+1, col] += 1
            U[(i0+1) * m + j0+1, col] -= 1
            V[(j0+1) * p + k0, col] += 1
            V[(j0+1) * p + k0+1, col] += 1
            W[k0 * n + i0, col] += 1
            col += 1
        
        def add_schoolbook(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        
        # Tiling: i={0,1}, j={0,1}, k={0,1},2,3,4,5,6
        for k0 in [0, 2, 4]:
            add_strassen_2x2x2(0, 0, k0)
        add_schoolbook([0, 1], [0, 1], [6])
        
        # i={0,1}, j={2,3}, k=0..6
        for k0 in [0, 2, 4]:
            add_strassen_2x2x2(0, 2, k0)
        add_schoolbook([0, 1], [2, 3], [6])
        
        # i={0,1}, j={4}, k=0..6
        add_schoolbook([0, 1], [4], range(7))
        
        # i={2}, j=0..4, k=0..6
        add_schoolbook([2], range(5), range(7))
        
        return U[:, :col], V[:, :col], W[:, :col], col
    
    U, V, W, rank = build_decomposition(None)
    
    # Verify
    def verify(U, V, W, rank):
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    expected = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
                    computed = np.dot(U[a], V[b] * W[c])
                    if abs(computed - expected) > 1e-9:
                        return False
        return True
    
    is_valid = verify(U, V, W, rank)
    if not is_valid:
        # Fallback
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
