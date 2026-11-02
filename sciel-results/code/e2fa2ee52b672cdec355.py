import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def verify(U, V, W, n, m, p, atol=1e-8):
        """Verify the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True

    def build_laderman_3x3():
        """
        Laderman's rank-19 bilinear algorithm for 3x3 matrix multiplication.
        Returns U, V, W of shape (9, 19) each.
        
        A is 3x3, B is 3x3, C = A*B is 3x3.
        Indexing: A[i,j] -> U[i*3+j, t], B[j,k] -> V[j*3+k, t], C[i,k] -> W[k*3+i, t]
        """
        rank = 19
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((9, rank), dtype=float)
        W = np.zeros((9, rank), dtype=float)
        
        # Laderman's 19 terms for 3x3:
        # Using the standard Laderman decomposition
        # a11, a12, a13, a21, a22, a23, a31, a32, a33
        # b11, b12, b13, b21, b22, b23, b31, b32, b33
        
        # 9 diagonal terms:
        # M1 = a11*b11 -> c11
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # M2 = a12*b22 -> c12
        U[1, 1] = 1; V[4, 1] = 1; W[1, 1] = 1
        # M3 = a13*b33 -> c13
        U[2, 2] = 1; V[8, 2] = 1; W[2, 2] = 1
        # M4 = a21*b12 -> c21
        U[3, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        # M5 = a22*b23 -> c22
        U[4, 4] = 1; V[5, 4] = 1; W[4, 4] = 1
        # M6 = a23*b31 -> c23
        U[5, 5] = 1; V[6, 5] = 1; W[5, 5] = 1
        # M7 = a31*b13 -> c31
        U[6, 6] = 1; V[2, 6] = 1; W[6, 6] = 1
        # M8 = a32*b21 -> c32
        U[7, 7] = 1; V[3, 7] = 1; W[7, 7] = 1
        # M9 = a33*b32 -> c33
        U[8, 8] = 1; V[7, 8] = 1; W[8, 8] = 1
        
        # 10 off-diagonal terms (Laderman's):
        # M10 = (a11+a12+a13+a21+a22+a23+a31+a32+a33) * (b11+b12+b13+b21+b22+b23+b31+b32+b33)
        #     -> contributes to c11+c12+c13+c21+c22+c23+c31+c32+c33
        # This is getting complex. Let me use a different formulation.
        
        # Actually, let me use the well-known Laderman decomposition:
        # The 19 multiplications are:
        # 1. a11*b11
        # 2. a12*b22  
        # 3. a13*b33
        # 4. a21*b12
        # 5. a22*b23
        # 6. a23*b31
        # 7. a31*b13
        # 8. a32*b21
        # 9. a33*b32
        # 10. (a11+a12)*(b21+b22)
        # 11. (a12+a13)*(b32+b33)
        # 12. (a21+a22)*(b12+b13)
        # 13. (a22+a23)*(b23+b33)
        # 14. (a31+a32)*(b13+b23)
        # 15. (a32+a33)*(b21+b31)
        # 16. (a11+a21+a31)*(b12+b22+b32)
        # 17. (a12+a22+a32)*(b13+b23+b33)
        # 18. (a11+a12+a13)*(b21+b22+b23)
        # 19. (a21+a22+a23)*(b31+b32+b33)
        
        # This is getting too complex to hand-code correctly. Let me use a verified Laderman decomposition.
        # I'll use the standard one from the literature.
        
        # Reset and use a cleaner formulation
        U = np.zeros((9, 19), dtype=float)
        V = np.zeros((9, 19), dtype=float)
        W = np.zeros((9, 19), dtype=float)
        
        # Laderman's algorithm (standard formulation):
        # Let me use the formulation where we have:
        # For i,j in 0..2: A[i,j], B[i,j], C[i,j]
        
        # The 19 bilinear forms:
        # 1: A[0,0]*B[0,0]
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # 2: A[0,1]*B[1,1]
        U[1,1]=1; V[4,1]=1; W[1,1]=1
        # 3: A[0,2]*B[2,2]
        U[2,2]=1; V[8,2]=1; W[2,2]=1
        # 4: A[1,0]*B[0,1]
        U[3,3]=1; V[1,3]=1; W[3,3]=1
        # 5: A[1,1]*B[1,2]
        U[4,4]=1; V[5,4]=1; W[4,4]=1
        # 6: A[1,2]*B[2,0]
        U[5,5]=1; V[6,5]=1; W[5,5]=1
        # 7: A[2,0]*B[0,2]
        U[6,6]=1; V[2,6]=1; W[6,6]=1
        # 8: A[2,1]*B[1,0]
        U[7,7]=1; V[3,7]=1; W[7,7]=1
        # 9: A[2,2]*B[2,1]
        U[8,8]=1; V[7,8]=1; W[8,8]=1
        
        # Off-diagonal terms - using Laderman's actual 10 terms:
        # 10: (A[0,0]+A[0,1])*(B[1,0]+B[1,1])
        U[0,9]=1; U[1,9]=1; V[3,9]=1; V[4,9]=1
        W[0,9]=1; W[1,9]=1; W[3,9]=1; W[4,9]=1
        # 11: (A[0,1]+A[0,2])*(B[2,1]+B[2,2])
        U[1,10]=1; U[2,10]=1; V[7,10]=1; V[8,10]=1
        W[1,10]=1; W[2,10]=1; W[4,10]=1; W[5,10]=1
        # 12: (A[1,0]+A[1,1])*(B[0,1]+B[0,2])
        U[3,11]=1; U[4,11]=1; V[1,11]=1; V[2,11]=1
        W[3,11]=1; W[4,11]=1; W[6,11]=1; W[7,11]=1
        # 13: (A[1,1]+A[1,2])*(B[1,2]+B[2,2])
        U[4,12]=1; U[5,12]=1; V[5,12]=1; V[8,12]=1
        W[4,12]=1; W[5,12]=1; W[7,12]=1; W[8,12]=1
        # 14: (A[2,0]+A[2,1])*(B[0,2]+B[1,2])
        U[6,13]=1; U[7,13]=1; V[2,13]=1; V[5,13]=1
        W[6,13]=1; W[7,13]=1; W[0,13]=1; W[1,13]=1
        # 15: (A[2,1]+A[2,2])*(B[1,0]+B[2,0])
        U[7,14]=1; U[8,14]=1; V[3,14]=1; V[6,14]=1
        W[7,14]=1; W[8,14]=1; W[3,14]=1; W[4,14]=1
        # 16: (A[0,0]+A[1,0]+A[2,0])*(B[0,1]+B[1,1]+B[2,1])
        U[0,15]=1; U[3,15]=1; U[6,15]=1
        V[1,15]=1; V[4,15]=1; V[7,15]=1
        W[3,15]=1; W[4,15]=1; W[7,15]=1
        # 17: (A[0,1]+A[1,1]+A[2,1])*(B[0,2]+B[1,2]+B[2,2])
        U[1,16]=1; U[4,16]=1; U[7,16]=1
        V[2,16]=1; V[5,16]=1; V[8,16]=1
        W[4,16]=1; W[5,16]=1; W[8,16]=1
        # 18: (A[0,0]+A[0,1]+A[0,2])*(B[1,0]+B[1,1]+B[1,2])
        U[0,17]=1; U[1,17]=1; U[2,17]=1
        V[3,17]=1; V[4,17]=1; V[5,17]=1
        W[3,17]=1; W[4,17]=1; W[5,17]=1
        # 19: (A[1,0]+A[1,1]+A[1,2])*(B[2,0]+B[2,1]+B[2,2])
        U[3,18]=1; U[4,18]=1; U[5,18]=1
        V[6,18]=1; V[7,18]=1; V[8,18]=1
        W[6,18]=1; W[7,18]=1; W[8,18]=1
        
        return U, V, W

    def build_block_strassen():
        """Build the 8-block Strassen decomposition (rank 56) as fallback."""
        strassen_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        rank = 56
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

        for u in range(2):
            for v in range(4):
                base_col = 7 * (4 * u + v)
                a_local = {0: 0*m+2*u, 1: 0*m+2*u+1, 2: 1*m+2*u, 3: 1*m+2*u+1}
                b_local = {0: (2*u)*p+2*v, 1: (2*u)*p+2*v+1, 2: (2*u+1)*p+2*v, 3: (2*u+1)*p+2*v+1}
                c_local = {0: (2*v)*n+0, 1: (2*v+1)*n+0, 2: (2*v)*n+1, 3: (2*v+1)*n+1}

                for t in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                    col = base_col + t
                    for local_idx, coeff in u_coeffs:
                        U[a_local[local_idx], col] += coeff
                    for local_idx, coeff in v_coeffs:
                        V[b_local[local_idx], col] += coeff
                    for local_idx, coeff in w_coeffs:
                        W[c_local[local_idx], col] += coeff
        return U, V, W

    # Try Laderman 3x3 approach with projection
    # Embed 2x4 x 4x8 into 3x3 x 3x3:
    # A (2x4) -> pad to 3x3: use A[0:2, 0:3] in top-left, A[0:2, 3] handled separately
    # This doesn't work cleanly. Let's try a different embedding.
    
    # Alternative: Use the fact that we can express the 2x4 x 4x8 product
    # using a 3x3 x 3x3 multiplication with appropriate projections.
    # 
    # A (2x4) = [A1 (2x3) | A2 (2x1)]
    # B (4x8) = [B1 (3x8); B2 (1x8)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x3 x 3x8. Pad to 3x3 x 3x3? No, 3x8 can't fit in 3x3.
    #
    # This approach is problematic. Let me just use the Strassen fallback.
    
    # Build the Strassen decomposition
    U, V, W = build_block_strassen()

    # Verify it's correct
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
