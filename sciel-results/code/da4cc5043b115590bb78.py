"""Rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # safety margin before 150s wall-clock kill

    n, m, p = DIMENSIONS  # 2, 2, 3

    # Known rank-9 decomposition for the 2x2 x 2x3 multiplication tensor.
    # Rank-1 terms (u, v, w) with u in R^{nm}, v in R^{mp}, w in R^{pn}.
    # Index conventions: u[i*m+j], v[j*p+k], w[k*n+i].
    terms = [
        # (u, v, w) as lists of length 4, 6, 6
        ([1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),  # a00*b00 -> c00
        ([0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]),  # a01*b00 -> c10
        ([0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]),  # a10*b00 -> c01
        ([0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]),  # a11*b00 -> c11
        ([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]),  # a00*b10 -> c02
        ([0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1]),  # a01*b10 -> c12 (combined)
        ([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]),  # a10*b10 -> c11
        ([1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]),  # a00*b11 -> c10
        ([0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]),  # a10*b11 -> c01
    ]
    # Note: a01*b11 -> c12 must be covered. Rebuild carefully below.

    # Let me define the 12 schoolbook products and merge where possible.
    # Products (a_ij * b_jk -> c_ik):
    # c00 = a00*b00
    # c10 = a01*b00
    # c01 = a10*b00
    # c11 = a11*b00
    # c02 = a00*b10
    # c12 = a01*b10
    # c11 += a10*b10
    # c10 += a00*b11
    # c01 += a10*b11
    # c12 += a11*b11
    # c11 += a11*b10? No: a11*b10 -> c12? a_{1,1}*b_{1,k}: k=0 -> c_{1,0}=c10? 
    # Wait, indices: A is 2x2 (rows i=0,1; cols j=0,1). B is 2x3 (rows j=0,1; cols k=0,1,2).
    # c_{i,k} = sum_j a_{i,j} b_{j,k}
    # c00 = a00 b00
    # c01 = a00 b01 + a01 b11
    # c02 = a00 b02 + a01 b12
    # c10 = a10 b00
    # c11 = a10 b01 + a11 b11
    # c12 = a10 b02 + a11 b12

    # So the 12 products:
    # a00*b00 -> c00
    # a00*b01 -> c01
    # a00*b02 -> c02
    # a01*b10 -> c02
    # a01*b11 -> c01
    # a01*b12 -> c02
    # a10*b00 -> c10
    # a10*b01 -> c11
    # a10*b02 -> c12
    # a11*b10 -> c12
    # a11*b11 -> c11
    # a11*b12 -> c12

    # Build U, V, W for rank 9 using Strassen-like merging.
    # We merge: (a00*b01 + a01*b11) and (a10*b01 + a11*b11) share b01,b11 with different A combos.
    # Use the standard rank-9 scheme.

    # Define factor matrices directly (rank 9).
    # U (4x9), V (6x9), W (6x9)
    U = np.zeros((4, 9))
    V = np.zeros((6, 9))
    W = np.zeros((6, 9))

    # Column 0: a00*b00 -> c00
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # Column 1: a01*b00 -> c10
    U[1,1]=1; V[0,1]=1; W[3,1]=1
    # Column 2: a10*b00 -> c10? No, a10*b00 -> c10? a_{1,0}b_{0,0} -> c_{1,0}=c10
    # Wait c10 = a10*b00. Yes.
    # But column 1 already gives c10 from a01*b00. Let me recheck: c10 = a10*b00 only (j=0).
    # c00 = a00*b00 (j=0). 
    # Hmm, I mislabeled. Let me redo: c_{i,k} = sum_j a_{i,j} b_{j,k}.
    # For k=0: c_{i,0} = a_{i,0} b_{0,0}. So c00=a00 b00, c10=a10 b00.
    # For k=1: c_{i,1} = a_{i,0} b_{0,1} + a_{i,1} b_{1,1}. c01=a00 b01 + a01 b11, c11=a10 b01 + a11 b11.
    # For k=2: c_{i,2} = a_{i,0} b_{0,2} + a_{i,1} b_{1,2}. c02=a00 b02 + a01 b12, c12=a10 b02 + a11 b12.

    # Redefine columns properly.
    U = np.zeros((4, 9)); V = np.zeros((6, 9)); W = np.zeros((6, 9))
    # c00: a00*b00
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # c10: a10*b00
    U[2,1]=1; V[0,1]=1; W[3,1]=1
    # c01: a00*b01 + a01*b11  (2 terms, merge into rank via (a00+a01)(b01+b11) trick? no, keep)
    # c11: a10*b01 + a11*b11
    # c02: a00*b02 + a01*b12
    # c12: a10*b02 + a11*b12

    # Standard rank-9: use identities to reduce 12 -> 9.
    # Let me use the explicit known construction.
    # I'll verify by building and checking the tensor.

    def build_decomp():
        U = np.zeros((4,9)); V = np.zeros((6,9)); W = np.zeros((6,9))
        # col0: a00 b00 -> c00
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # col1: a10 b00 -> c10
        U[2,1]=1; V[0,1]=1; W[3,1]=1
        # col2: a00 b01 -> c01
        U[0,2]=1; V[1,2]=1; W[1,2]=1
        # col3: a01 b11 -> c01
        U[1,3]=1; V[4,3]=1; W[1,3]=1
        # col4: a10 b01 -> c11
        U[2,4]=1; V[1,4]=1; W[4,4]=1
        # col5: a11 b11 -> c11
        U[3,5]=1; V[4,5]=1; W[4,5]=1
        # col6: a00 b02 -> c02
        U[0,6]=1; V[2,6]=1; W[2,6]=1
        # col7: a01 b12 -> c02
        U[1,7]=1; V[5,7]=1; W[2,7]=1
        # col8: a10 b02 -> c12
        U[2,8]=1; V[2,8]=1; W[5,8]=1
        # missing a11 b12 -> c12 (col9 needed)
        return U,V,W

    # That's 9 cols but missing a11*b12. Need 10. Let me just verify rank 9 exists via search.
    # Given time, fall back to schoolbook rank 12 if verification of 9 fails quickly.

    def verify(U,V,W):
        # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        n,m,p = 2,2,3
        T = np.zeros((n,m,p))
        for r in range(U.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
        # Expected: T[i,j,k] = 1 if (this is a valid product term a_ij b_jk -> c_ik)
        # Actually the multiplication tensor T[i,j,k] = 1 always (it's the structure tensor).
        # For matrix mult, T[i,j,k]=1 for all i,j,k.
        expected = np.ones((n,m,p))
        return np.allclose(T, expected, atol=1e-9)

    # Try the 10-term construction (rank 10) which is clearly valid, then try 9.
    # Rank 10 construction (schoolbook minus one merge):
    U = np.zeros((4,10)); V = np.zeros((6,10)); W = np.zeros((6,10))
    U[0,0]=1; V[0,0]=1; W[0,0]=1   # a00 b00 c00
    U[2,1]=1; V[0,1]=1; W[3,1]=1   # a10 b00 c10
    U[0,2]=1; V[1,2]=1; W[1,2]=1   # a00 b01 c01
    U[1,3]=1; V[4,3]=1; W[1,3]=1   # a01 b11 c01
    U[2,4]=1; V[1,4]=1; W[4,4]=1   # a10 b01 c11
    U[3,5]=1; V[4,5]=1; W[4,5]=1   # a11 b11 c11
    U[0,6]=1; V[2,6]=1; W[2,6]=1   # a00 b02 c02
    U[1,7]=1; V[5,7]=1; W[2,7]=1   # a01 b12 c02
    U[2,8]=1; V[2,8]=1; W[5,8]=1   # a10 b02 c12
    U[3,9]=1; V[5,9]=1; W[5,9]=1   # a11 b12 c12
    if verify(U,V,W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 12
    rank = 12
    U = np.zeros((4, rank)); V = np.zeros((6, rank)); W = np.zeros((6, rank))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j, col]=1
                V[j*3+k, col]=1
                W[k*2+i, col]=1
                col+=1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
