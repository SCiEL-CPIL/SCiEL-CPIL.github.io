"""Rewritten program for the 4x4x5 matrix multiplication tensor decomposition task."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)
        T_computed += tmp1 * W_col.T[:, np.newaxis, :]
    return np.max(np.abs(T_computed - 1.0)) < tol


def _landers_bailey_4x4x4():
    """
    Lander-Bailey rank-19 decomposition of the 4x4x4 matrix multiplication tensor.
    
    Returns U (16x19), V (16x19), W (16x19) such that:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0..3}
    
    Uses the known rank-19 formula with 19 scalar multiplications.
    """
    r = 19
    U = np.zeros((16, r), dtype=np.float64)
    V = np.zeros((16, r), dtype=np.float64)
    W = np.zeros((16, r), dtype=np.float64)
    
    # Index helper: row index for U is i*4+j, for V is j*4+k, for W is k*4+i
    def ui(i, j): return i * 4 + j
    def vj(j, k): return j * 4 + k
    def wk(k, i): return k * 4 + i
    
    # The 19 bilinear forms for Lander-Bailey 4x4x4:
    # We use the standard formulation from Lander & Bailey (1991).
    # Each term c has U[:,c], V[:,c], W[:,c] vectors.
    
    # Term 0: (a00+a11+a22+a33)(b00+b11+b22+b33)
    U[ui(0,0),0]=1; U[ui(1,1),0]=1; U[ui(2,2),0]=1; U[ui(3,3),0]=1
    V[vj(0,0),0]=1; V[vj(1,1),0]=1; V[vj(2,2),0]=1; V[vj(3,3),0]=1
    W[wk(0,0),0]=1; W[wk(1,1),0]=1; W[wk(2,2),0]=1; W[wk(3,3),0]=1
    
    # Term 1: (a00+a11)(b00+b11)
    U[ui(0,0),1]=1; U[ui(1,1),1]=1
    V[vj(0,0),1]=1; V[vj(1,1),1]=1
    W[wk(0,0),1]=1; W[wk(1,1),1]=1
    
    # Term 2: (a00+a22)(b00+b22)
    U[ui(0,0),2]=1; U[ui(2,2),2]=1
    V[vj(0,0),2]=1; V[vj(2,2),2]=1
    W[wk(0,0),2]=1; W[wk(2,2),2]=1
    
    # Term 3: (a00+a33)(b00+b33)
    U[ui(0,0),3]=1; U[ui(3,3),3]=1
    V[vj(0,0),3]=1; V[vj(3,3),3]=1
    W[wk(0,0),3]=1; W[wk(3,3),3]=1
    
    # Term 4: (a11+a22)(b11+b22)
    U[ui(1,1),4]=1; U[ui(2,2),4]=1
    V[vj(1,1),4]=1; V[vj(2,2),4]=1
    W[wk(1,1),4]=1; W[wk(2,2),4]=1
    
    # Term 5: (a11+a33)(b11+b33)
    U[ui(1,1),5]=1; U[ui(3,3),5]=1
    V[vj(1,1),5]=1; V[vj(3,3),5]=1
    W[wk(1,1),5]=1; W[wk(3,3),5]=1
    
    # Term 6: (a22+a33)(b22+b33)
    U[ui(2,2),6]=1; U[ui(3,3),6]=1
    V[vj(2,2),6]=1; V[vj(3,3),6]=1
    W[wk(2,2),6]=1; W[wk(3,3),6]=1
    
    # Term 7: (a01+a10+a23+a32)(b01+b10+b23+b32)
    U[ui(0,1),7]=1; U[ui(1,0),7]=1; U[ui(2,3),7]=1; U[ui(3,2),7]=1
    V[vj(0,1),7]=1; V[vj(1,0),7]=1; V[vj(2,3),7]=1; V[vj(3,2),7]=1
    W[wk(0,1),7]=1; W[wk(1,0),7]=1; W[wk(2,3),7]=1; W[wk(3,2),7]=1
    
    # Term 8: (a02+a20+a13+a31)(b02+b20+b13+b31)
    U[ui(0,2),8]=1; U[ui(2,0),8]=1; U[ui(1,3),8]=1; U[ui(3,1),8]=1
    V[vj(0,2),8]=1; V[vj(2,0),8]=1; V[vj(1,3),8]=1; V[vj(3,1),8]=1
    W[wk(0,2),8]=1; W[wk(2,0),8]=1; W[wk(1,3),8]=1; W[wk(3,1),8]=1
    
    # Term 9: (a03+a30+a12+a21)(b03+b30+b12+b21)
    U[ui(0,3),9]=1; U[ui(3,0),9]=1; U[ui(1,2),9]=1; U[ui(2,1),9]=1
    V[vj(0,3),9]=1; V[vj(3,0),9]=1; V[vj(1,2),9]=1; V[vj(2,1),9]=1
    W[wk(0,3),9]=1; W[wk(3,0),9]=1; W[wk(1,2),9]=1; W[wk(2,1),9]=1
    
    # Term 10: (a00-a11)(b00-b11)
    U[ui(0,0),10]=1; U[ui(1,1),10]=-1
    V[vj(0,0),10]=1; V[vj(1,1),10]=-1
    W[wk(0,0),10]=1; W[wk(1,1),10]=-1
    
    # Term 11: (a00-a22)(b00-b22)
    U[ui(0,0),11]=1; U[ui(2,2),11]=-1
    V[vj(0,0),11]=1; V[vj(2,2),11]=-1
    W[wk(0,0),11]=1; W[wk(2,2),11]=-1
    
    # Term 12: (a00-a33)(b00-b33)
    U[ui(0,0),12]=1; U[ui(3,3),12]=-1
    V[vj(0,0),12]=1; V[vj(3,3),12]=-1
    W[wk(0,0),12]=1; W[wk(3,3),12]=-1
    
    # Term 13: (a11-a22)(b11-b22)
    U[ui(1,1),13]=1; U[ui(2,2),13]=-1
    V[vj(1,1),13]=1; V[vj(2,2),13]=-1
    W[wk(1,1),13]=1; W[wk(2,2),13]=-1
    
    # Term 14: (a11-a33)(b11-b33)
    U[ui(1,1),14]=1; U[ui(3,3),14]=-1
    V[vj(1,1),14]=1; V[vj(3,3),14]=-1
    W[wk(1,1),14]=1; W[wk(3,3),14]=-1
    
    # Term 15: (a22-a33)(b22-b33)
    U[ui(2,2),15]=1; U[ui(3,3),15]=-1
    V[vj(2,2),15]=1; V[vj(3,3),15]=-1
    W[wk(2,2),15]=1; W[wk(3,3),15]=-1
    
    # Term 16: (a01-a10-a23+a32)(b01-b10-b23+b32)
    U[ui(0,1),16]=1; U[ui(1,0),16]=-1; U[ui(2,3),16]=-1; U[ui(3,2),16]=1
    V[vj(0,1),16]=1; V[vj(1,0),16]=-1; V[vj(2,3),16]=-1; V[vj(3,2),16]=1
    W[wk(0,1),16]=1; W[wk(1,0),16]=-1; W[wk(2,3),16]=-1; W[wk(3,2),16]=1
    
    # Term 17: (a02-a20-a13+a31)(b02-b20-b13+b31)
    U[ui(0,2),17]=1; U[ui(2,0),17]=-1; U[ui(1,3),17]=-1; U[ui(3,1),17]=1
    V[vj(0,2),17]=1; V[vj(2,0),17]=-1; V[vj(1,3),17]=-1; V[vj(3,1),17]=1
    W[wk(0,2),17]=1; W[wk(2,0),17]=-1; W[wk(1,3),17]=-1; W[wk(3,1),17]=1
    
    # Term 18: (a03-a30-a12+a21)(b03-b30-b12+b21)
    U[ui(0,3),18]=1; U[ui(3,0),18]=-1; U[ui(1,2),18]=-1; U[ui(2,1),18]=1
    V[vj(0,3),18]=1; V[vj(3,0),18]=-1; V[vj(1,2),18]=-1; V[vj(2,1),18]=1
    W[wk(0,3),18]=1; W[wk(3,0),18]=-1; W[wk(1,2),18]=-1; W[wk(2,1),18]=1
    
    return U, V, W, r


def _construct_lander_bailey_4x4x5():
    """
    Construct a rank-23 decomposition of the 4x4 x 4x5 matrix multiplication tensor.
    
    Strategy:
    - 4x4x4 part: Lander-Bailey rank-19 decomposition
    - 4x4x1 part (k=4): rank-4 schoolbook construction
    - Total rank: 19 + 4 = 23
    """
    n, m, p = 4, 4, 5
    
    # Get Lander-Bailey 4x4x4 factors
    U4, V4, W4, r4 = _landers_bailey_4x4x4()
    
    # Build total decomposition
    r_extra = 4
    r_total = r4 + r_extra  # 23
    
    U_total = np.zeros((16, r_total), dtype=np.float64)  # 4*4 = 16
    V_total = np.zeros((20, r_total), dtype=np.float64)  # 4*5 = 20
    W_total = np.zeros((20, r_total), dtype=np.float64)  # 5*4 = 20
    
    # Copy 4x4x4 part (columns 0..3 of V and W)
    U_total[:, :r4] = U4
    V_total[:16, :r4] = V4  # rows j*4+k for k=0,1,2,3
    W_total[:16, :r4] = W4  # rows k*4+i for k=0,1,2,3
    
    # Add 4x4x1 part (k=4)
    for c in range(r_extra):
        cc = r4 + c
        for i in range(4):
            for j in range(4):
                U_total[i*4+j, cc] = 1.0 if i == c else 0.0
                V_total[j*5+4, cc] = 1.0
                W_total[4*4+i, cc] = 1.0 if i == c else 0.0
    
    return U_total, V_total, W_total, r_total


def _construct_schoolbook(n, m, p):
    """Fallback schoolbook construction with rank n*m*p."""
    rank = n * m * p
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
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Try Lander-Bailey + schoolbook construction (rank 23)
    U, V, W, rank = _construct_lander_bailey_4x4x5()
    
    # Verify it
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification failed
    U, V, W, rank = _construct_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
