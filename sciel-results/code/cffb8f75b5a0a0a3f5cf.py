import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 115.0  # Safety margin before 120s limit

    n, m, p = 2, 3, 8
    
    # Decompose M_{2,3,8} into 4 blocks of M_{2,3,2}
    # Each M_{2,3,2} has rank 11 (Strassen-7 for 2x2x2 core + 4 rank-1 terms)
    # Total rank = 4 * 11 = 44
    
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44
    
    # U: (n*m, rank) = (6, 44)
    # V: (m*p, rank) = (24, 44)
    # W: (p*n, rank) = (16, 44)
    U = np.zeros((n*m, total_rank), dtype=np.int8)
    V = np.zeros((m*p, total_rank), dtype=np.int8)
    W = np.zeros((p*n, total_rank), dtype=np.int8)
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        # Block indices:
        # A rows: 0, 1 (global U indices: 0-2, 3-5)
        # B cols: 2*blk, 2*blk+1 (global V indices: 2*blk, 2*blk+1 for row 0; p+... for row 1; 2p+... for row 2)
        # C cols: 2*blk, 2*blk+1 (global W indices: (2*blk)*n + i, (2*blk+1)*n + i)
        
        # W index helper: W[k*n + i, r] where k is col index, i is row index
        # For block blk, k is 2*blk or 2*blk+1
        # i is 0 or 1
        # W_idx(i, k) = k*2 + i
        
        # --- Strassen Part (7 terms) for A[:, 0:2] * B[0:2, :] ---
        # Let A_core be 2x2 (rows 0,1 of A; cols 0,1 of A)
        # Let B_core be 2x2 (rows 0,1 of B; cols 2*blk, 2*blk+1 of B)
        
        # m1 = (A00+A11)(B00+B11)
        # U: A[0,0]=1, A[1,1]=1 -> U[0], U[4]
        # V: B[0, 2*blk]=1, B[1, 2*blk+1]=1 -> V[2*blk], V[p + 2*blk+1]
        # W: C[0, 2*blk]=1, C[1, 2*blk+1]=1 -> W[2*(2*blk)+0], W[2*(2*blk+1)+1]
        U[0, base+0] = 1; U[4, base+0] = 1
        V[2*blk, base+0] = 1; V[p + 2*blk+1, base+0] = 1
        W[4*blk, base+0] = 1; W[4*blk+3, base+0] = 1
        
        # m2 = (A10+A11)B00
        # U: A[1,0]=1, A[1,1]=1 -> U[3], U[4]
        # V: B[0, 2*blk]=1 -> V[2*blk]
        # W: C[1, 2*blk]=1, C[1, 2*blk+1]=-1 -> W[4*blk+1], W[4*blk+3]
        U[3, base+1] = 1; U[4, base+1] = 1
        V[2*blk, base+1] = 1
        W[4*blk+1, base+1] = 1; W[4*blk+3, base+1] = -1
        
        # m3 = A00(B01-B11)
        # U: A[0,0]=1 -> U[0]
        # V: B[0, 2*blk+1]=1, B[1, 2*blk+1]=-1 -> V[2*blk+1], V[p + 2*blk+1]
        # W: C[0, 2*blk+1]=1, C[1, 2*blk+1]=1 -> W[4*blk+2], W[4*blk+3]
        U[0, base+2] = 1
        V[2*blk+1, base+2] = 1; V[p + 2*blk+1, base+2] = -1
        W[4*blk+2, base+2] = 1; W[4*blk+3, base+2] = 1
        
        # m4 = A11(B10-B00)
        # U: A[1,1]=1 -> U[4]
        # V: B[1, 2*blk]=1, B[0, 2*blk]=-1 -> V[p + 2*blk], V[2*blk]
        # W: C[0, 2*blk]=1, C[1, 2*blk]=1 -> W[4*blk], W[4*blk+1]
        U[4, base+3] = 1
        V[p + 2*blk, base+3] = 1; V[2*blk, base+3] = -1
        W[4*blk, base+3] = 1; W[4*blk+1, base+3] = 1
        
        # m5 = (A00+A01)B11
        # U: A[0,0]=1, A[0,1]=1 -> U[0], U[1]
        # V: B[1, 2*blk+1]=1 -> V[p + 2*blk+1]
        # W: C[0, 2*blk]=-1, C[0, 2*blk+1]=1 -> W[4*blk], W[4*blk+2]
        U[0, base+4] = 1; U[1, base+4] = 1
        V[p + 2*blk+1, base+4] = 1
        W[4*blk, base+4] = -1; W[4*blk+2, base+4] = 1
        
        # m6 = (A10-A00)(B00+B01)
        # U: A[1,0]=1, A[0,0]=-1 -> U[3], U[0]
        # V: B[0, 2*blk]=1, B[0, 2*blk+1]=1 -> V[2*blk], V[2*blk+1]
        # W: C[1, 2*blk+1]=1 -> W[4*blk+3]
        U[3, base+5] = 1; U[0, base+5] = -1
        V[2*blk, base+5] = 1; V[2*blk+1, base+5] = 1
        W[4*blk+3, base+5] = 1
        
        # m7 = (A01-A11)(B10+B11)
        # U: A[0,1]=1, A[1,1]=-1 -> U[1], U[4]
        # V: B[1, 2*blk]=1, B[1, 2*blk+1]=1 -> V[p + 2*blk], V[p + 2*blk+1]
        # W: C[0, 2*blk]=1 -> W[4*blk]
        U[1, base+6] = 1; U[4, base+6] = -1
        V[p + 2*blk, base+6] = 1; V[p + 2*blk+1, base+6] = 1
        W[4*blk, base+6] = 1
        
        # --- Extra Part (4 terms) for A[:, 2] * B[2, :] ---
        # C[i, k] += A[i, 2] * B[2, k]
        # U: A[i, 2] -> U[i*3 + 2]
        # V: B[2, k] -> V[2*p + k]
        # W: C[i, k] -> W[k*2 + i]
        
        # Term 8: C[0, 2*blk] += A[0,2]*B[2, 2*blk]
        U[0*3+2, base+7] = 1
        V[2*p + 2*blk, base+7] = 1
        W[4*blk + 0, base+7] = 1
        
        # Term 9: C[0, 2*blk+1] += A[0,2]*B[2, 2*blk+1]
        U[0*3+2, base+8] = 1
        V[2*p + (2*blk+1), base+8] = 1
        W[4*blk + 2, base+8] = 1
        
        # Term 10: C[1, 2*blk] += A[1,2]*B[2, 2*blk]
        U[1*3+2, base+9] = 1
        V[2*p + 2*blk, base+9] = 1
        W[4*blk + 1, base+9] = 1
        
        # Term 11: C[1, 2*blk+1] += A[1,2]*B[2, 2*blk+1]
        U[1*3+2, base+10] = 1
        V[2*p + (2*blk+1), base+10] = 1
        W[4*blk + 3, base+10] = 1

    if time.time() > _deadline:
        # Should never happen, but if it does, return whatever is constructed so far
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
