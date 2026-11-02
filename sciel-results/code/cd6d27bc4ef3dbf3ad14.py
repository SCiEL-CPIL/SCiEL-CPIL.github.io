import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def verify(U, V, W, n, m, p, atol=1e-8):
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

    def schoolbook():
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
        return U, V, W

    def strassen_2x2x2():
        """Rank-7 Strassen for 2x2x2."""
        rank = 7
        U = np.zeros((4, rank), dtype=float)
        V = np.zeros((4, rank), dtype=float)
        W = np.zeros((4, rank), dtype=float)
        # A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
        # B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
        # C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,2,3 (W[k*n+i])
        
        # M1 = (A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # M2 = A22*(B21-B11)
        U[3,1]=1; V[2,1]=1; V[0,1]=-1; W[2,1]=1; W[3,1]=1
        # M3 = A11*(B22-B12)
        U[0,2]=1; V[3,2]=1; V[1,2]=-1; W[1,2]=1; W[2,2]=1
        # M4 = (A11+A12)B22
        U[0,3]=1; U[1,3]=1; V[3,3]=1; W[0,3]=1; W[1,3]=1
        # M5 = (A21+A22)B11
        U[2,4]=1; U[3,4]=1; V[0,4]=1; W[2,4]=1; W[3,4]=1
        # M6 = (A12-A22)(B21+B22)
        U[1,5]=1; U[3,5]=-1; V[2,5]=1; V[3,5]=1; W[1,5]=1; W[3,5]=1
        # M7 = (A21-A11)(B11+B12)
        U[2,6]=1; U[0,6]=-1; V[0,6]=1; V[1,6]=1; W[0,6]=1; W[2,6]=1
        return U, V, W

    def build_block_strassen():
        """8 blocks of 2x2x2, rank 56."""
        sU, sV, sW = strassen_2x2x2()
        rank = 56
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        for u in range(2):  # 2 blocks in A's row direction (2 rows -> 1 block of 2)
            for v in range(4):  # 4 blocks in B's column direction
                base = 7 * v
                # A block: rows 2u, 2u+1; cols 2v, 2v+1 -> but A is 2x4, so rows 0,1; cols 2v,2v+1
                a_rows = [0, 1]
                a_cols = [2*v, 2*v+1]
                # B block: rows 2v, 2v+1; cols 4v, 4v+1 -> B is 4x8, rows 2v,2v+1; cols 4v,4v+1
                b_rows = [2*v, 2*v+1]
                b_cols = [4*v, 4*v+1]
                # C block: rows 0,1; cols 4v,4v+1 -> C is 2x8
                c_rows = [0, 1]
                c_cols = [4*v, 4*v+1]
                
                for t in range(7):
                    col = base + t
                    for local_r in range(2):
                        for local_c in range(2):
                            a_idx = a_rows[local_r] * m + a_cols[local_c]
                            U[a_idx, col] = sU[local_r*2+local_c, t]
                    for local_r in range(2):
                        for local_c in range(2):
                            b_idx = b_rows[local_r] * p + b_cols[local_c]
                            V[b_idx, col] = sV[local_r*2+local_c, t]
                    for local_r in range(2):
                        for local_c in range(2):
                            c_idx = c_cols[local_c] * n + c_rows[local_r]
                            W[c_idx, col] = sW[local_r*2+local_c, t]
        return U, V, W

    # Try to find a lower rank decomposition
    # Strategy: try rank 40, 36, 32, etc.
    
    # First, get the baseline
    U_base, V_base, W_base = build_block_strassen()
    if not verify(U_base, V_base, W_base, n, m, p):
        U_base, V_base, W_base = schoolbook()
    
    best_rank = U_base.shape[1]
    best_U, best_V, best_W = U_base, V_base, W_base

    # Try to reduce rank by zeroing out columns and checking
    # This is a heuristic: try removing columns and see if it still works
    
    # More principled: use the tensor structure
    # The 2x4x8 multiplication tensor has a specific structure we can exploit
    
    # Let me try a direct construction with rank 40
    # Split B into two 4x4 halves: B = [B1 | B2]
    # C = A*B1 + A*B2 = [C1 | C2]
    # Each 2x4 x 4x4 -> 2x4
    # Embed 2x4x4 into 4x4x4 using rank-20
    
    # For now, return the Strassen result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    construct()
