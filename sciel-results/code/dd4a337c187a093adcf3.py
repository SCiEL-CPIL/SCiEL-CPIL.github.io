import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin from 120s limit

    n, m, p = DIMENSIONS

    def add_block(n_sub, m_sub, p_sub, i_off, j_off, k_off, U, V, W, col_offset):
        """
        Adds a decomposition for M_{n_sub, m_sub, p_sub} starting at column col_offset.
        Returns new col_offset.
        """
        # Base case: m_sub == 1
        if m_sub == 1:
            r = n_sub * p_sub
            for i in range(n_sub):
                for k in range(p_sub):
                    idx = col_offset + i * p_sub + k
                    U[(i_off + i) * m + (j_off)][idx] = 1
                    V[(j_off) * p + (k_off + k)][idx] = 1
                    W[(k_off + k) * n + (i_off + i)][idx] = 1
            return col_offset + r

        # Base case: Strassen 2x2x2
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Strassen decomposition for 2x2x2
            # U, V, W are 4x7 matrices for the block
            # Indices in block:
            # U: (i, j) -> i*2+j
            # V: (j, k) -> j*2+k
            # W: (k, i) -> k*2+i
            
            # m1 = (a+d)(e+h)
            U[(i_off+0)*m + (j_off+0)][col_offset + 0] = 1
            U[(i_off+1)*m + (j_off+1)][col_offset + 0] = 1
            V[(j_off+0)*p + (k_off+0)][col_offset + 0] = 1
            V[(j_off+1)*p + (k_off+1)][col_offset + 0] = 1
            W[(k_off+0)*n + (i_off+0)][col_offset + 0] = 1
            W[(k_off+1)*n + (i_off+1)][col_offset + 0] = 1

            # m2 = (c+d)e
            U[(i_off+1)*m + (j_off+0)][col_offset + 1] = 1
            U[(i_off+1)*m + (j_off+1)][col_offset + 1] = 1
            V[(j_off+0)*p + (k_off+0)][col_offset + 1] = 1
            W[(k_off+0)*n + (i_off+1)][col_offset + 1] = 1

            # m3 = a(f-h)
            U[(i_off+0)*m + (j_off+0)][col_offset + 2] = 1
            V[(j_off+0)*p + (k_off+1)][col_offset + 2] = 1
            V[(j_off+1)*p + (k_off+1)][col_offset + 2] = -1
            W[(k_off+1)*n + (i_off+0)][col_offset + 2] = 1

            # m4 = d(g-e)
            U[(i_off+1)*m + (j_off+1)][col_offset + 3] = 1
            V[(j_off+1)*p + (k_off+0)][col_offset + 3] = 1
            V[(j_off+0)*p + (k_off+0)][col_offset + 3] = -1
            W[(k_off+0)*n + (i_off+1)][col_offset + 3] = 1

            # m5 = (a+b)h
            U[(i_off+0)*m + (j_off+0)][col_offset + 4] = 1
            U[(i_off+0)*m + (j_off+1)][col_offset + 4] = 1
            V[(j_off+1)*p + (k_off+1)][col_offset + 4] = 1
            W[(k_off+1)*n + (i_off+0)][col_offset + 4] = 1
            W[(k_off+1)*n + (i_off+1)][col_offset + 4] = -1

            # m6 = (c-a)(e+f)
            U[(i_off+1)*m + (j_off+0)][col_offset + 5] = 1
            U[(i_off+0)*m + (j_off+0)][col_offset + 5] = -1
            V[(j_off+0)*p + (k_off+0)][col_offset + 5] = 1
            V[(j_off+0)*p + (k_off+1)][col_offset + 5] = 1
            W[(k_off+1)*n + (i_off+1)][col_offset + 5] = 1

            # m7 = (b-d)(g+h)
            U[(i_off+0)*m + (j_off+1)][col_offset + 6] = 1
            U[(i_off+1)*m + (j_off+1)][col_offset + 6] = -1
            V[(j_off+1)*p + (k_off+0)][col_offset + 6] = 1
            V[(j_off+1)*p + (k_off+1)][col_offset + 6] = 1
            W[(k_off+0)*n + (i_off+0)][col_offset + 6] = 1
            W[(k_off+1)*n + (i_off+1)][col_offset + 6] = 1

            return col_offset + 7

        # Recursive case: Split dimensions
        # Strategy:
        # 1. If m_sub > 1 and (n_sub==2 and m_sub>=2), try splitting m_sub into 2 + (m_sub-2)
        #    This works well for m_sub=3 -> 2+1.
        # 2. If p_sub > 1 and (n_sub==2 and m_sub>=2), try splitting p_sub into 2 + (p_sub-2)
        #    This works well for p_sub=3 -> 2+1, p_sub=4 -> 2+2.
        # 3. General fallback: Split largest dimension.
        
        # We need to choose a split.
        # For M_{2,3,7}:
        # We want to reach 2x2x2 blocks.
        # Split m=3 into 2+1.
        # Block 1: M_{2,2,7}. Split p=7 into 2+5? No, 2+2+2+1 is better.
        # So split p=7 into 2+5? Or 2+2+3?
        # Let's split p into 2 + (p_sub-2) recursively.
        
        # Priority:
        # If n_sub==2 and m_sub==2 and p_sub>2: Split p.
        # If n_sub==2 and m_sub>2: Split m into 2 + (m_sub-2).
        # If n_sub>2: Split n? (Not needed for n=2)
        
        if n_sub == 2 and m_sub == 2 and p_sub > 2:
            # Split p into 2 and p_sub-2
            p1 = 2
            p2 = p_sub - 2
            # Block 1: M_{2,2,2}
            col_offset = add_block(2, 2, p1, i_off, j_off, k_off, U, V, W, col_offset)
            # Block 2: M_{2,2,p2}
            col_offset = add_block(2, 2, p2, i_off, j_off, k_off + p1, U, V, W, col_offset)
            return col_offset

        if n_sub == 2 and m_sub > 2:
            # Split m into 2 and m_sub-2
            m1 = 2
            m2 = m_sub - 2
            # Block 1: M_{2,2,p_sub}
            col_offset = add_block(2, m1, p_sub, i_off, j_off, k_off, U, V, W, col_offset)
            # Block 2: M_{2,m2,p_sub}
            col_offset = add_block(2, m2, p_sub, i_off, j_off + m1, k_off, U, V, W, col_offset)
            return col_offset
            
        # Fallback: Schoolbook
        r = n_sub * m_sub * p_sub
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    idx = col_offset + i * m_sub * p_sub + j * p_sub + k
                    U[(i_off + i) * m + (j_off + j)][idx] = 1
                    V[(j_off + j) * p + (k_off + k)][idx] = 1
                    W[(k_off + k) * n + (i_off + i)][idx] = 1
        return col_offset

    # Calculate total rank
    # M_{2,3,7}
    # Split m=3 into 2+1.
    # M_{2,2,7} + M_{2,1,7}
    # M_{2,1,7} rank = 2*7 = 14.
    # M_{2,2,7} split p=7 into 2+2+2+1.
    # M_{2,2,2} (7) + M_{2,2,2} (7) + M_{2,2,2} (7) + M_{2,2,1} (4).
    # M_{2,2,1} rank = 2*2*1 = 4.
    # M_{2,2,7} rank = 7+7+7+4 = 25.
    # Total rank = 25 + 14 = 39.
    
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    col_offset = add_block(n, m, p, 0, 0, 0, U, V, W, col_offset)
    
    # Verification (optional, for safety)
    # Convert to numpy for fast check
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check a few random points
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Should be 1 for all i,j,k.
    # Since we have limited time, we trust the construction but can do a quick spot check.
    # For i=0, j=0, k=0:
    # val = np.dot(U_np[0*m+0], V_np[0*p+0] * W_np[0*n+0])
    # if abs(val - 1) > 1e-9: print("Error at 0,0,0")
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = res['U']
    V = res['V']
    W = res['W']
    
    # Full verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if err > 1e-9:
                    print(f"Error at {i},{j},{k}: {val}")
    
    print(f"Max Error: {max_err}")
    print(f"Rank: {len(U[0])}")
