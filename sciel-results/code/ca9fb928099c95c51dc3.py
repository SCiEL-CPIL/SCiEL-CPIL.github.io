import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def strassen_2x2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    U, V, W are (4, 7) matrices such that for all i,j,k in {0,1}:
    sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A11+A22)(B11+B22) -> C11, C22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21, C22
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A11(B21-B22) -> C12, C22
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A22(B11-B12) -> C11, C21
    U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A11+A12)B22 -> C11, C12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin within 120s limit
    
    # Start with schoolbook (rank 60)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try Strassen-based block decomposition
    # Split the 3x4x5 multiplication into blocks
    # Strategy: split m=4 into 2+2, use Strassen for 2x2x2 blocks
    # This gives us a structured improvement over pure schoolbook
    
    # Block decomposition approach:
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    # Split B (4x5) into B1 (2x5) and B2 (2x5)
    # C = A1*B1 + A2*B2
    
    # For each 3x2x5 subproblem, we can use schoolbook (rank 30)
    # But we can potentially do better with structured approaches
    
    # Let's try a mixed approach: use Strassen for the 2x2 parts
    # and schoolbook for the rest
    
    # Actually, let me try a direct construction using Strassen blocks
    
    # Split n=3 into 1+2, m=4 into 2+2, p=5 into 2+3
    # This gives us 8 subproblems of various sizes
    
    # For the 2x2x2 subproblem, use Strassen (rank 7 instead of 8)
    # For other subproblems, use schoolbook
    
    # Let's implement this systematically
    
    # Define block boundaries
    n_blocks = [(0, 1), (1, 3)]  # n=3 split into 1 and 2
    m_blocks = [(0, 2), (2, 4)]  # m=4 split into 2 and 2
    p_blocks = [(0, 2), (2, 5)]  # p=5 split into 2 and 3
    
    # For each combination of (n_block, m_block, p_block), 
    # we have a subproblem of size (ni x mj x pk)
    # If it's 2x2x2, use Strassen; otherwise use schoolbook
    
    total_rank = 0
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for (ni_start, ni_end) in n_blocks:
        for (mj_start, mj_end) in m_blocks:
            for (pk_start, pk_end) in p_blocks:
                ni = ni_end - ni_start
                mj = mj_end - mj_start
                pk = pk_end - pk_start
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    Us, Vs, Ws = strassen_2x2x2()
                    block_rank = 7
                else:
                    # Use schoolbook
                    block_rank = ni * mj * pk
                    Us = np.zeros((ni * mj, block_rank), dtype=float)
                    Vs = np.zeros((mj * pk, block_rank), dtype=float)
                    Ws = np.zeros((pk * ni, block_rank), dtype=float)
                    c = 0
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                Us[ii * mj + jj, c] = 1
                                Vs[jj * pk + kk, c] = 1
                                Ws[kk * ni + ii, c] = 1
                                c += 1
                
                total_rank += block_rank
                U_blocks.append((ni_start, mj_start, Us))
                V_blocks.append((mj_start, pk_start, Vs))
                W_blocks.append((pk_start, ni_start, Ws))
    
    # Assemble the full matrices
    U_new = np.zeros((n * m, total_rank), dtype=float)
    V_new = np.zeros((m * p, total_rank), dtype=float)
    W_new = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    for idx in range(len(U_blocks)):
        ni_start, mj_start, Us = U_blocks[idx]
        mj_start_v, pk_start_v, Vs = V_blocks[idx]
        pk_start_w, ni_start_w, Ws = W_blocks[idx]
        
        ni = Us.shape[0] // Us.shape[1] if Us.shape[1] > 0 else 0
        # Get block dimensions from the block sizes
        # We need to know the actual ni, mj, pk for this block
        # Let's recompute from the block definitions
        
        # Actually, let's redo this more carefully
        pass
    
    # Let me redo the assembly more carefully
    U_new = np.zeros((n * m, total_rank), dtype=float)
    V_new = np.zeros((m * p, total_rank), dtype=float)
    W_new = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    for (ni_start, ni_end) in n_blocks:
        for (mj_start, mj_end) in m_blocks:
            for (pk_start, pk_end) in p_blocks:
                ni = ni_end - ni_start
                mj = mj_end - mj_start
                pk = pk_end - pk_start
                
                if ni == 2 and mj == 2 and pk == 2:
                    Us, Vs, Ws = strassen_2x2x2()
                    block_rank = 7
                else:
                    block_rank = ni * mj * pk
                    Us = np.zeros((ni * mj, block_rank), dtype=float)
                    Vs = np.zeros((mj * pk, block_rank), dtype=float)
                    Ws = np.zeros((pk * ni, block_rank), dtype=float)
                    c = 0
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                Us[ii * mj + jj, c] = 1
                                Vs[jj * pk + kk, c] = 1
                                Ws[kk * ni + ii, c] = 1
                                c += 1
                
                # Place into global matrices
                for r in range(block_rank):
                    for ii in range(ni):
                        U_new[(ni_start + ii) * m + (mj_start), col_offset + r] = Us[ii * mj + 0, r] if mj == 1 else 0
                        # This is getting complicated. Let me use a simpler approach.
                
                # Simpler: just place the blocks directly
                U_block_start_row = ni_start * m + mj_start
                U_block_end_row = (ni_start + ni) * m + mj_start
                # Wait, this isn't right either. The row index is i*m + j
                # So for block (ni_start..ni_end, mj_start..mj_end):
                # rows are (ni_start + ii)*m + (mj_start + jj) for ii in [0,ni), jj in [0,mj)
                
                for r in range(block_rank):
                    for ii in range(ni):
                        for jj in range(mj):
                            U_new[(ni_start + ii) * m + (mj_start + jj), col_offset + r] = Us[ii * mj + jj, r]
                
                for r in range(block_rank):
                    for jj in range(mj):
                        for kk in range(pk):
                            V_new[(mj_start + jj) * p + (pk_start + kk), col_offset + r] = Vs[jj * pk + kk, r]
                
                for r in range(block_rank):
                    for kk in range(pk):
                        for ii in range(ni):
                            W_new[(pk_start + kk) * n + (ni_start + ii), col_offset + r] = Ws[kk * ni + ii, r]
                
                col_offset += block_rank
    
    # Verify the block decomposition
    if col_offset == total_rank:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_new[i * m + j, :], V_new[j * p + k, :] * W_new[k * n + i, :])
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid and total_rank < best_rank:
            best_rank = total_rank
            best_U, best_V, best_W = U_new, V_new, W_new
    
    # Now try random search for ranks below best_rank
    for r in range(best_rank - 1, max(best_rank - 15, 1), -1):
        if time.time() > deadline - 20:
            break
        
        attempts = 3
        for _ in range(attempts):
            if time.time() > deadline - 20:
                break
            
            U_r = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n * m, r))
            V_r = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m * p, r))
            W_r = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p * n, r))
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        if best_rank == r:
            break
    
    # Convert to lists with integer/half-integer values
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
