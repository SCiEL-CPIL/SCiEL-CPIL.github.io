import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110

    def compute_error(U, V, W, r):
        """Compute max error of the trilinear constraint."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def add_strassen_block(U, V, W, offset, ai, aj, bi, bj, ki_start, ki_end):
        """
        Add a Strassen rank-7 decomposition for a 2x2x2 sub-problem.
        ai, aj: row indices in A (i dimension)
        bi, bj: col indices in B inner dim (j dimension)  
        ki_start to ki_end-1: col indices in B output (k dimension), must be 2 consecutive
        offset: starting column index in the factor matrices
        """
        # A[i, j] where i in {ai, ai+1}, j in {bi, bi+1}
        # B[j, k] where j in {bi, bi+1}, k in {ki_start, ki_start+1}
        # C[i, k] where i in {ai, ai+1}, k in {ki_start, ki_start+1}
        
        # Strassen's 7 products
        # P1 = A[0,0] * (B[1,1] - B[0,1])
        # P2 = (A[0,0] + A[0,1]) * B[0,1]
        # P3 = (A[1,0] + A[1,1]) * B[0,0]
        # P4 = A[1,1] * (B[0,0] - B[1,0])
        # P5 = (A[0,0] + A[1,1]) * (B[0,1] + B[1,1])
        # P6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        # P7 = (A[0,0] - A[1,0]) * (B[0,0] + B[0,1])
        
        # C[0,0] = P5 + P4 - P2 + P6
        # C[0,1] = P1 + P5
        # C[1,0] = P3 + P7
        # C[1,1] = P1 + P4 - P3 + P5
        
        # For each product, we add a rank-1 term
        # Product c: U[i,j] * V[j,k] * W[k,i]
        
        # P1: A[ai,bi] * (B[bi+1,ki_start+1] - B[bi,ki_start+1])
        # U[ai,bi] = 1, V[bi+1,ki_start+1] = 1, V[bi,ki_start+1] = -1, W[ki_start+1,ai] = 1
        U[ai*m+bi, offset] += 1
        V[(bi+1)*p+(ki_start+1), offset] += 1
        V[bi*p+(ki_start+1), offset] += -1
        W[(ki_start+1)*n+ai, offset] += 1
        
        # P2: (A[ai,bi] + A[ai,bi+1]) * B[bi,ki_start+1]
        U[ai*m+bi, offset+1] += 1
        U[ai*m+(bi+1), offset+1] += 1
        V[bi*p+(ki_start+1), offset+1] += 1
        W[(ki_start+1)*n+ai, offset+1] += 1
        
        # P3: (A[ai+1,bi] + A[ai+1,bi+1]) * B[bi,ki_start]
        U[(ai+1)*m+bi, offset+2] += 1
        U[(ai+1)*m+(bi+1), offset+2] += 1
        V[bi*p+ki_start, offset+2] += 1
        W[ki_start*n+(ai+1), offset+2] += 1
        
        # P4: A[ai+1,bi+1] * (B[bi,ki_start] - B[bi+1,ki_start])
        U[(ai+1)*m+(bi+1), offset+3] += 1
        V[bi*p+ki_start, offset+3] += 1
        V[(bi+1)*p+ki_start, offset+3] += -1
        W[ki_start*n+(ai+1), offset+3] += 1
        
        # P5: (A[ai,bi] + A[ai+1,bi+1]) * (B[bi,ki_start+1] + B[bi+1,ki_start+1])
        U[ai*m+bi, offset+4] += 1
        U[(ai+1)*m+(bi+1), offset+4] += 1
        V[bi*p+(ki_start+1), offset+4] += 1
        V[(bi+1)*p+(ki_start+1), offset+4] += 1
        W[(ki_start+1)*n+ai, offset+4] += 1
        
        # P6: (A[ai,bi+1] - A[ai+1,bi+1]) * (B[bi+1,ki_start] + B[bi+1,ki_start+1])
        U[ai*m+(bi+1), offset+5] += 1
        U[(ai+1)*m+(bi+1), offset+5] += -1
        V[(bi+1)*p+ki_start, offset+5] += 1
        V[(bi+1)*p+(ki_start+1), offset+5] += 1
        W[ki_start*n+ai, offset+5] += 1
        W[(ki_start+1)*n+ai, offset+5] += 1
        
        # P7: (A[ai,bi] - A[ai+1,bi]) * (B[bi,ki_start] + B[bi,ki_start+1])
        U[ai*m+bi, offset+6] += 1
        U[(ai+1)*m+bi, offset+6] += -1
        V[bi*p+ki_start, offset+6] += 1
        V[bi*p+(ki_start+1), offset+6] += 1
        W[ki_start*n+ai, offset+6] += 1
        W[(ki_start+1)*n+ai, offset+6] += 1
        
        return offset + 7

    def add_schoolbook_2x2x1(U, V, W, offset, ai, aj, bi, ki):
        """
        Add schoolbook rank-4 decomposition for 2x2x1 sub-problem.
        A[ai:ai+2, aj:aj+2] * B[aj:aj+2, ki] = C[ai:ai+2, ki]
        """
        for di in range(2):
            for dj in range(2):
                U[(ai+di)*m+(aj+dj), offset] += 1
                V[(aj+dj)*p+ki, offset] += 1
                W[ki*n+(ai+di), offset] += 1
                offset += 1
        return offset

    # Build the rank-36 decomposition
    # Split: A is 2x4, split inner dim into 2+2: A = [A1 | A2]
    # B is 4x5, split into B_top (2x5) and B_bottom (2x5)
    # C = A1*B_top + A2*B_bottom
    # For each 2x2 * 2x5: split 5 cols into 4+1
    # For the 4 cols: split into 2+2, each 2x2*2x2 uses Strassen (rank 7)
    # For the 1 col: 2x2*2x1 uses schoolbook (rank 4)
    
    # Total: 2 (top/bottom) * (2*7 + 4) = 2 * 18 = 36
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    offset = 0
    
    # A1 * B_top: A rows 0-1, B inner rows 0-1, output cols 0-4
    # Split output cols: 0-3 (two 2x2 blocks) and 4 (2x1 block)
    
    # A1 * B_top[:, 0:2] -> C[:, 0:2]
    offset = add_strassen_block(U, V, W, offset, 0, 0, 0, 1, 0, 2)
    
    # A1 * B_top[:, 2:4] -> C[:, 2:4]
    offset = add_strassen_block(U, V, W, offset, 0, 0, 0, 1, 2, 4)
    
    # A1 * B_top[:, 4] -> C[:, 4]
    offset = add_schoolbook_2x2x1(U, V, W, offset, 0, 0, 0, 4)
    
    # A2 * B_bottom: A rows 0-1, B inner rows 2-3, output cols 0-4
    
    # A2 * B_bottom[:, 0:2] -> C[:, 0:2]
    offset = add_strassen_block(U, V, W, offset, 0, 2, 2, 3, 0, 2)
    
    # A2 * B_bottom[:, 2:4] -> C[:, 2:4]
    offset = add_strassen_block(U, V, W, offset, 0, 2, 2, 3, 2, 4)
    
    # A2 * B_bottom[:, 4] -> C[:, 4]
    offset = add_schoolbook_2x2x1(U, V, W, offset, 0, 2, 2, 4)
    
    # Verify
    err = compute_error(U, V, W, rank)
    if err < 1e-10:
        # Convert to list with integer values
        U_list = [[int(U[i,j]) if abs(U[i,j] - round(U[i,j])) < 1e-10 else float(U[i,j]) for j in range(rank)] for i in range(n*m)]
        V_list = [[int(V[i,j]) if abs(V[i,j] - round(V[i,j])) < 1e-10 else float(V[i,j]) for j in range(rank)] for i in range(m*p)]
        W_list = [[int(W[i,j]) if abs(W[i,j] - round(W[i,j])) < 1e-10 else float(W[i,j]) for j in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook rank 40
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    
    U_list = [[int(U[i,j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[int(V[i,j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[int(W[i,j]) for j in range(rank)] for i in range(p*n)]
    return {"U": U_list, "V": V_list, "W": W_list}
