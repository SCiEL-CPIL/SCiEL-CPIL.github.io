import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p, i_off, j_off, k_off):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[(i + i_off) * m + (j + j_off), col] = 1
                    V[(j + j_off) * p + (k + k_off), col] = 1
                    W[(k + k_off) * n + (i + i_off), col] = 1
                    col += 1
        return U, V, W, rank

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_2x2x2(i_off, j_off, k_off):
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00+A10)B11
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[3, 4] = 1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[2, 6] = 1; W[0, 6] = 1
        
        # Note: The indices in U, V, W are local to the 2x2x2 block.
        # U row: i*2 + j. i, j in {0,1}.
        # V row: j*2 + k. j, k in {0,1}.
        # W row: k*2 + i. k, i in {0,1}.
        
        # We need to shift these to global indices.
        # But the caller will handle the shifting by slicing or adding offsets.
        # Actually, it's easier to return local matrices and let the caller place them.
        # However, the schoolbook helper above uses offsets directly.
        # Let's make strassen return local matrices (0-indexed) and handle offset in the recursive function.
        
        return U, V, W, rank

    def decompose(n, m, p, i_off, j_off, k_off):
        """
        Returns U, V, W matrices for the sub-problem n x m x p 
        starting at global offsets i_off, j_off, k_off.
        The matrices U, V, W will have the correct global row indices.
        """
        if n == 2 and m == 2 and p == 2:
            U_loc, V_loc, W_loc, r = strassen_2x2x2(0, 0, 0)
            # Shift rows to global indices
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # U local row i*2+j -> global (i+i_off)*m + (j+j_off)
            for i in range(n):
                for j in range(m):
                    local_row = i * 2 + j
                    global_row = (i + i_off) * m + (j + j_off)
                    U[global_row, :] = U_loc[local_row, :]
            
            # V local row j*2+k -> global (j+j_off)*p + (k+k_off)
            for j in range(m):
                for k in range(p):
                    local_row = j * 2 + k
                    global_row = (j + j_off) * p + (k + k_off)
                    V[global_row, :] = V_loc[local_row, :]
            
            # W local row k*2+i -> global (k+k_off)*n + (i+i_off)
            for k in range(p):
                for i in range(n):
                    local_row = k * 2 + i
                    global_row = (k + k_off) * n + (i + i_off)
                    W[global_row, :] = W_loc[local_row, :]
                    
            return U, V, W, r

        if n == 2 and m == 1 and p == 2:
            return schoolbook(n, m, p, i_off, j_off, k_off)
            
        if n == 2 and m == 5 and p == 1:
            return schoolbook(n, m, p, i_off, j_off, k_off)

        if n == 4 and m == 5 and p == 6:
            # Split into 4 blocks of 2x5x3
            U_list = []
            V_list = []
            W_list = []
            total_rank = 0
            
            blocks = [
                (0, 0, 0),
                (0, 0, 3),
                (2, 0, 0),
                (2, 0, 3)
            ]
            
            for bi, bj, bk in blocks:
                U_b, V_b, W_b, r_b = decompose(2, 5, 3, i_off + bi, j_off + bj, k_off + bk)
                U_list.append(U_b)
                V_list.append(V_b)
                W_list.append(W_b)
                total_rank += r_b
                
            U = np.hstack(U_list)
            V = np.hstack(V_list)
            W = np.hstack(W_list)
            return U, V, W, total_rank

        if n == 2 and m == 5 and p == 3:
            # Split into 2x5x2 and 2x5x1
            U_list = []
            V_list = []
            W_list = []
            total_rank = 0
            
            # Block 1: 2x5x2
            U_b1, V_b1, W_b1, r_b1 = decompose(2, 5, 2, i_off, j_off, k_off)
            U_list.append(U_b1)
            V_list.append(V_b1)
            W_list.append(W_b1)
            total_rank += r_b1
            
            # Block 2: 2x5x1
            U_b2, V_b2, W_b2, r_b2 = decompose(2, 5, 1, i_off, j_off, k_off + 2)
            U_list.append(U_b2)
            V_list.append(V_b2)
            W_list.append(W_b2)
            total_rank += r_b2
            
            U = np.hstack(U_list)
            V = np.hstack(V_list)
            W = np.hstack(W_list)
            return U, V, W, total_rank

        if n == 2 and m == 5 and p == 2:
            # Split into 2x2x2 and 2x3x2
            U_list = []
            V_list = []
            W_list = []
            total_rank = 0
            
            # Block 1: 2x2x2
            U_b1, V_b1, W_b1, r_b1 = decompose(2, 2, 2, i_off, j_off, k_off)
            U_list.append(U_b1)
            V_list.append(V_b1)
            W_list.append(W_b1)
            total_rank += r_b1
            
            # Block 2: 2x3x2
            U_b2, V_b2, W_b2, r_b2 = decompose(2, 3, 2, i_off, j_off + 2, k_off)
            U_list.append(U_b2)
            V_list.append(V_b2)
            W_list.append(W_b2)
            total_rank += r_b2
            
            U = np.hstack(U_list)
            V = np.hstack(V_list)
            W = np.hstack(W_list)
            return U, V, W, total_rank

        if n == 2 and m == 3 and p == 2:
            # Split into 2x2x2 and 2x1x2
            U_list = []
            V_list = []
            W_list = []
            total_rank = 0
            
            # Block 1: 2x2x2
            U_b1, V_b1, W_b1, r_b1 = decompose(2, 2, 2, i_off, j_off, k_off)
            U_list.append(U_b1)
            V_list.append(V_b1)
            W_list.append(W_b1)
            total_rank += r_b1
            
            # Block 2: 2x1x2
            U_b2, V_b2, W_b2, r_b2 = decompose(2, 1, 2, i_off, j_off + 2, k_off)
            U_list.append(U_b2)
            V_list.append(V_b2)
            W_list.append(W_b2)
            total_rank += r_b2
            
            U = np.hstack(U_list)
            V = np.hstack(V_list)
            W = np.hstack(W_list)
            return U, V, W, total_rank

        # Fallback: Schoolbook
        return schoolbook(n, m, p, i_off, j_off, k_off)

    U, V, W, rank = decompose(n, m, p, 0, 0, 0)
    
    # Convert to lists of lists as required by interface? 
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't explicitly demand list of lists, but the seed program returns list of lists.
    # "Keep the function names and signatures of the seed program."
    # Seed program returns {"U": U, "V": V, "W": W} where U, V, W are list of lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
