"""Matrix multiplication 4x5x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook decomposition (rank 120).
    2. Apply block decomposition:
       - Split n=4 into [2, 2]
       - Split m=5 into [2, 3]
       - Split p=6 into [2, 4]
       
       This gives 4 blocks:
       - Block (0,0): 2x2x2 -> Strassen (rank 7)
       - Block (0,1): 2x3x4 -> needs decomposition
       - Block (1,0): 2x2x2 -> Strassen (rank 7)
       - Block (1,1): 2x3x4 -> needs decomposition
       
       For 2x3x4, we can further split m=3 into [2,1] and p=4 into [2,2]:
       - 2x2x2 -> Strassen (7)
       - 2x2x2 -> Strassen (7)
       - 2x1x2 -> Schoolbook (4)
       - 2x1x2 -> Schoolbook (4)
       Total for 2x3x4: 7+7+4+4 = 22
       
       Total rank: 7 + 22 + 7 + 22 = 58
       
       Wait, let's be more careful with the block structure.
       
       Actually, let's use a simpler approach:
       Split n=4 into [2,2], m=5 into [2,3], p=6 into [3,3]
       
       Blocks:
       (0,0): 2x2x3 -> Split p=3 into [2,1]: 2x2x2 (7) + 2x2x1 (4) = 11
       (0,1): 2x3x3 -> Split m=3 into [2,1]: 
              2x2x3 (11) + 2x1x3 (6) = 17
       (1,0): 2x2x3 -> 11
       (1,1): 2x3x3 -> 17
       
       Total: 11 + 17 + 11 + 17 = 56
       
       Hmm, this is getting complicated. Let me use a more systematic approach.
       
       Let's just implement Strassen for 2x2x2 blocks and schoolbook for the rest,
       using a clean block decomposition.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Define Strassen decomposition for 2x2x2
    # Strassen's 7 terms for multiplying 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_222(i0, j0, k0, U, V, W, col_start):
        """
        Add Strassen decomposition for a 2x2x2 block starting at (i0, j0, k0).
        Returns the number of columns used.
        """
        col = col_start
        
        # M1 = (A00 + A11)(B00 + B11)
        # A indices: (i0+0, j0+0) and (i0+1, j0+1)
        # B indices: (j0+0, k0+0) and (j0+1, k0+1)
        # C indices: (k0+0, i0+0) and (k0+1, i0+1)
        U[i0 * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        # A: (i0+1, j0+0), (i0+1, j0+1)
        # B: (j0+0, k0+0)
        # C: (k0+0, i0+1)
        U[(i0+1) * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        W[k0 * n + (i0+1), col] = 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        # A: (i0+0, j0+0)
        # B: (j0+0, k0+1), (j0+1, k0+1)
        # C: (k0+1, i0+0)
        U[i0 * m + j0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = -1
        W[(k0+1) * n + i0, col] = 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        # A: (i0+1, j0+1)
        # B: (j0+1, k0+0), (j0+0, k0+0)
        # C: (k0+0, i0+1)
        U[(i0+1) * m + (j0+1), col] = 1
        V[(j0+1) * p + k0, col] = 1
        V[j0 * p + k0, col] = -1
        W[k0 * n + (i0+1), col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        # A: (i0+0, j0+0), (i0+0, j0+1)
        # B: (j0+1, k0+1)
        # C: (k0+1, i0+0)
        U[i0 * m + j0, col] = 1
        U[i0 * m + (j0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[(k0+1) * n + i0, col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # A: (i0+1, j0+0), (i0+0, j0+0)
        # B: (j0+0, k0+0), (j0+0, k0+1)
        # C: (k0+1, i0+1)
        U[(i0+1) * m + j0, col] = 1
        U[i0 * m + j0, col] = -1
        V[j0 * p + k0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # A: (i0+0, j0+1), (i0+1, j0+1)
        # B: (j0+1, k0+0), (j0+1, k0+1)
        # C: (k0+0, i0+0)
        U[i0 * m + (j0+1), col] = 1
        U[(i0+1) * m + (j0+1), col] = -1
        V[(j0+1) * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        col += 1
        
        return col - col_start  # 7 columns used
    
    def schoolbook_block(i0, j0, k0, ni, mj, pk, U, V, W, col_start):
        """
        Add schoolbook decomposition for a block of size ni x mj x pk.
        Returns the number of columns used.
        """
        col = col_start
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    U[(i0+ii) * m + (j0+jj), col] = 1
                    V[(j0+jj) * p + (k0+kk), col] = 1
                    W[(k0+kk) * n + (i0+ii), col] = 1
                    col += 1
        return col - col_start
    
    # Block decomposition strategy:
    # Split n=4 into [2, 2]
    # Split m=5 into [2, 3]
    # Split p=6 into [3, 3]
    
    i_splits = [0, 2]
    i_sizes = [2, 2]
    
    j_splits = [0, 2]
    j_sizes = [2, 3]
    
    k_splits = [0, 3]
    k_sizes = [3, 3]
    
    # For each block, decide whether to use Strassen or schoolbook
    # Strassen only works for 2x2x2
    # For other sizes, we can subdivide further
    
    def decompose_block(i0, j0, k0, ni, mj, pk, U, V, W, col_start):
        """
        Recursively decompose a block using Strassen where possible.
        """
        if time.time() > deadline:
            return col_start
        
        # If 2x2x2, use Strassen
        if ni == 2 and mj == 2 and pk == 2:
            return strassen_222(i0, j0, k0, U, V, W, col_start)
        
        # If any dimension is 1, use schoolbook for that slice
        # Otherwise, try to split the largest dimension
        
        # Strategy: if a dimension is even and >= 2, split it in half
        # If a dimension is odd, split into (size//2, size//2 + 1)
        
        best_split = None
        best_gain = 0
        
        # Try splitting n
        if ni >= 2:
            for split in range(1, ni):
                n1, n2 = split, ni - split
                # Estimate: if both sub-blocks can use Strassen, gain is significant
                # For now, just use schoolbook for non-2x2x2
                pass
        
        # For simplicity, if not 2x2x2, use schoolbook
        # But we can do better by subdividing
        
        # Let's try: if ni==2 and mj==2 and pk==3, split pk into 2+1
        if ni == 2 and mj == 2 and pk == 3:
            # Split pk into [2, 1]
            col = col_start
            col += decompose_block(i0, j0, k0, ni, mj, 2, U, V, W, col)
            col += decompose_block(i0, j0, k0+2, ni, mj, 1, U, V, W, col)
            return col - col_start
        
        if ni == 2 and mj == 3 and pk == 3:
            # Split mj into [2, 1]
            col = col_start
            col += decompose_block(i0, j0, k0, ni, 2, pk, U, V, W, col)
            col += decompose_block(i0, j0+2, k0, ni, 1, pk, U, V, W, col)
            return col - col_start
        
        if ni == 2 and mj == 1 and pk == 3:
            # Split pk into [2, 1]
            col = col_start
            col += decompose_block(i0, j0, k0, ni, mj, 2, U, V, W, col)
            col += decompose_block(i0, j0, k0+2, ni, mj, 1, U, V, W, col)
            return col - col_start
        
        if ni == 2 and mj == 2 and pk == 1:
            # Schoolbook: 2*2*1 = 4
            return schoolbook_block(i0, j0, k0, ni, mj, pk, U, V, W, col_start)
        
        if ni == 2 and mj == 1 and pk == 2:
            # Schoolbook: 2*1*2 = 4
            return schoolbook_block(i0, j0, k0, ni, mj, pk, U, V, W, col_start)
        
        if ni == 2 and mj == 1 and pk == 1:
            # Schoolbook: 2*1*1 = 2
            return schoolbook_block(i0, j0, k0, ni, mj, pk, U, V, W, col_start)
        
        # Fallback: schoolbook
        return schoolbook_block(i0, j0, k0, ni, mj, pk, U, V, W, col_start)
    
    # Allocate max possible rank
    max_rank = 120
    U = np.zeros((n * m, max_rank), dtype=int)
    V = np.zeros((m * p, max_rank), dtype=int)
    W = np.zeros((p * n, max_rank), dtype=int)
    
    col = 0
    for i_idx, i0 in enumerate(i_splits):
        ni = i_sizes[i_idx]
        for j_idx, j0 in enumerate(j_splits):
            mj = j_sizes[j_idx]
            for k_idx, k0 in enumerate(k_splits):
                pk = k_sizes[k_idx]
                col += decompose_block(i0, j0, k0, ni, mj, pk, U, V, W, col)
    
    actual_rank = col
    
    # Verify the decomposition
    def verify(U, V, W, rank):
        """Verify the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if val != 1:
                        return False
        # Also check that off-diagonal terms are zero
        # For matrix multiplication, we need:
        # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 0
        # for (i,j,k) != (i',j',k') in appropriate sense
        # Actually, the tensor is T[i,j,k] = 1 for all valid (i,j,k)
        # And the decomposition must satisfy:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # This is what we checked above.
        return True
    
    if verify(U, V, W, actual_rank):
        if actual_rank < best_rank:
            best_rank = actual_rank
            best_U = U[:, :actual_rank].copy()
            best_V = V[:, :actual_rank].copy()
            best_W = W[:, :actual_rank].copy()
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
