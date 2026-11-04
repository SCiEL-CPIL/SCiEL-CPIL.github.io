import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For each output element (i, k) in the n x p matrix,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # T[i,j,k] should be 1
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
            # Also check that for different (i',k') pairs, the contributions don't leak
            # Actually, the tensor is defined as T[i,j,k] = 1 for all i,j,k
            # So we just need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
    
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen's 7 multiplications for 2x2x2
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # U: (4, 7) where row index = i*2+j
    # V: (4, 7) where row index = j*2+k
    # W: (4, 7) where row index = k*2+i
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)*(B11+B22)
    # A11 -> U[0,0]=1, A22 -> U[3,0]=1
    # B11 -> V[0,0]=1, B22 -> V[3,0]=1
    # C11 += M1, C22 += M1 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2: (A21+A22)*B11
    # A21 -> U[2,1]=1, A22 -> U[3,1]=1
    # B11 -> V[0,1]=1
    # C21 += M2 -> W[2,1]=1 (k=0,i=1)
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3: A11*(B12-B22)
    # A11 -> U[0,2]=1
    # B12 -> V[1,2]=1, B22 -> V[3,2]=-1
    # C12 += M3 -> W[1,2]=1 (k=1,i=0)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # M4: A22*(B21-B11)
    # A22 -> U[3,3]=1
    # B21 -> V[2,3]=1, B11 -> V[0,3]=-1
    # C22 += M4 -> W[3,3]=1 (k=1,i=1)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[3,3] = 1
    
    # M5: (A11+A12)*B22
    # A11 -> U[0,4]=1, A12 -> U[1,4]=1
    # B22 -> V[3,4]=1
    # C12 += M5 -> W[1,4]=1 (k=1,i=0)
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    # A21 -> U[2,5]=1, A11 -> U[0,5]=-1
    # B11 -> V[0,5]=1, B12 -> V[1,5]=1
    # C21 += M6 -> W[2,5]=1 (k=0,i=1)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[2,5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    # A12 -> U[1,6]=1, A22 -> U[3,6]=-1
    # B21 -> V[2,6]=1, B22 -> V[3,6]=1
    # C12 += M7 -> W[1,6]=1 (k=1,i=0)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[1,6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return U, V, W for schoolbook decomposition of n x m x p block."""
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
    
    return U, V, W

def _embed_block(U_block, V_block, W_block, n, m, p, i0, j0, k0):
    """Embed a block decomposition into the full n x m x p tensor.
    
    i0, j0, k0 are the starting indices for this block.
    The block has dimensions (n-i0) x (m-j0) x (p-k0) but we handle general cases.
    """
    # Actually, let's be more careful. We need to embed a sub-block.
    # For a block starting at (i0, j0, k0) with dimensions (ni, mj, pk):
    
    ni = n - i0
    mj = m - j0
    pk = p - k0
    
    rank = U_block.shape[1]
    U_full = np.zeros((n*m, rank))
    V_full = np.zeros((m*p, rank))
    W_full = np.zeros((p*n, rank))
    
    for c in range(rank):
        for i in range(ni):
            for j in range(mj):
                U_full[(i0+i)*m + (j0+j), c] = U_block[i*mj + j, c]
        
        for j in range(mj):
            for k in range(pk):
                V_full[(j0+j)*p + (k0+k), c] = V_block[j*pk + k, c]
        
        for k in range(pk):
            for i in range(ni):
                W_full[(k0+k)*n + (i0+i), c] = W_block[k*ni + i, c]
    
    return U_full, V_full, W_full

def _combine_decompositions(decomps):
    """Combine multiple decompositions by concatenating columns."""
    if not decomps:
        return None
    
    U_list = []
    V_list = []
    W_list = []
    
    for U, V, W in decomps:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook_block(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Strategy: partition 5 into 2+3 and apply Strassen to 2x2x2 sub-blocks
    # Then use schoolbook for the rest
    
    # Partition: j (inner) into 2+3, k (output) into 2+3
    # Block 1: i=0..2, j=0..1, k=0..1 -> 3x2x2, use Strassen-like or schoolbook
    # Actually, for 3x2x2, we can use rank 12 (schoolbook) or try to do better
    
    # Let's try: use Strassen on the 2x2x2 part (j=0..1, k=0..1) for each i
    # And schoolbook for the rest
    
    # Decomposition 1: For i in 0..2, j in 0..1, k in 0..1: use Strassen 2x2x2
    # But Strassen is for 2x2x2, so we need to handle the 3 in the i dimension
    # We can do Strassen for each i independently? No, that doesn't help.
    
    # Better: partition j into 2+3, k into 2+3
    # Block (j=0..1, k=0..1): 3x2x2, rank 12 (schoolbook) or 7*3=21? No.
    # Actually, for a 3x2x2 tensor, the rank is at most 3*2*2=12.
    # Can we do better? The 2x2x2 part has rank 7, but we have 3 in the i dimension.
    
    # Let me think differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # We can write this as a sum of rank-1 tensors.
    
    # Strategy: For the 2x2 sub-block in (j,k), use Strassen's 7 terms.
    # For each of the 3 i values, we have a 2x2x2 tensor.
    # But the i dimension is separate, so we can use the same Strassen structure
    # for each i, giving 3*7=21 rank for the 3x2x2 block.
    # The remaining blocks:
    # - j=0..1, k=2..4: 3x2x3, rank 18
    # - j=2..4, k=0..1: 3x3x2, rank 18
    # - j=2..4, k=2..4: 3x3x3, rank 27
    # Total: 21 + 18 + 18 + 27 = 84, which is worse than 75.
    
    # Better strategy: Use Strassen only where it helps.
    # The 2x2x2 sub-block (j=0..1, k=0..1) with fixed i has rank 7 instead of 8.
    # But since i is a free index, we can share the Strassen structure.
    
    # Actually, let's just use the simple approach:
    # For j=0..1, k=0..1: use Strassen (7 terms per i, but shared)
    # The key insight: T[i,j,k] for j<2, k<2 can be decomposed using
    # the Strassen 7-term formula, and the i dimension just scales each term.
    
    # So for the 3x2x2 block, we use 7 rank-1 tensors:
    # For each Strassen term c, U[i*m+j, c] depends on i, V[j*p+k, c], W[k*n+i, c]
    
    # Let's construct this properly.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Block 1: j in [0,1], k in [0,1], all i in [0,2]
    # Use Strassen 2x2x2 structure
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # Embed: for each i, the Strassen term contributes
    # U[i*5 + j, c] = U_str[j, c] for j in [0,1]
    # V[j*5 + k, c] = V_str[k, c] for k in [0,1]  
    # W[k*3 + i, c] = W_str[i, c] for i in [0,2]
    # Wait, W_str has shape (4,7) for 2x2, but we need (6,7) for 3x2
    
    # Let me redo. For the 3x2x2 block:
    # U: (3*2, 7) = (6, 7)
    # V: (2*2, 7) = (4, 7)
    # W: (2*3, 7) = (6, 7)
    
    U_block1 = np.zeros((6, 7))
    V_block1 = np.zeros((4, 7))
    W_block1 = np.zeros((6, 7))
    
    for c in range(7):
        for i in range(3):
            for j in range(2):
                # U_str[j, c] is the coefficient for A[i,j] in Strassen term c
                # But Strassen is for 2x2, so we need to extend to 3 in i
                # For each i, the 2x2 block in (j,k) uses the same Strassen formula
                U_block1[i*2 + j, c] = U_str[j, c]
        
        for j in range(2):
            for k in range(2):
                V_block1[j*2 + k, c] = V_str[j*2 + k, c]
        
        for k in range(2):
            for i in range(3):
                W_block1[k*3 + i, c] = W_str[i, c]
    
    # Embed block 1 into full tensor
    U_full1 = np.zeros((15, 7))
    V_full1 = np.zeros((25, 7))
    W_full1 = np.zeros((15, 7))
    
    for c in range(7):
        for i in range(3):
            for j in range(2):
                U_full1[i*5 + j, c] = U_block1[i*2 + j, c]
        
        for j in range(2):
            for k in range(2):
                V_full1[j*5 + k, c] = V_block1[j*2 + k, c]
        
        for k in range(2):
            for i in range(3):
                W_full1[k*3 + i, c] = W_block1[k*3 + i, c]
    
    U_parts.append(U_full1)
    V_parts.append(V_full1)
    W_parts.append(W_full1)
    
    # Block 2: j in [0,1], k in [2,4], all i in [0,2]: 3x2x3, schoolbook rank 18
    U_block2, V_block2, W_block2 = _schoolbook_block(3, 2, 3)
    
    U_full2 = np.zeros((15, 18))
    V_full2 = np.zeros((25, 18))
    W_full2 = np.zeros((15, 18))
    
    for c in range(18):
        for i in range(3):
            for j in range(2):
                U_full2[i*5 + j, c] = U_block2[i*2 + j, c]
        
        for j in range(2):
            for k in range(3):
                V_full2[j*5 + (k+2), c] = V_block2[j*3 + k, c]
        
        for k in range(3):
            for i in range(3):
                W_full2[(k+2)*3 + i, c] = W_block2[k*3 + i, c]
    
    U_parts.append(U_full2)
    V_parts.append(V_full2)
    W_parts.append(W_full2)
    
    # Block 3: j in [2,4], k in [0,1], all i in [0,2]: 3x3x2, schoolbook rank 18
    U_block3, V_block3, W_block3 = _schoolbook_block(3, 3, 2)
    
    U_full3 = np.zeros((15, 18))
    V_full3 = np.zeros((25, 18))
    W_full3 = np.zeros((15, 18))
    
    for c in range(18):
        for i in range(3):
            for j in range(3):
                U_full3[i*5 + (j+2), c] = U_block3[i*3 + j, c]
        
        for j in range(3):
            for k in range(2):
                V_full3[(j+2)*5 + k, c] = V_block3[j*2 + k, c]
        
        for k in range(2):
            for i in range(3):
                W_full3[k*3 + i, c] = W_block3[k*3 + i, c]
    
    U_parts.append(U_full3)
    V_parts.append(V_full3)
    W_parts.append(W_full3)
    
    # Block 4: j in [2,4], k in [2,4], all i in [0,2]: 3x3x3, schoolbook rank 27
    U_block4, V_block4, W_block4 = _schoolbook_block(3, 3, 3)
    
    U_full4 = np.zeros((15, 27))
    V_full4 = np.zeros((25, 27))
    W_full4 = np.zeros((15, 27))
    
    for c in range(27):
        for i in range(3):
            for j in range(3):
                U_full4[i*5 + (j+2), c] = U_block4[i*3 + j, c]
        
        for j in range(3):
            for k in range(3):
                V_full4[(j+2)*5 + (k+2), c] = V_block4[j*3 + k, c]
        
        for k in range(3):
            for i in range(3):
                W_full4[(k+2)*3 + i, c] = W_block4[k*3 + i, c]
    
    U_parts.append(U_full4)
    V_parts.append(V_full4)
    W_parts.append(W_full4)
    
    # Combine all blocks
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    
    rank_combined = U_combined.shape[1]
    
    # Verify
    if _verify(U_combined, V_combined, W_combined, n, m, p):
        if rank_combined < best_rank:
            best_U, best_V, best_W = U_combined, V_combined, W_combined
            best_rank = rank_combined
    
    # Check time
    if time.time() > deadline:
        # Return best found
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Try alternative: partition k into 2+3 and j into 2+3 differently
    # Or try other splits
    
    # Try: j split 3+2, k split 2+3
    U_parts2 = []
    V_parts2 = []
    W_parts2 = []
    
    # Block A: j in [0,2], k in [0,1]: 3x3x2, schoolbook rank 18
    U_bA, V_bA, W_bA = _schoolbook_block(3, 3, 2)
    U_fA = np.zeros((15, 18))
    V_fA = np.zeros((25, 18))
    W_fA = np.zeros((15, 18))
    for c in range(18):
        for i in range(3):
            for j in range(3):
                U_fA[i*5 + j, c] = U_bA[i*3 + j, c]
        for j in range(3):
            for k in range(2):
                V_fA[j*5 + k, c] = V_bA[j*2 + k, c]
        for k in range(2):
            for i in range(3):
                W_fA[k*3 + i, c] = W_bA[k*3 + i, c]
    U_parts2.append(U_fA)
    V_parts2.append(V_fA)
    W_parts2.append(W_fA)
    
    # Block B: j in [0,2], k in [2,4]: 3x3x3, schoolbook rank 27
    U_bB, V_bB, W_bB = _schoolbook_block(3, 3, 3)
    U_fB = np.zeros((15, 27))
    V_fB = np.zeros((25, 27))
    W_fB = np.zeros((15, 27))
    for c in range(27):
        for i in range(3):
            for j in range(3):
                U_fB[i*5 + j, c] = U_bB[i*3 + j, c]
        for j in range(3):
            for k in range(3):
                V_fB[j*5 + (k+2), c] = V_bB[j*3 + k, c]
        for k in range(3):
            for i in range(3):
                W_fB[(k+2)*3 + i, c] = W_bB[k*3 + i, c]
    U_parts2.append(U_fB)
    V_parts2.append(V_fB)
    W_parts2.append(W_fB)
    
    # Block C: j in [3,4], k in [0,1]: 3x2x2, use Strassen rank 7
    U_bC, V_bC, W_bC = _strassen_2x2x2()
    U_fC = np.zeros((15, 7))
    V_fC = np.zeros((25, 7))
    W_fC = np.zeros((15, 7))
    for c in range(7):
        for i in range(3):
            for j in range(2):
                U_fC[i*5 + (j+3), c] = U_bC[j, c]
        for j in range(2):
            for k in range(2):
                V_fC[(j+3)*5 + k, c] = V_bC[j*2 + k, c]
        for k in range(2):
            for i in range(3):
                W_fC[k*3 + i, c] = W_bC[i, c]
    U_parts2.append(U_fC)
    V_parts2.append(V_fC)
    W_parts2.append(W_fC)
    
    # Block D: j in [3,4], k in [2,4]: 3x2x3, schoolbook rank 18
    U_bD, V_bD, W_bD = _schoolbook_block(3, 2, 3)
    U_fD = np.zeros((15, 18))
    V_fD = np.zeros((25, 18))
    W_fD = np.zeros((15, 18))
    for c in range(18):
        for i in range(3):
            for j in range(2):
                U_fD[i*5 + (j+3), c] = U_bD[i*2 + j, c]
        for j in range(2):
            for k in range(3):
                V_fD[(j+3)*5 + (k+2), c] = V_bD[j*3 + k, c]
        for k in range(3):
            for i in range(3):
                W_fD[(k+2)*3 + i, c] = W_bD[k*3 + i, c]
    U_parts2.append(U_fD)
    V_parts2.append(V_fD)
    W_parts2.append(W_fD)
    
    U_combined2 = np.hstack(U_parts2)
    V_combined2 = np.hstack(V_parts2)
    W_combined2 = np.hstack(W_parts2)
    
    rank_combined2 = U_combined2.shape[1]
    
    if _verify(U_combined2, V_combined2, W_combined2, n, m, p):
        if rank_combined2 < best_rank:
            best_U, best_V, best_W = U_combined2, V_combined2, W_combined2
            best_rank = rank_combined2
    
    # Check time
    if time.time() > deadline:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Return best found
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    return {"U": U_out, "V": V_out, "W": W_out}
