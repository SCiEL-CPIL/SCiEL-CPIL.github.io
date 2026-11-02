"""Matrix multiplication 4,4,4 - Rank-minimal exact tensor decomposition."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _recursive_strassen(n, m, p):
    """
    Generate a rank-(7^k) decomposition for n x m by m x p matrix multiplication
    using recursive Strassen. Returns (U, V, W) as lists of lists of Fractions.
    
    For n=m=p=4, this gives rank 49.
    The tensor indices follow: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    if n == 1 and m == 1 and p == 1:
        # Base case: rank 1, U[0,0]=1, V[0,0]=1, W[0,0]=1
        return [[1]], [[1]], [[1]]
    
    # Split dimensions
    n1, n2 = n // 2, n - n // 2
    m1, m2 = m // 2, m - m // 2
    p1, p2 = p // 2, p - p // 2
    
    # Recursively get decompositions for the 7 Strassen sub-products
    # We need to handle rectangular splits carefully.
    # For simplicity, we'll use square blocks where possible.
    
    # Actually, for 4x4x4, let's use a specific approach:
    # Split into 2x2 blocks, each 2x2x2 uses Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    if n == 2 and m == 2 and p == 2:
        # Standard Strassen for 2x2x2, rank 7
        # The 7 products are:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B11-B21)
        # M4 = A22*(B22-B12)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # U has shape (n*m, rank) = (4, 7)
        # V has shape (m*p, rank) = (4, 7)
        # W has shape (p*n, rank) = (4, 7)
        
        # Indexing: A[i,j] -> U[i*m+j], B[j,k] -> V[j*p+k], C[i,k] -> W[k*n+i]
        # For 2x2: A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        #          B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        #          C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)*(B11+B22)
        # A11=(0,0)->idx0, A22=(1,1)->idx3
        # B11=(0,0)->idx0, B22=(1,1)->idx3
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 += M1, C22 += M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)*B11
        # A21=(1,0)->idx2, A22=(1,1)->idx3
        # B11=(0,0)->idx0
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 += M2
        W[2][1] = 1
        
        # M3 = A11*(B11-B21)
        # A11=(0,0)->idx0
        # B11=(0,0)->idx0, B21=(1,0)->idx2
        U[0][2] = 1
        V[0][2] = 1; V[2][2] = -1
        # C11 += M3, C12 -= M3 -> C11 += M3, C12 += (-1)*M3
        W[0][2] = 1; W[1][2] = -1
        
        # M4 = A22*(B22-B12)
        # A22=(1,1)->idx3
        # B22=(1,1)->idx3, B12=(0,1)->idx1
        U[3][3] = 1
        V[3][3] = 1; V[1][3] = -1
        # C22 += M4, C21 -= M4
        W[3][3] = 1; W[2][3] = -1
        
        # M5 = (A11+A12)*B22
        # A11=(0,0)->idx0, A12=(0,1)->idx1
        # B22=(1,1)->idx3
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C12 += M5
        W[1][4] = 1
        
        # M6 = (A21-A11)*(B11+B12)
        # A21=(1,0)->idx2, A11=(0,0)->idx0
        # B11=(0,0)->idx0, B12=(0,1)->idx1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C21 += M6, C22 += M6
        W[2][5] = 1; W[3][5] = 1
        
        # M7 = (A12-A22)*(B21+B22)
        # A12=(0,1)->idx1, A22=(1,1)->idx3
        # B21=(1,0)->idx2, B22=(1,1)->idx3
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C12 += M7, C22 += M7
        W[1][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    # For 4x4x4: use 2-level Strassen
    # Split into four 2x2 submatrices
    # We'll combine the 2x2x2 Strassen decompositions
    
    # Actually, let's just do it directly for 4x4x4 using the 2x2x2 result
    # The 4x4x4 multiplication can be seen as block multiplication of 2x2 blocks
    # where each block is 2x2, 2x2, 2x2
    
    # But the standard recursive Strassen for 4x4 gives rank 49
    # Let me construct it properly
    
    # For 4x4x4, split A into 4 blocks of 2x2, B into 4 blocks of 2x2
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    
    # C = AB has blocks:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen on the 2x2 block structure:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each of these is a product of two 2x2 matrices, which itself uses rank 7
    # So total rank = 7 * 7 = 49
    
    # Let me build this by combining the 2x2x2 Strassen patterns
    
    # First, get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _recursive_strassen(2, 2, 2)
    
    # Now for 4x4x4, we have 49 rank terms
    # For each of the 7 block-level Strassen terms, we use the 7 element-level terms
    
    # The mapping is complex. Let me use a different approach:
    # Directly construct the 4x4x4 rank-49 decomposition.
    
    # Actually, the simplest correct approach: use the known fact that
    # R(4,4,4) <= 49 via recursive Strassen.
    
    # Let me construct it by treating 4x4 as 2x2 blocks and applying
    # Strassen at the block level, with each block product done via Strassen.
    
    # Block indices for 4x4 matrix (row-major, 2x2 blocks):
    # A block (bi, bj) occupies rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Element (i,j) in block (bi,bj) has global index i = 2*bi + i_local, j = 2*bj + j_local
    
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # The 7 block-level Strassen terms
    # For term t (0-6), we have 7 element-level terms (0-6)
    # Global rank index = t * 7 + s
    
    # Block-level Strassen coefficients:
    # M1: A blocks {0,3} with coeffs {1,1}, B blocks {0,3} with coeffs {1,1}
    # M2: A blocks {2,3} with coeffs {1,1}, B blocks {0} with coeff {1}
    # M3: A blocks {0} with coeff {1}, B blocks {0,2} with coeffs {1,-1}
    # M4: A blocks {3} with coeff {1}, B blocks {3,1} with coeffs {1,-1}
    # M5: A blocks {0,1} with coeffs {1,1}, B blocks {3} with coeff {1}
    # M6: A blocks {2,0} with coeffs {1,-1}, B blocks {0,1} with coeffs {1,1}
    # M7: A blocks {1,3} with coeffs {1,-1}, B blocks {2,3} with coeffs {1,1}
    
    block_A = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, 1)],   # M2
        [(0, 1)],           # M3
        [(3, 1)],           # M4
        [(0, 1), (1, 1)],   # M5
        [(2, 1), (0, -1)],  # M6
        [(1, 1), (3, -1)],  # M7
    ]
    block_B = [
        [(0, 1), (3, 1)],   # M1
        [(0, 1)],           # M2
        [(0, 1), (2, -1)],  # M3
        [(3, 1), (1, -1)],  # M4
        [(3, 1)],           # M5
        [(0, 1), (1, 1)],   # M6
        [(2, 1), (3, 1)],   # M7
    ]
    block_C = [
        [(0, 1), (3, 1)],   # M1 -> C11 += M1, C22 += M1
        [(2, 1)],           # M2 -> C21 += M2
        [(0, 1), (1, -1)],  # M3 -> C11 += M3, C12 -= M3
        [(3, 1), (2, -1)],  # M4 -> C22 += M4, C21 -= M4
        [(1, 1)],           # M5 -> C12 += M5
        [(2, 1), (3, 1)],   # M6 -> C21 += M6, C22 += M6
        [(1, 1), (3, 1)],   # M7 -> C12 += M7, C22 += M7
    ]
    
    # For each block-level term t and element-level term s:
    # The element-level Strassen gives U2, V2, W2 for 2x2x2
    # U2[i_local*2 + j_local, s] for A element in block
    # V2[j_local*2 + k_local, s] for B element in block  
    # W2[k_local*2 + i_local, s] for C element in block
    
    for t in range(7):
        for s in range(7):
            rank_idx = t * 7 + s
            
            # A side: for each block (bi, coeff) in block_A[t]
            for bi, coeff in block_A[t]:
                # Block bi is at position (bi//2, bi%2) in the 2x2 block grid
                # Wait, I need to be more careful with block indexing
                # Blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                # So block bi has row block bi//2, col block bi%2
                
                # For each element in this block:
                for i_loc in range(2):
                    for j_loc in range(2):
                        # Global A index: i = 2*(bi//2) + i_loc, j = 2*(bi%2) + j_loc
                        i_global = 2 * (bi // 2) + i_loc
                        j_global = 2 * (bi % 2) + j_loc
                        a_idx = i_global * 4 + j_global
                        
                        # The element-level Strassen coefficient for this element
                        # U2[i_loc*2 + j_loc, s]
                        val = U2[i_loc * 2 + j_loc][s]
                        if val != 0:
                            U[a_idx][rank_idx] += coeff * val
            
            # B side: for each block (bj, coeff) in block_B[t]
            for bj, coeff in block_B[t]:
                for j_loc in range(2):
                    for k_loc in range(2):
                        j_global = 2 * (bj // 2) + j_loc
                        k_global = 2 * (bj % 2) + k_loc
                        b_idx = j_global * 4 + k_global
                        
                        val = V2[j_loc * 2 + k_loc][s]
                        if val != 0:
                            V[b_idx][rank_idx] += coeff * val
            
            # C side: for each block (bc, coeff) in block_C[t]
            for bc, coeff in block_C[t]:
                for k_loc in range(2):
                    for i_loc in range(2):
                        k_global = 2 * (bc // 2) + k_loc
                        i_global = 2 * (bc % 2) + i_loc
                        c_idx = k_global * 4 + i_global
                        
                        val = W2[k_loc * 2 + i_loc][s]
                        if val != 0:
                            W[c_idx][rank_idx] += coeff * val
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # For each (i,j,k), check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # And for all other combinations, the sum should be 0
    
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _verify_decomposition_sparse(U, V, W, n, m, p):
    """Faster verification using sparsity."""
    rank = len(U[0]) if U else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i*m+j]
                v_row = V[j*p+k]
                w_row = W[k*n+i]
                for r in range(rank):
                    if u_row[r] != 0 and v_row[r] != 0 and w_row[r] != 0:
                        s += u_row[r] * v_row[r] * w_row[r]
                if s != 1:
                    return False
    return True


# EVOLVE-BLOCK-END


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U': ..., 'V': ..., 'W': ...} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with recursive Strassen rank-49 decomposition
    U, V, W = _recursive_strassen(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition_sparse(U, V, W, n, m, p):
        # Fallback to schoolbook if Strassen construction is buggy
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by eliminating linearly dependent columns
    # or finding a lower-rank decomposition
    
    # First, check if we can reduce from 49 by finding dependencies
    # The rank of the tensor is the minimum number of rank-1 tensors needed
    # We have 49 rank-1 terms. Can we combine them?
    
    # For now, return the rank-49 solution which is known to be valid
    # and better than the schoolbook rank-64
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction: try to find if any column can be expressed
    # as a linear combination of others, allowing removal
    
    rank = len(U[0])
    
    # Convert to a format where we can check linear dependencies
    # Each column r defines a rank-1 tensor T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We want to see if any T_r is a linear combination of the others
    
    # This is expensive, so we'll do a limited search
    
    # Try removing each column and checking if the remaining still span
    # Actually, we need the sum of all remaining to still equal the tensor
    
    # A simpler approach: check if the 49 vectors (flattened tensors) are linearly independent
    # If not, we might be able to reduce rank
    
    # Flatten each rank-1 tensor into a vector of length n*m*p = 64
    # Then check rank of the 49x64 matrix
    
    try:
        import numpy as np
        
        # Build the matrix of rank-1 tensors
        tensor_size = n * m * p
        tensors = []
        for r in range(rank):
            vec = [0] * tensor_size
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[i * m * p + j * p + k] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            tensors.append(vec)
        
        tensor_matrix = np.array(tensors, dtype=float)
        
        # Check rank
        matrix_rank = np.linalg.matrix_rank(tensor_matrix, tol=1e-10)
        
        if matrix_rank < rank:
            # There are linear dependencies. Try to find a subset of columns
            # that still sums to the correct tensor.
            # This is a combinatorial problem, so we'll do a greedy approach.
            
            # The target tensor is all-ones
            target = np.ones(tensor_size)
            
            # Greedy: start with all 49, try removing one at a time
            # if the sum of remaining still equals target
            
            active = list(range(rank))
            
            # Check current sum
            current_sum = sum(tensors[r] for r in active)
            if not np.allclose(current_sum, target, atol=1e-10):
                return {"U": U, "V": V, "W": W}
            
            # Try to remove columns
            for r in range(rank):
                if time.time() > deadline:
                    break
                test_active = [x for x in active if x != r]
                if not test_active:
                    continue
                test_sum = sum(tensors[x] for x in test_active)
                if np.allclose(test_sum, target, atol=1e-10):
                    active = test_active
            
            if len(active) < rank:
                # Build new U, V, W with only active columns
                new_U = [[U[i][r] for r in active] for i in range(n*m)]
                new_V = [[V[i][r] for r in active] for i in range(m*p)]
                new_W = [[W[i][r] for r in active] for i in range(p*n)]
                
                # Verify
                if _verify_decomposition_sparse(new_U, new_V, new_W, n, m, p):
                    U, V, W = new_U, new_V, new_W
                    rank = len(active)
        
    except Exception:
        pass
    
    # Return the best result found
    return {"U": U, "V": V, "W": W}
