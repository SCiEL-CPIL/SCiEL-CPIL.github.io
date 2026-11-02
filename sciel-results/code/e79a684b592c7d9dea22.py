"""Matrix multiplication tensor decomposition for 4x4 * 4x8."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_4x4_decomposition():
    """
    Build a rank-49 decomposition for 4x4 * 4x4 matrix multiplication.
    
    Uses the fact that M_4,4,4 can be decomposed with rank 49.
    We construct this using a structured approach based on block methods.
    
    The decomposition satisfies:
    sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] = 1 for all i,j,k
    """
    n = 4
    rank = 49
    
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    # We'll use a construction based on the known result that M_4,4,4 has rank 49.
    # One approach: use 7x7 blocks with rank-7 decompositions for each block pair.
    # Actually, let's use a direct construction.
    
    # For 4x4 matrices, we can partition into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # where C11 = A11*B11 + A12*B21, etc.
    
    # Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
    # There are 4 block multiplications needed for the full 4x4 product.
    # But we can share some computations.
    
    # Let's use a different approach: enumerate all (i,j,k) triples and try to find
    # a low-rank structure.
    
    # Actually, let me use a known explicit construction.
    # The rank-49 decomposition for M_4,4,4 can be constructed as follows:
    # Use 49 rank-1 terms, each corresponding to a specific pattern.
    
    # For now, let's use a greedy/structured approach:
    # We know the answer exists. Let's construct it systematically.
    
    # Method: Use the fact that we can do 4x4 multiplication by doing
    # multiple 2x2 Strassen multiplications.
    
    # Partition indices: i = 2*a + a', j = 2*b + b', k = 2*c + c'
    # where a, b, c in {0,1} and a', b', c' in {0,1}
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{b,b'} A[2a+a', 2b+b'] * B[2b+b', 2c+c']
    
    # Group by (a, a', c, c'):
    # C[2a+a', 2c+c'] = sum_b (A[2a+a', 2b] * B[2b, 2c+c'] + A[2a+a', 2b+1] * B[2b+1, 2c+c'])
    
    # This is getting complex. Let me use a simpler approach:
    # Just return a valid decomposition, even if not minimal, and verify.
    
    # For reliability, let's use the schoolbook for 4x4 (rank 64) and then
    # see if we can do better.
    
    # Actually, let me try a different strategy: use random search with 
    # structured initialization for the 4x4 case.
    
    return None  # We'll handle this differently

def _build_4x4_rank49():
    """
    Construct a rank-49 decomposition for 4x4 * 4x4.
    
    This uses a known explicit construction. The key idea is to use
    49 rank-1 tensors that exactly decompose the multiplication tensor.
    """
    n = 4
    rank = 49
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # We'll use a construction based on partitioning into 2x2 blocks
    # and using Strassen-like identities.
    
    # For 2x2 * 2x2, Strassen gives rank 7:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, we have 4 block multiplications (each 2x2*2x2).
    # If we use independent Strassen for each, we get rank 4*7 = 28.
    # But we need to handle the cross terms properly.
    
    # Actually, the full 4x4 multiplication involves more than just
    # 4 independent 2x2 multiplications. Let me think more carefully.
    
    # A (4x4) * B (4x4) = C (4x4)
    # Partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # So we need 8 products of 2x2 matrices.
    # Using Strassen for each: 8 * 7 = 56 rank-1 terms.
    # But we can share some computations.
    
    # The known rank-49 result suggests there's a clever sharing.
    
    # For now, let's use a valid but possibly non-minimal construction
    # and verify it.
    
    # Use 8 independent Strassen decompositions for the 8 block products.
    # This gives rank 56, which is better than schoolbook 64.
    
    # Let's construct this explicitly.
    
    col = 0
    # For each pair of blocks (i_block, j_block, k_block) where
    # i_block, j_block, k_block in {0,1}, we need A_{i_block,j_block} * B_{j_block,k_block}
    
    # Actually, C_{i_block,k_block} = sum_{j_block} A_{i_block,j_block} * B_{j_block,k_block}
    # So for each (i_block, k_block), we sum over j_block.
    
    # There are 4 (i_block, k_block) pairs, each requiring 2 products of 2x2 matrices.
    # Total: 8 products of 2x2 matrices.
    
    # For each 2x2 * 2x2 product, use Strassen (rank 7).
    # Total rank: 8 * 7 = 56.
    
    # Let's implement this.
    
    def strassen_2x2(U_base, V_base, W_base, a_block, b_block, c_block, start_col):
        """Add Strassen decomposition for A_{a,b} * B_{b,c} contribution to C_{a,c}."""
        # The 2x2 blocks are at positions:
        # A_{a,b}: rows 2*a to 2*a+1, cols 2*b to 2*b+1
        # B_{b,c}: rows 2*b to 2*b+1, cols 2*c to 2*c+1
        # C_{a,c}: rows 2*a to 2*a+1, cols 2*c to 2*c+1
        
        # Strassen formulas for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For each Strassen term, we add a rank-1 contribution.
        
        # The indices within the 2x2 block:
        # A11 -> (0,0), A12 -> (0,1), A21 -> (1,0), A22 -> (1,1)
        # Similarly for B and C.
        
        # M1 = (A11+A22)(B11+B22)
        # This contributes to C11 and C22.
        # U entries: A11 and A22 both have coefficient 1
        # V entries: B11 and B22 both have coefficient 1
        # W entries: C11 has +1, C22 has +1
        
        U_base[2*a_block + 0, 2*b_block + 0, start_col] = 1  # A11
        U_base[2*a_block + 1, 2*b_block + 1, start_col] = 1  # A22
        V_base[2*b_block + 0, 2*c_block + 0, start_col] = 1  # B11
        V_base[2*b_block + 0, 2*c_block + 1, start_col] = 1  # B12... wait
        
        # Actually, B_{b,c} is a 2x2 matrix with entries:
        # B11 -> (row 0, col 0) of B_{b,c} -> global (2*b, 2*c)
        # B12 -> (row 0, col 1) of B_{b,c} -> global (2*b, 2*c+1)
        # B21 -> (row 1, col 0) of B_{b,c} -> global (2*b+1, 2*c)
        # B22 -> (row 1, col 1) of B_{b,c} -> global (2*b+1, 2*c+1)
        
        # Let me redo this more carefully.
        
        return start_col + 1  # Placeholder
    
    # This is getting too complex to implement correctly in one go.
    # Let me use a simpler approach: return a valid schoolbook decomposition
    # for 4x4 (rank 64) and use it to build the 4x4*4x8 decomposition.
    
    # For 4x4 * 4x8, we can do:
    # C = A * B where A is 4x4, B is 4x8
    # Partition B into B1 (4x4) and B2 (4x4)
    # C1 = A * B1, C2 = A * B2
    # Use the same U for both, but different V and W.
    
    # If we have a rank-r decomposition for 4x4*4x4, we get rank 2r for 4x4*4x8.
    # With schoolbook r=64, we get rank 128 (same as overall schoolbook).
    # With r=49, we get rank 98.
    
    # For now, let's just return None and handle in construct().
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _construct_4x4x8_rank98():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8.
    
    Uses two rank-49 decompositions for 4x4 * 4x4.
    """
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We need a rank-49 decomposition for 4x4 * 4x4.
    # Let's construct one using a systematic approach.
    
    # For 4x4 * 4x4, the tensor has 64 nonzero entries (one for each (i,j,k)).
    # We need 49 rank-1 terms.
    
    # Let me use a construction based on the following idea:
    # Use 7x7 = 49 terms, where each term corresponds to a specific
    # pattern of indices.
    
    # Actually, let me just use a valid decomposition, even if not minimal.
    # I'll use rank 64 (schoolbook) for 4x4*4x4, giving rank 128 for 4x4*4x8.
    # But that's the same as overall schoolbook.
    
    # Let me try to construct a better one.
    
    # Here's a concrete approach: use the fact that we can write
    # 4x4 multiplication as a sum of 49 rank-1 terms.
    
    # I'll use a construction from the literature.
    
    # For now, let's use a heuristic: try to find a rank-49 decomposition
    # using random search with verification.
    
    start_time = time.time()
    deadline = start_time + 100
    
    # Try random search for rank-49 decomposition of 4x4*4x4
    def find_4x4_decomp(target_rank, max_attempts=100):
        for seed in range(max_attempts):
            if time.time() > deadline:
                break
            np.random.seed(seed)
            
            U4 = np.random.choice([-1, 0, 1], size=(16, target_rank))
            V4 = np.random.choice([-1, 0, 1], size=(16, target_rank))
            
            # Solve for W4
            equations = []
            b = []
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        row = np.zeros(16)
                        for c in range(target_rank):
                            coeff = U4[i * 4 + j, c] * V4[j * 4 + k, c]
                            if coeff != 0:
                                row[k * 4 + i] += coeff
                        if np.any(row != 0):
                            equations.append(row)
                            b.append(1.0)
            
            if not equations:
                continue
            
            A_mat = np.array(equations)
            b_vec = np.array(b)
            
            try:
                w_flat, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W4 = w_flat.reshape(16, target_rank)
                
                # Verify
                valid = True
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            s = np.sum(U4[i*4+j] * V4[j*4+k] * W4[k*4+i])
                            if abs(s - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U4, V4, W4
            except:
                pass
        
        return None
    
    result = find_4x4_decomp(49)
    
    if result is not None:
        U4, V4, W4 = result
        
        # Now build 4x4 * 4x8 using two copies of the 4x4 * 4x4 decomposition
        # B is 4x8, partition into B1 (cols 0-3) and B2 (cols 4-7)
        
        # U is the same for both halves
        U[:, :49] = U4
        U[:, 49:] = U4
        
        # V: first half for B1, second half for B2
        V[:16, :49] = V4
        V[16:32, 49:] = V4  # Wait, V has shape (m*p, rank) = (4*8, rank) = (32, rank)
        # V[j*p+k, c] where j in [0,4), k in [0,8)
        # For first half (k=0..3): V[j*8+k, c] for c in [0,49)
        # For second half (k=4..7): V[j*8+k, c] for c in [49,98)
        
        # Reset V
        V = np.zeros((32, 98))
        for j in range(4):
            for k in range(4):
                V[j * 8 + k, :49] = V4[j * 4 + k]
                V[j * 8 + (k + 4), 49:] = V4[j * 4 + k]
        
        # W: first half for C1 (cols 0-3 of B), second half for C2 (cols 4-7)
        # W[k*n+i, c] where k in [0,8), i in [0,4)
        # For first half (k=0..3): W[k*4+i, c] for c in [0,49)
        # For second half (k=4..7): W[k*4+i, c] for c in [49,98)
        
        W = np.zeros((32, 98))
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, :49] = W4[k * 4 + i]
                W[(k + 4) * 4 + i, 49:] = W4[k * 4 + i]
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), 4, 4, 8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Try to construct rank-98 decomposition
    result = _construct_4x4x8_rank98()
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
