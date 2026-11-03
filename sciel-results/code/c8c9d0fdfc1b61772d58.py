"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using tensor contraction via einsum."""
    rank = len(U[0]) if len(U) > 0 else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape: U is (n*m, rank) -> (n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want T[i,j,k] == 1 for all i,j,k
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_block(n, m, p, seed=42):
    """
    Try to exploit block structure.
    For 3x4x4:
    Split rows of A (n=3) into groups of size 1: 3 groups.
    Split cols of A / rows of B (m=4) into groups of size 2: 2 groups.
    Split cols of B (p=4) into groups of size 2: 2 groups.
    
    This creates 3 * 2 * 2 = 12 blocks of size (1x2) * (2x2) = 1x2.
    Each block multiplication (1x2) * (2x2) can be done with 2 multiplications (schoolbook for 1x2x2).
    Total rank = 12 * 2 = 24.
    
    Wait, let's verify this logic.
    A is 3x4. B is 4x4.
    Let A be partitioned into row blocks A_i (1x4) for i=0,1,2.
    Let B be partitioned into row blocks B_j (4x2) for j=0,1? No, B is 4x4.
    Let's partition the inner dimension m=4.
    A = [A_0, A_1] where A_0, A_1 are 3x2.
    B = [B_0; B_1] where B_0, B_1 are 2x4.
    C = A_0 B_0 + A_1 B_1.
    This doesn't reduce rank directly unless we use Strassen on the blocks.
    But A_0 is 3x2, B_0 is 2x4. Product is 3x4.
    Rank of 3x2x4 multiplication?
    Schoolbook is 3*2*4 = 24.
    If we do this for two blocks, rank is 24+24=48. No gain.
    
    Let's try a different partition.
    A is 3x4. B is 4x4.
    Use Strassen on 2x2 blocks?
    We can pad A to 4x4? No, that increases rank.
    
    Let's look at the structure: <3,4,4>.
    Lower bound is max(3*4, 4*4, 4*3) = 16.
    Known rank for <2,2,2> is 7.
    
    Can we embed <2,2,2> into <3,4,4>?
    If we take a 2x2 sub-block of A, a 2x2 sub-block of B, and compute a 2x2 output.
    That uses 7 multiplications.
    But we have to compute the full 3x4 output.
    
    Let's try a randomized local search from the schoolbook solution.
    We represent the decomposition as matrices U, V, W of size (n*m x r), (m*p x r), (p*n x r).
    We want to minimize r such that the tensor equation holds.
    Since we need exact integer/half-integer solutions, random search in continuous space is hard.
    However, we can try to find a sparse solution or use the known structure.
    
    Actually, there is a known result for <3,3,3> rank 23 (or 21?).
    For <3,4,4>, let's try to use the fact that 4 = 2+2.
    
    Let's try a specific construction:
    Split the 4x4 multiplication into four 2x2 multiplications?
    A is 3x4. We can't split it into 2x2 blocks neatly for the first dimension (3).
    But we can split the inner dimension (4) into 2+2.
    C = A[:, 0:2] B[0:2, :] + A[:, 2:4] B[2:4, :]
    Let A1 = A[:, 0:2] (3x2), B1 = B[0:2, :] (2x4).
    Let A2 = A[:, 2:4] (3x2), B2 = B[2:4, :] (2x4).
    C = A1 B1 + A2 B2.
    
    Multiplying a 3x2 by a 2x4 matrix.
    This is equivalent to computing 4 separate 3x2 * 2x1 products? No.
    It's a bilinear map (3x2) x (2x4) -> (3x4).
    The rank of this specific tensor <3,2,4> is what we need.
    Schoolbook rank is 3*2*4 = 24.
    If we can do <3,2,4> with rank < 24, we save.
    But <3,2,4> is essentially two <3,2,1> products? No, the output is 3x4.
    It's like computing 4 columns of A1*B1.
    Each column is a 3x2 * 2x1 product. Rank of <3,2,1> is 3 (since 3*2*1=3, and lower bound max(3,2,1)=3).
    So 4 columns * 3 = 12 multiplications for A1 B1.
    Similarly 12 for A2 B2.
    Total 24. Same as schoolbook.
    
    Is there a way to share multiplications between the columns?
    The columns of B1 are independent in the output, but they share the same A1.
    This is the "batch" aspect.
    Tensor <3,2,4> = <3,2,1> x 4? Not exactly, because the input A1 is shared.
    Actually, the tensor for (A1, B1) -> C1 where C1 is 3x4.
    C1[i,k] = sum_j A1[i,j] B1[j,k].
    This is just the standard matrix multiplication tensor for dimensions 3,2,4.
    The rank is R(3,2,4).
    R(3,2,4) <= R(3,2,1)*4 = 12? No, that's if we treat them independently.
    But we can use the structure.
    R(3,2,4) is at most 3*2*4 = 24.
    Is it less?
    R(2,2,2) = 7.
    R(3,2,2) ?
    Let's try to find R(3,2,4) using known bounds or small searches.
    
    If R(3,2,4) is 24, then total is 48.
    If R(3,2,4) is less, say 20, total is 40.
    
    Let's try to search for a decomposition of <3,2,4> with rank < 24.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Baseline: Schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify baseline
    if not _verify_fast(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Should not happen
        return best_result

    # Strategy: Try to find a decomposition with rank < 48.
    # We will use a random search in the space of matrices U, V, W with fixed rank r.
    # We want to satisfy U[i*m+j, r'] V[j*p+k, r'] W[k*n+i, r'] = 1 (summed over r').
    # This is a system of polynomial equations.
    # For small ranks, we can try to solve this using random initialization and gradient descent,
    # or simply random sampling if the solution space is large enough.
    # Given the difficulty, we try a few specific structured decompositions.
    
    # Idea: Use Strassen for 2x2 blocks where possible.
    # Pad n=3 to 4? No, that increases size.
    # Split m=4 into 2+2.
    # C = A[:, 0:2] B[0:2, :] + A[:, 2:4] B[2:4, :]
    # Let M1 = A[:, 0:2] B[0:2, :] (3x4)
    # Let M2 = A[:, 2:4] B[2:4, :] (3x4)
    # Rank of <3,2,4> multiplication.
    # Let's try to find rank of <3,2,4>.
    # Lower bound: max(3*2, 2*4, 4*3) = 12.
    # Schoolbook: 24.
    # Can we do better?
    # R(3,2,4) <= R(3,2,2) + R(3,2,2)? No.
    # R(3,2,4) <= 2 * R(3,2,2)?
    # If we split the 4 columns into two 2-column blocks.
    # M1[:, 0:2] = A[:, 0:2] B[0:2, 0:2] -> <3,2,2>
    # M1[:, 2:4] = A[:, 0:2] B[0:2, 2:4] -> <3,2,2>
    # So R(3,2,4) <= 2 * R(3,2,2).
    # What is R(3,2,2)?
    # Lower bound max(6, 4, 6) = 6.
    # Schoolbook 12.
    # Is R(3,2,2) < 12?
    # R(2,2,2) = 7.
    # R(3,2,2) can be seen as 3 rows of <2,2,2>? No.
    # It's (3x2) * (2x2) -> (3x2).
    # This is equivalent to 2 independent <3,2,1> products? No, the 2x2 input is shared.
    # Actually, C = A B. A is 3x2, B is 2x2.
    # C[:,0] = A B[:,0]. C[:,1] = A B[:,1].
    # This is a bilinear map (3x2) x (2x2) -> (3x2).
    # Tensor T[i,j,k,l] where i in 3, j in 2, k in 2 (cols of B), l in 2 (cols of C)?
    # No, the output is 3x2.
    # Let's use the standard tensor definition.
    # T_{i,j,k} for C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Here i in 0..2, j in 0..1, k in 0..1.
    # This is just <3,2,2>.
    # Rank is at most 3*2*2 = 12.
    # Can we do 11? 10?
    # If R(3,2,2) = 10, then R(3,2,4) <= 20.
    # Then total rank <= 40.
    
    # Let's try to search for R(3,2,2) < 12.
    # We can use a random search for <3,2,2> with rank 11, 10, etc.
    
    def _search_rank(n_sub, m_sub, p_sub, target_rank, num_trials=100, seed=0):
        """Random search for decomposition of <n_sub, m_sub, p_sub> with target_rank."""
        rng = random.Random(seed)
        U_size = n_sub * m_sub
        V_size = m_sub * p_sub
        W_size = p_sub * n_sub
        
        for trial in range(num_trials):
            # Initialize with small random integers or half-integers
            # To increase chance of exact solution, start with integers
            U = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(U_size)]
            V = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(V_size)]
            W = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(W_size)]
            
            # Verify
            if _verify_fast(U, V, W, n_sub, m_sub, p_sub, tol=1e-9):
                return {"U": U, "V": V, "W": W}
            
            # Also try with 0/1 matrices
            U = [[rng.randint(0, 1) for _ in range(target_rank)] for _ in range(U_size)]
            V = [[rng.randint(0, 1) for _ in range(target_rank)] for _ in range(V_size)]
            W = [[rng.randint(0, 1) for _ in range(target_rank)] for _ in range(W_size)]
            if _verify_fast(U, V, W, n_sub, m_sub, p_sub, tol=1e-9):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Search for R(3,2,2) < 12
    # If found, we can construct R(3,4,4) <= 2 * R(3,2,2)
    # Actually, if we find a decomposition for <3,2,2>, we can use it twice.
    # But we need to map the indices correctly.
    
    # Let's try to find R(3,2,2)
    # Lower bound is 6.
    # Try ranks 11, 10, 9, 8, 7, 6
    
    found_sub = None
    sub_rank = 12
    for r in range(11, 5, -1):
        if time.time() > deadline:
            break
        # Increase trials for lower ranks
        trials = 100 if r >= 10 else 50
        result = _search_rank(3, 2, 2, r, trials, seed=r*100)
        if result:
            found_sub = result
            sub_rank = r
            break
    
    if found_sub:
        # Construct full decomposition using the sub-decomposition
        # We split the 4x4 multiplication into two 2x4 multiplications?
        # No, we split the inner dimension 4 into 2+2.
        # C = A[:, 0:2] B[0:2, :] + A[:, 2:4] B[2:4, :]
        # Let X1 = A[:, 0:2] (3x2), Y1 = B[0:2, :] (2x4).
        # Let X2 = A[:, 2:4] (3x2), Y2 = B[2:4, :] (2x4).
        # We need to compute X1 Y1 and X2 Y2.
        # Each is a <3,2,4> multiplication.
        # We split Y1 into Y1[:, 0:2] and Y1[:, 2:2].
        # So X1 Y1 = X1 Y1[:, 0:2] + X1 Y1[:, 2:2].
        # Each term is <3,2,2>.
        # So we need 4 instances of <3,2,2>.
        # Total rank = 4 * sub_rank.
        
        total_rank = 4 * sub_rank
        if total_rank < best_rank:
            # Construct U, V, W for the full problem
            # U has size 12 x total_rank
            # V has size 16 x total_rank
            # W has size 12 x total_rank
            
            U = [[0] * total_rank for _ in range(12)]
            V = [[0] * total_rank for _ in range(16)]
            W = [[0] * total_rank for _ in range(12)]
            
            # Mapping:
            # Instance 1: X1 (rows 0-1 of A, cols 0-1 of B) * Y1_left (rows 0-1 of B, cols 0-1 of C)
            # X1 is A[:, 0:2]. Indices in U: i*m + j. i in 0..2, j in 0..1.
            # Y1_left is B[0:2, 0:2]. Indices in V: j*p + k. j in 0..1, k in 0..1.
            # Output is C[:, 0:2]. Indices in W: k*n + i. k in 0..1, i in 0..2.
            
            # We have a decomposition for <3,2,2> with rank r.
            # U_sub (3*2 x r), V_sub (2*2 x r), W_sub (2*3 x r).
            # U_sub[i*2+j, c], V_sub[j*2+k, c], W_sub[k*3+i, c].
            
            for block in range(4):
                # block 0: X1 * Y1[:, 0:2]
                # block 1: X1 * Y1[:, 2:2]
                # block 2: X2 * Y2[:, 0:2]
                # block 3: X2 * Y2[:, 2:2]
                
                offset = block * sub_rank
                a_col_start = 0 if block < 2 else 2  # X1 or X2
                b_row_start = 0 if block < 2 else 2  # Y1 or Y2
                c_col_start = 0 if block % 2 == 0 else 2  # Left or Right half of C
                
                for i in range(3):
                    for j in range(2):
                        u_idx = i * 2 + j
                        for c in range(sub_rank):
                            val = found_sub["U"][u_idx][c]
                            if val != 0:
                                # Map to full U index
                                # Full U index: i*m + j_full
                                # j_full = a_col_start + j
                                full_u_idx = i * 4 + (a_col_start + j)
                                U[full_u_idx][offset + c] += val
                
                for j in range(2):
                    for k in range(2):
                        v_idx = j * 2 + k
                        for c in range(sub_rank):
                            val = found_sub["V"][v_idx][c]
                            if val != 0:
                                # Map to full V index
                                # Full V index: j_full * p + k_full
                                # j_full = b_row_start + j
                                # k_full = c_col_start + k
                                full_v_idx = (b_row_start + j) * 4 + (c_col_start + k)
                                V[full_v_idx][offset + c] += val
                
                for k in range(2):
                    for i in range(3):
                        w_idx = k * 3 + i
                        for c in range(sub_rank):
                            val = found_sub["W"][w_idx][c]
                            if val != 0:
                                # Map to full W index
                                # Full W index: k_full * n + i
                                # k_full = c_col_start + k
                                full_w_idx = (c_col_start + k) * 3 + i
                                W[full_w_idx][offset + c] += val
            
            # Verify the full decomposition
            if _verify_fast(U, V, W, 3, 4, 4):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
    
    # If no improvement found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
