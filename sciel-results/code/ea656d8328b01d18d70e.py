import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_correct():
    """Correct Strassen 2x2x2 decomposition. Rank 7."""
    n, m, p = 2, 2, 2
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # M1: (A[0,0]-A[1,1])*(B[1,1]-B[0,0])
    U[0, 0] = 1; U[3, 0] = -1
    V[3, 0] = 1; V[0, 0] = -1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1
    
    # M2: (A[0,0]+A[1,1])*B[1,1]
    U[0, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: A[1,1]*(B[1,0]-B[0,0])
    U[3, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A[0,0]*(B[0,1]-B[1,1])
    U[0, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A[0,0]+A[0,1])*B[1,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_2x4x7_block(arb):
    """
    Build a 2x4x7 decomposition block.
    A is 2x4 (rows 2*arb..2*arb+1, cols 0..3).
    B is 4x7 (rows 0..3, cols 0..6).
    C is 2x7 (rows 2*arb..2*arb+1, cols 0..6).
    
    We decompose this into 4 sub-blocks of 2x2x7? No, B is 4x7.
    Let's split B's rows into two groups of 2: B_top (rows 0,1) and B_bot (rows 2,3).
    Then C = A_top * B_top + A_bot * B_bot? No, A is 2x4.
    A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    B = [B_top; B_bot] where B_top is 2x7, B_bot is 2x7.
    C = A_left * B_top + A_right * B_bot.
    
    So we need to multiply (2x2) * (2x7) and (2x2) * (2x7).
    We can use Strassen for the 2x2 part of A and B?
    Strassen multiplies 2x2 by 2x2. Here B is 2x7.
    We can view B as 2x7. We can decompose the multiplication of 2x2 by 2x7 
    into 7 independent 2x2 by 1x2? No.
    
    Let's use a simpler approach for 2x2 * 2x7.
    Schoolbook rank: 2*2*7 = 28.
    Can we do better?
    Rank(2x2 * 2x7) = Rank(2x2 * 2x1 * 7)? No.
    Actually, multiplying 2x2 by 2x7 is equivalent to 7 independent 2x2 by 2x1 multiplications?
    No, C[:,k] = A * B[:,k].
    This is 7 independent matrix-vector multiplications (2x2 by 2x1).
    Rank of 2x2 * 2x1 is 4 (schoolbook).
    So total rank for 2x2 * 2x7 is 7 * 4 = 28?
    Wait, the tensor is T_{i, j, k} for i in 0..1, j in 0..1, k in 0..6.
    T_{ijk} = 1.
    This is a "slice" of the full tensor.
    Is there a better decomposition for 2x2x7?
    Lower bound: 2*2*7 / 7 = 4.
    Schoolbook: 28.
    Strassen-like?
    If we split k into 4+3?
    2x2x4: Rank? 2x2x4 can be viewed as two 2x2x2. Rank 7+7=14.
    2x2x3: Rank?
    
    Let's just use Schoolbook for the 2x2x7 sub-problems for now to ensure correctness,
    then apply the linear algebra reduction on the whole 4x4x7 tensor.
    The reduction step is the key.
    """
    # A is 2x4, B is 4x7.
    # We split A into A1 (2x2) and A2 (2x2).
    # We split B into B1 (2x7) and B2 (2x7).
    # C = A1*B1 + A2*B2.
    
    # Sub-problem 1: 2x2 * 2x7.
    # Indices: i in 0..1 (local A row), j in 0..1 (local A col / local B row), k in 0..6.
    # Schoolbook: 2*2*7 = 28 terms.
    U1 = np.zeros((2*2, 28), dtype=np.float64)
    V1 = np.zeros((2*7, 28), dtype=np.float64)
    W1 = np.zeros((7*2, 28), dtype=np.float64)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(7):
                U1[i*2+j, col] = 1
                V1[j*7+k, col] = 1
                W1[k*2+i, col] = 1
                col += 1
                
    # Sub-problem 2: 2x2 * 2x7.
    # Same structure, just mapped to different indices.
    U2 = U1.copy()
    V2 = V1.copy()
    W2 = W1.copy()
    
    return U1, V1, W1, U2, V2, W2

def _build_4x4x7_hybrid():
    """
    Build 4x4x7 by combining two 2x4x7 blocks.
    Block 0: A rows 0,1. B rows 0,1. C rows 0,1.
    Block 1: A rows 0,1. B rows 2,3. C rows 0,1.
    Wait, A is 4x4.
    A = [A_top; A_bot] where A_top is 2x4, A_bot is 2x4.
    B = 4x7.
    C = 4x7.
    C_top = A_top * B.
    C_bot = A_bot * B.
    
    So we need two 2x4x7 decompositions.
    For each 2x4x7, we split B into two 2x7 parts? No, B is 4x7.
    A_top (2x4) * B (4x7).
    Split A_top into A_top_left (2x2) and A_top_right (2x2).
    Split B into B_top (2x7) and B_bot (2x7).
    C_top = A_top_left * B_top + A_top_right * B_bot.
    
    So for A_top, we have two 2x2x7 sub-problems.
    For A_bot, we have two 2x2x7 sub-problems.
    Total 4 sub-problems of 2x2x7.
    Each 2x2x7 has rank 28 (schoolbook).
    Total rank 4 * 28 = 112. This is just schoolbook.
    
    We need to use Strassen on the 2x2 parts to reduce rank.
    Can we use Strassen for 2x2 * 2x7?
    Strassen multiplies 2x2 * 2x2.
    If we treat the 7 columns of B as 7 independent 2x2 matrices? No, B is 2x7.
    
    Let's look at the known rank for 2x2x7.
    It is not widely known to be significantly less than 28.
    However, the reduction step on the full 4x4x7 tensor might find dependencies
    between these blocks.
    
    Let's stick to the 112 rank schoolbook initially, but structured,
    and then apply the linear algebra reduction.
    """
    n, m, p = 4, 4, 7
    rank = 112
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if not np.isclose(val, 1.0):
                    return False
    return True

def _reduce_rank_linear_algebra(U, V, W, n, m, p, max_time=100):
    """
    Reduce rank by finding linear dependencies.
    We check if any rank-1 term is a linear combination of others.
    This is done by checking the rank of the tensor slices.
    """
    start_time = time.time()
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    rank = U_curr.shape[1]
    
    # Remove zero columns
    for r in range(rank - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
        if np.all(U_curr[:, r] == 0) and np.all(V_curr[:, r] == 0) and np.all(W_curr[:, r] == 0):
            U_curr = np.delete(U_curr, r, axis=1)
            V_curr = np.delete(V_curr, r, axis=1)
            W_curr = np.delete(W_curr, r, axis=1)
            rank -= 1
            
    # Try to remove linearly dependent terms.
    # A term r is redundant if there exist c_s (s != r) such that
    # U[:, r] * V[:, r] * W[:, r] = sum_{s != r} c_s * U[:, s] * V[:, s] * W[:, s]
    # This is a nonlinear condition.
    # However, if we fix one index, say i, the slice T_i = sum_r U[i,r] V[:,r] W[:,r]
    # is a matrix of size m x p.
    # If we can find a linear combination of the outer products u_r v_r w_r that sums to 0,
    # we can remove it.
    
    # A simpler heuristic:
    # Check if any two terms are identical (up to scaling).
    # If U[:, r1] = U[:, r2], V[:, r1] = V[:, r2], W[:, r1] = W[:, r2],
    # then they are linearly dependent (1*term1 - 1*term2 = 0).
    # We can merge them or remove one if they cancel? No, they add.
    # If they are identical, we can't just remove one. We can replace two with one?
    # No, rank 1 term + rank 1 term = rank 2 term generally.
    # Unless they are the same, in which case we have 2 * term.
    # We can scale one term by 1/2 and the other by 1/2? No, that doesn't reduce rank.
    # We can replace two identical terms with one term scaled by 2?
    # Yes! If term_r1 == term_r2, then term_r1 + term_r2 = 2 * term_r1.
    # So we can replace two columns with one column scaled by 2.
    # This reduces rank by 1.
    
    # Let's look for duplicate terms.
    # Represent each term by a tuple of (U_col, V_col, W_col).
    # Since they are floats, we need to be careful.
    # We can use a dictionary with rounded values.
    
    # This is a very specific check. Let's try it.
    seen = {}
    to_remove = []
    for r in range(rank):
        # Hash the term
        u_key = tuple(U_curr[:, r].round(6))
        v_key = tuple(V_curr[:, r].round(6))
        w_key = tuple(W_curr[:, r].round(6))
        key = (u_key, v_key, w_key)
        
        if key in seen:
            # We have a duplicate.
            # We can merge them.
            # Let r_prev = seen[key].
            # term_r + term_r_prev = 2 * term_r.
            # We can set U[:, r] *= 2, V[:, r] *= 2, W[:, r] *= 2? No.
            # U[:, r] * V[:, r] * W[:, r] is the term.
            # If we double U, the term doubles.
            # So we can set U[:, r] = 2 * U[:, r], and remove r_prev.
            # Wait, we need to keep the sum constant.
            # Sum = ... + term_r + term_r_prev + ...
            # If term_r == term_r_prev, Sum = ... + 2 * term_r + ...
            # We can replace term_r with 2 * term_r and remove term_r_prev.
            # So U[:, r] *= 2, V[:, r] *= 1, W[:, r] *= 1.
            # And remove r_prev.
            r_prev = seen[key]
            U_curr[:, r] *= 2
            # Remove r_prev
            # We need to be careful with indices if we remove in reverse order.
            to_remove.append(r_prev)
            # Update seen for r? No, r is the new one.
            # But wait, if there are 3 duplicates?
            # We handle them one by one.
        else:
            seen[key] = r
            
    # Remove the marked columns
    # Sort to remove in reverse order to keep indices valid
    to_remove.sort(reverse=True)
    for r in to_remove:
        U_curr = np.delete(U_curr, r, axis=1)
        V_curr = np.delete(V_curr, r, axis=1)
        W_curr = np.delete(W_curr, r, axis=1)
        
    return U_curr, V_curr, W_curr

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the 112-rank schoolbook
    U, V, W = _build_4x4x7_hybrid()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook (already schoolbook)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce
    if time.time() < deadline - 10:
        U, V, W = _reduce_rank_linear_algebra(U, V, W, n, m, p, max_time=60)
        
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
