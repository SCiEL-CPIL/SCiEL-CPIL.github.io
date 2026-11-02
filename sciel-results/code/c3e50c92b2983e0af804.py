import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    T_recon[ab, cd, ki] = sum_l U[ab,l] V[cd,l] W[ki,l]
    Target T[ab, cd, ki] = 1 if a==i, b==d, c==k else 0
    """
    r = U.shape[1]
    # T shape: (n, m, m, p, n, p)
    # U index: a*m+b
    # V index: c*p+d
    # W index: k*n+i
    
    T_recon = np.zeros((n, m, m, p, n, p))
    for l in range(r):
        u_l = U[:, l].reshape(n, m)
        v_l = V[:, l].reshape(m, p)
        w_l = W[:, l].reshape(p, n)
        # term[ab, cd, ki] = u_l[ab] * v_l[cd] * w_l[ki]
        term = np.einsum('ab,cd,ki->abcdki', u_l, v_l, w_l)
        T_recon += term
        
    T_target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                # T_target[a, b, c, b, a, c] = 1.0
                T_target[a, b, c, b, a, c] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return False
    return True

def get_rank_11_m232():
    """
    Returns U, V, W for M_{2,3,2} with rank 11.
    n=2, m=3, p=2.
    U: (6, 11), V: (6, 11), W: (4, 11)
    """
    # We need a rank 11 decomposition for 2x3 * 3x2.
    # Schoolbook is 12.
    # We can derive this from Strassen-like identities or search.
    # For now, I will use a known minimal decomposition or construct one.
    # A rank 11 decomposition for 2x3x2 exists.
    # Let's try to construct it using the 7-term Strassen for 2x2x2 and handling the extra dimension.
    # Actually, 2x3x2 can be seen as 2x2x2 + 2x1x2.
    # Rank(2x2x2) = 7. Rank(2x1x2) = 2. Total 9? No, they share factors.
    # Let's use a specific hardcoded solution if available, or a simple valid one.
    # If I cannot guarantee rank 11, I will fall back to rank 12 (schoolbook) which gives total rank 36.
    # But the goal is 33.
    # Let's use a known rank 11 solution for 2x3x2.
    # Reference: "The rank of matrix multiplication" etc.
    # I will use a solution derived from the 7-term algorithm for 2x2 and adding 4 terms for the rest?
    
    # Let's try a specific construction:
    # C = A B, A is 2x3, B is 3x2.
    # A = [a11 a12 a13; a21 a22 a23]
    # B = [b11 b12; b21 b22; b31 b32]
    # C = [a11b11+a12b21+a13b31, a11b12+a12b22+a13b32;
    #      a21b11+a22b21+a23b31, a21b12+a22b22+a23b32]
    
    # We can use the following 11 terms (one known minimal decomposition):
    # This is tricky to hardcode without error. 
    # Alternative: Use rank 12 (schoolbook) for each block -> Total 36.
    # But we want 33.
    # Is there a rank 11 solution? Yes.
    # I will implement a search for rank 11 for 2x3x2 if I have time, but for now,
    # I will use a known valid rank 11 decomposition from literature or construct one carefully.
    
    # Let's use a rank 12 solution for safety and see if we can compress.
    # Actually, the prompt says "rank-minimal".
    # If I return 36, it's worse than 34.
    # If I return 33, it's better.
    
    # Let's use the following strategy:
    # The rank of M_{2,3,2} is 11.
    # I will use a hardcoded rank 11 solution.
    # If I can't find it, I'll use a rank 12 solution and hope for a better global strategy?
    # No, 3*12=36 is worse than 34.
    # 3*11=33 is better than 34.
    
    # I will use a known rank 11 decomposition for 2x3x2.
    # One such decomposition is given by:
    # (Source: Various tensor decomposition papers)
    # I will generate a valid one using a simple algorithm if possible, or hardcode a known one.
    
    # Let's try to build it from Strassen 2x2x2 (rank 7) and 2x1x2 (rank 2)?
    # No, that's not how it works.
    
    # I will use a rank 11 solution found in literature.
    # U (6x11), V (6x11), W (4x11)
    
    U = np.zeros((6, 11))
    V = np.zeros((6, 11))
    W = np.zeros((4, 11))
    
    # Term 1-7: Strassen-like for the 2x2 part (cols 0,1 of B, rows 0,1 of A)
    # We treat A as [A1 A2 A3] and B as [B1; B2; B3]
    # This is complex. Let's use a simpler approach:
    # If I can't guarantee rank 11, I should return the best known valid solution.
    # The current best is 34.
    # If I return 36, it's bad.
    # I will use a rank 11 solution if I can verify it.
    
    # Let's use a known rank 11 solution for 2x3x2:
    # (Hardcoded from a known source or derived)
    # I will use the following:
    # This is a guess. I will instead use a rank 12 solution for each block, but try to merge them?
    # No, the blocks are independent.
    
    # Let's use a rank 11 solution.
    # I will use the one from "Optimal algorithms for matrix multiplication" or similar.
    # Actually, I can just use a rank 12 solution and accept 36? No.
    # The current program has 34.
    # I need to find a way to get 33.
    
    # Let's use the fact that M_{2,3,2} has rank 11.
    # I will use a specific decomposition:
    # U, V, W for 2x3x2 rank 11.
    
    # I will use a valid rank 11 decomposition.
    # (I am constructing this based on the structure of the tensor)
    
    # For safety, I will use a rank 12 solution for 2x3x2, which gives 36.
    # But I want 33.
    # I will use a rank 11 solution.
    # Here is a known rank 11 decomposition for 2x3x2:
    
    # U (6x11)
    # V (6x11)
    # W (4x11)
    
    # I will use a placeholder rank 11 solution that I will verify.
    # If it fails, I will fallback to the rank 34 solution from the previous program.
    
    # Let's use a rank 11 solution from a known database.
    # I will use the following:
    
    # Term 0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1
    
    # Term 1
    U[1, 1] = 1; U[4, 1] = 1
    V[1, 1] = 1; V[4, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # Term 2
    U[0, 2] = 1
    V[2, 2] = 1
    W[0, 2] = 1
    
    # Term 3
    U[1, 3] = 1
    V[5, 3] = 1
    W[1, 3] = 1
    
    # Term 4
    U[3, 4] = 1
    V[2, 4] = 1
    W[2, 4] = 1
    
    # Term 5
    U[4, 5] = 1
    V[5, 5] = 1
    W[3, 5] = 1
    
    # Term 6
    U[0, 6] = 1; U[1, 6] = 1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    # Term 7
    U[3, 7] = 1; U[4, 7] = 1
    V[3, 7] = 1; V[4, 7] = 1
    W[2, 7] = 1; W[3, 7] = -1
    
    # Term 8
    U[0, 8] = 1
    V[4, 8] = 1
    W[0, 8] = 1
    
    # Term 9
    U[1, 9] = 1
    V[3, 9] = 1
    W[1, 9] = 1
    
    # Term 10
    U[5, 10] = 1
    V[5, 10] = 1
    W[0, 10] = 1; W[1, 10] = 1; W[2, 10] = 1; W[3, 10] = 1
    
    # This is a guess. I will verify it.
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    # Fallback: Rank 34 Solution from previous program
    def get_rank_34_solution():
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    U_fallback, V_fallback, W_fallback = get_rank_34_solution()
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    best_rank = 34

    # Try to construct Rank 33 solution using 3 blocks of Rank 11 (2x3x2)
    def try_rank_33():
        U232, V232, W232 = get_rank_11_m232()
        
        # Verify the 2x3x2 decomposition
        if not check_decomposition(U232, V232, W232, 2, 3, 2):
            return None, None, None
            
        # Construct 2x3x6 by replicating the 2x3x2 decomposition 3 times
        # for the 3 column blocks of B (and corresponding output blocks of C)
        U_comb = np.zeros((6, 33))
        V_comb = np.zeros((18, 33))
        W_comb = np.zeros((12, 33))
        
        for block in range(3):
            # U is the same for all blocks (A is shared)
            U_comb[:, block*11:(block+1)*11] = U232
            
            # V: B is 3x6. Blocks are columns 0-1, 2-3, 4-5.
            # V index: c*p + d. p=6.
            # For block 0 (cols 0,1): d in {0,1}. c in {0,1,2}.
            # V232 index: c*2 + d (since p=2 for the sub-problem).
            # We need to map V232 indices to V_comb indices.
            for c in range(3):
                for d in range(2):
                    v_idx_sub = c * 2 + d
                    v_idx_full = c * 6 + (block * 2 + d)
                    V_comb[v_idx_full, block*11:(block+1)*11] = V232[v_idx_sub]
                    
            # W: C is 2x6. Blocks are cols 0-1, 2-3, 4-5.
            # W index: k*n + i. n=2. k is row of C? No, k is index in W formula.
            # W[k*n + i]. k corresponds to the row of the output?
            # In the tensor definition: T[ab, cd, ki].
            # W index is k*n + i.
            # k is the row index of the output matrix C?
            # C is 2x6. Rows are 0,1. Cols are 0..5.
            # In the sub-problem 2x3x2:
            # C_sub is 2x2. Rows 0,1. Cols 0,1.
            # W_sub index: k*2 + i.
            # k is the row of C_sub. i is the col of C_sub?
            # Let's check the target tensor definition again.
            # T_target[a, b, c, b, a, c] = 1.
            # a: row of A (0..1)
            # b: col of A / row of B (0..2)
            # c: row of B / col of C? No.
            # B is 3x6. c is row of B (0..2).
            # d: col of B (0..5).
            # i: row of C (0..1).
            # k: col of C (0..5).
            # W index: k*n + i.
            # So k is the column of C, i is the row of C.
            
            # In the sub-problem 2x3x2:
            # k_sub is col of C_sub (0..1).
            # i_sub is row of C_sub (0..1).
            # W_sub index: k_sub*2 + i_sub.
            
            # For block `block` (cols 2*block, 2*block+1 of C):
            # k_full = 2*block + k_sub.
            # i_full = i_sub.
            for k_sub in range(2):
                for i_sub in range(2):
                    w_idx_sub = k_sub * 2 + i_sub
                    w_idx_full = (2 * block + k_sub) * 2 + i_sub
                    W_comb[w_idx_full, block*11:(block+1)*11] = W232[w_idx_sub]
        
        if check_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return U_comb, V_comb, W_comb
        else:
            return None, None, None

    U_33, V_33, W_33 = try_rank_33()
    if U_33 is not None:
        best_U, best_V, best_W = U_33, V_33, W_33
        best_rank = 33

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    construct()
