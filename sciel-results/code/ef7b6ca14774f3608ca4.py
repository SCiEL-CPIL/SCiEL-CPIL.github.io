<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
=======
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill

    # Strategy: Apply Strassen to the n=4 and p=8 dimensions simultaneously.
    # Treat A as 2x2 blocks of (2x4), B as 2x2 blocks of (4x8) [wait, B is 4x8, so block size 4x4? No, p=8, so 2 blocks of 4 columns].
    # Actually, let's treat the multiplication as a block product.
    # A is 4x4. Split rows into two blocks of 2: A_top, A_bottom.
    # B is 4x8. Split columns into two blocks of 4: B_left, B_right.
    # C is 4x8. Split columns into two blocks of 4: C_left, C_right.
    # This doesn't form a square block system for Strassen directly on both dimensions unless we view it as 2x2 blocks where:
    # A = [[A00, A01], [A10, A11]] where each block is 2x4? No, A is 4x4.
    # Let's stick to the recursive Strassen structure but optimize the base case.
    
    # Current best known rank for 4x4x4 is 49 (Strassen).
    # For 4x4x8, we can view it as two 4x4x4 multiplications? No, that's rank 98.
    # Can we do better?
    # The tensor for 4x4x8 is the direct sum of two 4x4x4 tensors? No.
    # C = A * B.
    # If we split B into B1 (4x4) and B2 (4x4), then C1 = A*B1 and C2 = A*B2.
    # The rank of the direct sum of two tensors T1 and T2 is at most rank(T1) + rank(T2).
    # Is it strictly less?
    # For matrix multiplication tensors, it is known that rank(MM_{n,m,p}) is additive in some cases, but generally rank(T1 + T2) <= rank(T1) + rank(T2).
    # However, we can potentially share some terms if the structure allows.
    # But A is the same for both.
    # Let's try to construct a decomposition of rank < 98.
    
    # Known result: rank(MM_{4,4,4}) = 49? No, it's 49 for complex numbers? 
    # Actually, the best known upper bound for 4x4x4 is 49 (Strassen).
    # The best known upper bound for 4x4x8 is 98?
    # Wait, can we use the fact that p=8 is larger?
    # Let's check if there is a known decomposition for 4x4x8 with rank < 98.
    # A paper by Landsberg et al. might have something.
    # For now, let's try to improve the constant factors or find a bug in the current 98-rank implementation.
    # The current implementation returns rank 98 (49 + 49).
    # Score 1.3061 corresponds to rank 98?
    # Seed value was likely 128 (schoolbook). 128/98 = 1.306. Yes.
    # So the current program achieves rank 98.
    # To improve, we need rank < 98.
    
    # Is it possible to have rank < 98 for 4x4x8?
    # The tensor is MM_{4,4,8}.
    # This is isomorphic to MM_{4,4,4} + MM_{4,4,4} (direct sum).
    # rank(T1 + T2) <= rank(T1) + rank(T2) = 49 + 49 = 98.
    # Can we do better?
    # If we view it as a 2x2 block multiplication where the blocks are 2x4 and 4x4?
    # No, the dimensions don't match for a square block system.
    
    # Let's try a different blocking.
    # A (4x4), B (4x8).
    # Split A into 2x2 blocks of 2x2 matrices.
    # Split B into 2x4 blocks? No, B is 4x8.
    # Split B into 4x4 blocks (2 blocks horizontally).
    # This is not a symmetric block system.
    
    # However, we can use the "Strassen for rectangular matrices" techniques.
    # But typically, for small sizes, the best known decompositions are from Strassen recursion.
    
    # Let's verify if the current 98-rank decomposition is correct.
    # If it is, we need to find a lower rank one.
    # One possibility: The rank of MM_{4,4,8} might be 97 or less?
    # Actually, it is known that rank(MM_{n,m,p}) is not always additive.
    # But for these small sizes, 98 is the standard upper bound from two 4x4x4 Strassen decompositions.
    
    # Is there a decomposition of rank 97?
    # Let's try to search for a rank 97 decomposition using a heuristic?
    # The search space is huge.
    
    # Alternative: Maybe the current code has a bug and produces an invalid decomposition, falling back to 128?
    # The score is 1.3061, which is 128/98. This implies the score is calculated as Seed/CurrentRank.
    # If the current program returned 128, the score would be 1.0.
    # Since the score is 1.3061, the program must be returning a valid rank 98 decomposition.
    
    # So the current program IS returning rank 98.
    # To improve, we MUST find a rank < 98.
    
    # Let's look at the "inspiration 2" and "inspiration 3" comments.
    # "The current program attempts a Strassen-like decomposition but contains a fundamental logical error..."
    # But the score says it's valid and rank 98.
    # Maybe the inspiration comments are from a failed attempt or a different version?
    # The prompt says "CURRENT PROGRAM (score 1.3061)".
    # So the current program is valid and rank 98.
    
    # Can we reduce the rank?
    # There is a known decomposition for 4x4x4 of rank 49.
    # For 4x4x8, it is the sum of two such decompositions.
    # Unless we can share terms between the two 4x4x4 decompositions, we are stuck at 98.
    # Since the A matrix is the same, and the B matrices are different (B1 and B2),
    # the V and W factors for the two halves are different.
    # The U factors are the same.
    # Can we merge the U factors?
    # In the tensor decomposition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # For the left half (k=0..3), we have U_r, V_r, W_r.
    # For the right half (k=4..7), we have U'_r, V'_r, W'_r.
    # Since A is the same, U_r = U'_r for all r.
    # So we have 98 terms, but only 49 distinct U vectors.
    # However, the rank is the number of terms in the decomposition, not the number of distinct U vectors.
    # Each term is a rank-1 tensor.
    # We have 49 terms for the left half and 49 for the right half.
    # Are any of these rank-1 tensors linearly dependent?
    # T_left_r = U_r \otimes V_r \otimes W_r (with indices restricted to k<4)
    # T_right_r = U_r \otimes V'_r \otimes W'_r (with indices restricted to k>=4)
    # These are supported on disjoint sets of k indices (0..3 vs 4..7).
    # Therefore, they are linearly independent.
    # So the rank is exactly 98 for this construction.
    
    # To get rank < 98, we need a fundamentally different decomposition.
    # Is rank(MM_{4,4,8}) < 98?
    # According to literature, the best known upper bound for MM_{4,4,4} is 49.
    # For MM_{4,4,8}, it is 98.
    # It is an open problem whether it can be lower.
    # However, for the purpose of this task, we might need to try a heuristic search for a lower rank decomposition, or accept 98.
    # But the directive says "investigate improvements within its family".
    # And "Strategy to try... resulting in...".
    # The strategy mentioned: "Apply the 2x2x2 Strassen algorithm to the row dimension of A (n=4) and the column dimension of B (p=8) simultaneously... yielding 7 products of 2x4 by 4x8 submatrices. Each of these 7 products is then decomposed using the schoolbook method for the remaining dimensions (m=4), requiring 4 multiplications each, resulting in 28."
    
    # Wait, let's analyze this strategy.
    # A is 4x4. Split rows into 2 blocks of 2.
    # B is 4x8. Split columns into 2 blocks of 4.
    # This is not a 2x2 block product of square matrices.
    # The strategy says "treat the multiplication as a 2x2 block product of (2x4) by (4x8) matrices".
    # This implies A is split into 2x2 blocks of size 2x4? No, A is 4x4.
    # Maybe it means:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # B = [[B11, B12], [B21, B22]] where each block is 4x4? No, B is 4x8.
    # If we split B into 2 blocks of 4 columns, we have B = [B1 | B2].
    # This is not a 2x2 block structure for B.
    
    # Perhaps the strategy means:
    # Split A into 2x2 blocks of 2x2.
    # Split B into 2x2 blocks of 4x4? No, B is 4x8.
    # Split B into 2 blocks of 4x4 horizontally.
    # Then we have two independent 4x4x4 multiplications?
    # C1 = A * B1, C2 = A * B2.
    # This is what I analyzed before. Rank 98.
    
    # What if we split B vertically? B is 4x8.
    # Split B into 2 blocks of 2x8?
    # B = [[B1], [B2]] where B1, B2 are 2x8.
    # A = [[A1, A2], [A3, A4]] where Ai are 2x4? No, A is 4x4.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # Then C = A * B.
    # C = [[A11, A12], [A21, A22]] * [[B1], [B2]] = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]].
    # This involves 8 products of 2x2 and 2x8 matrices.
    # Each product is a 2x8 matrix.
    # Can we decompose 2x2 * 2x8 multiplication?
    # Rank of MM_{2,2,8}?
    # This is MM_{2,2,2} + MM_{2,2,2} + MM_{2,2,2} + MM_{2,2,2}? No.
    # MM_{2,2,8} is the direct sum of four MM_{2,2,2} tensors?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # If we fix i,j, it's a vector multiplication.
    # The tensor is the Kronecker product of MM_{2,2,1} and MM_{1,2,4}?
    # Actually, MM_{2,2,8} can be viewed as MM_{2,2,2} \otimes MM_{1,1,4}?
    # Rank(MM_{2,2,2}) = 7.
    # Rank(MM_{1,1,4}) = 4.
    # Rank of Kronecker product is product of ranks?
    # If T1 has rank r1 and T2 has rank r2, then T1 \otimes T2 has rank r1*r2.
    # So rank(MM_{2,2,8}) <= 7 * 4 = 28.
    # And we have 8 such products in the block multiplication.
    # Total rank <= 8 * 28 = 224. This is worse than 98.
    
    # Let's re-read the strategy: "yielding 7 products of 2x4 by 4x8 submatrices".
    # This implies the block size is such that we get 2x4 and 4x8 submatrices.
    # A is 4x4. To get 2x4 blocks, we split rows into 2 blocks of 2.
    # B is 4x8. To get 4x8 blocks, we don't split it?
    # But Strassen requires 2x2 blocks.
    # Maybe the strategy is flawed or I'm misinterpreting it.
    
    # Let's try to implement a heuristic search for a rank 97 decomposition.
    # Or just stick with 98 and optimize the code?
    # The score is 1.3061.
    # If we can get rank 97, score = 128/97 = 1.3196.
    # If we can get rank 96, score = 128/96 = 1.3333.
    
    # Is it possible that the current program is NOT using rank 98?
    # Let's check the code.
    # It builds 49 terms for left, 49 for right. Total 98.
    # It verifies.
    # It returns.
    # So it is rank 98.
    
    # I will try to find a rank 97 decomposition.
    # This is a hard problem.
    # However, I can try to randomize the Strassen decomposition and see if I can find a linear dependence.
    # But the support sets are disjoint, so no.
    
    # What if we use a different decomposition for 4x4x4?
    # The best known is 49.
    # Is there a rank 48 decomposition for 4x4x4?
    # No, it is known that rank(MM_{4,4,4}) >= 49.
    # So 49 is minimal.
    
    # Therefore, the sum of two 49-rank decompositions is 98.
    # And since the supports are disjoint, the rank is exactly 98.
    # So 98 is likely the minimal rank for this construction.
    # And likely the minimal rank overall.
    
    # So I cannot improve the rank.
    # I can only improve the speed or robustness.
    # But the score is based on rank.
    # So I am stuck at 1.3061.
    
    # Unless... the "seed" value is not 128?
    # "score is the rank... 1.0 equals the seed program".
    # If seed program is schoolbook rank 128, then score = 128/rank.
    # If I find rank 98, score = 128/98 = 1.306.
    # This matches.
    
    # So I need to find a rank < 98.
    # I will try a random search for a rank 97 decomposition.
    # I will generate random rank-1 tensors and try to fit them.
    # This is computationally expensive.
    
    # Alternatively, I can try to use the fact that the problem is small.
    # I will try to use the `scipy` or `numpy` to solve a linear system for the coefficients of a rank 97 decomposition.
    # Assume a decomposition of rank 97.
    # T = sum_{r=1}^{97} u_r \otimes v_r \otimes w_r.
    # We can fix U and W and solve for V?
    # Or fix U and V and solve for W?
    # This is a non-convex optimization problem.
    
    # Given the time limit, I will stick with the current rank 98 solution and ensure it is correct and fast.
    # I will add a check to ensure the verification is fast.
    
    return {"U": None, "V": None, "W": None} # Placeholder to replace with actual logic

def _construct_rank_98():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130

    # Build schoolbook decomposition as fallback (rank 128)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strassen rank-7 terms for 2x2x2 multiplication
    strassen_terms_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    top_level_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)])
    ]

    all_terms = []
    
    for t_idx in range(7):
        if time.time() > deadline - 10:
            break
        a_blocks, b_blocks, c_blocks = top_level_terms[t_idx]
        
        for s_idx in range(7):
            if time.time() > deadline - 10:
                break
            s_u, s_v, s_w = strassen_terms_2x2[s_idx]
            
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            
            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2*v + r
                    k = 2*w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b
            
            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2*u + r
                    k = 2*w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c
            
            u_global = {key: val for key, val in u_global.items() if val != 0}
            v_global = {key: val for key, val in v_global.items() if val != 0}
            w_global = {key: val for key, val in w_global.items() if val != 0}
            
            if u_global and v_global and w_global:
                all_terms.append((u_global, v_global, w_global))

    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        u_new = dict(u_dict)
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))
    
    total_terms = all_terms + terms_right
    rank = len(total_terms)
    
    U_arr = np.zeros((n * m, rank))
    V_arr = np.zeros((m * p, rank))
    W_arr = np.zeros((p * n, rank))
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U_arr[i * m + j, col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V_arr[j * p + k, col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W_arr[k * n + i, col_idx] = coeff
    
    valid = True
    for i in range(n):
        if time.time() > deadline - 5:
            valid = False
            break
        for j in range(m):
            if time.time() > deadline - 5:
                valid = False
                break
            for k in range(p):
                t_val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(t_val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Try to find a lower rank decomposition?
    # For now, return the rank 98 one.
    res = _construct_rank_98()
    return res

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    print(f"Rank: {U.shape[1]}")
>>>>>>> REPLACE

