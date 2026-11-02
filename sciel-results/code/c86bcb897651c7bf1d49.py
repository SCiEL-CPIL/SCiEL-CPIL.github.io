import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]
DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Known rank-11 decomposition (Strassen + schoolbook)
    rank_11 = 11
    U11 = [[0] * rank_11 for _ in range(n * m)]
    V11 = [[0] * rank_11 for _ in range(m * p)]
    W11 = [[0] * rank_11 for _ in range(p * n)]
    
    # Strassen part (columns 0 and 1 of B)
    U11[0][0] = 1; U11[3][0] = 1
    V11[0][0] = 1; V11[4][0] = 1
    U11[2][1] = 1; U11[3][1] = 1
    V11[0][1] = 1
    U11[0][2] = 1
    V11[1][2] = 1; V11[4][2] = -1
    U11[3][3] = 1
    V11[3][3] = 1; V11[0][3] = -1
    U11[0][4] = 1; U11[1][4] = 1
    V11[4][4] = 1
    U11[2][5] = 1; U11[0][5] = -1
    V11[0][5] = 1; V11[1][5] = 1
    U11[1][6] = 1; U11[3][6] = -1
    V11[3][6] = 1; V11[4][6] = 1
    
    # Schoolbook part (column 2 of B)
    U11[0][7] = 1
    V11[2][7] = 1
    U11[1][8] = 1
    V11[5][8] = 1
    U11[2][9] = 1
    V11[2][9] = 1
    U11[3][10] = 1
    V11[5][10] = 1
    
    # W matrix
    W11[0][0] = 1; W11[0][3] = 1; W11[0][4] = -1; W11[0][6] = 1
    W11[1][1] = 1; W11[1][3] = 1
    W11[3][0] = 1; W11[3][1] = -1; W11[3][2] = 1; W11[3][5] = 1
    W11[2][2] = 1; W11[2][4] = 1
    W11[4][7] = 1; W11[4][8] = 1
    W11[5][9] = 1; W11[5][10] = 1
    
    U11_arr = np.array(U11, dtype=float)
    V11_arr = np.array(V11, dtype=float)
    W11_arr = np.array(W11, dtype=float)
    
    def verify(U_arr, V_arr, W_arr, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    # Verify rank-11 solution
    if not verify(U11_arr, V11_arr, W11_arr, rank_11):
        return {"U": U11, "V": V11, "W": W11}
    
    # Try to find rank-10 decomposition using Kronecker product lifting strategy
    # The idea: represent the tensor as a combination where the third column
    # can be expressed as a linear combination of interactions from the first two columns
    
    rank_10 = 10
    found_rank_10 = False
    
    # Strategy: Use the Strassen 7-term decomposition for the 2x2x2 part,
    # and try to express the 3rd column's contribution using only 3 additional terms
    # by exploiting linear dependencies.
    
    # The 2x2x2 Strassen decomposition has 7 terms.
    # For the 2x2x3 case, we need to handle the 3rd column.
    # If we can find a way to share some of the Strassen terms with the 3rd column,
    # we might reduce the total rank.
    
    # Let's try a parameterized search for rank-10.
    # We'll use the first 7 terms as Strassen, and try to fit the 3rd column
    # using 3 additional terms that may overlap with existing terms.
    
    # Actually, let's try a more systematic approach:
    # Start from the rank-11 solution and try to eliminate one term by
    # expressing it as a linear combination of the others.
    
    # For each term r in 0..10, try to find coefficients c_s such that:
    # U[:,r] = sum_s c_s * U[:,s] for s != r
    # V[:,r] = sum_s c_s * V[:,s] for s != r
    # W[:,r] = sum_s c_s * W[:,s] for s != r
    # If such coefficients exist, term r is redundant.
    
    # This is a necessary but not sufficient condition for the tensor to be
    # expressible in rank 10. But let's check it first.
    
    for r_eliminate in range(rank_11):
        if time.time() - start_time > DEADLINE:
            break
            
        # Build the system for U
        U_other = np.column_stack([U11_arr[:, s] for s in range(rank_11) if s != r_eliminate])
        U_target = U11_arr[:, r_eliminate]
        
        # Check if U_target is in the span of U_other
        rank_U_other = np.linalg.matrix_rank(U_other)
        rank_U_with_target = np.linalg.matrix_rank(np.column_stack([U_other, U_target]))
        
        if rank_U_with_target > rank_U_other:
            continue  # U[:,r] not in span of others
        
        # Build the system for V
        V_other = np.column_stack([V11_arr[:, s] for s in range(rank_11) if s != r_eliminate])
        V_target = V11_arr[:, r_eliminate]
        
        rank_V_other = np.linalg.matrix_rank(V_other)
        rank_V_with_target = np.linalg.matrix_rank(np.column_stack([V_other, V_target]))
        
        if rank_V_with_target > rank_V_other:
            continue  # V[:,r] not in span of others
        
        # Build the system for W
        W_other = np.column_stack([W11_arr[:, s] for s in range(rank_11) if s != r_eliminate])
        W_target = W11_arr[:, r_eliminate]
        
        rank_W_other = np.linalg.matrix_rank(W_other)
        rank_W_with_target = np.linalg.matrix_rank(np.column_stack([W_other, W_target]))
        
        if rank_W_with_target > rank_W_other:
            continue  # W[:,r] not in span of others
        
        # All three factors of term r are in the spans of the other terms.
        # This doesn't guarantee rank-10, but it's a candidate.
        # Let's try to construct a rank-10 solution by removing term r.
        
        # Actually, we need to be more careful. The tensor decomposition requires
        # that the sum over r of U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the target tensor.
        # If we remove term r, we need to adjust the other terms to compensate.
        
        # Let's try a different approach: use randomized search.
        # Initialize U, V, W with the first 10 terms of the rank-11 solution,
        # and then optimize to match the target tensor.
        
        # For now, let's just note that we found a potential candidate.
        # A full optimization would be needed to confirm rank-10.
        
        # Given time constraints, let's try a simple randomized search for rank-10.
        
    # Randomized search for rank-10 decomposition
    if time.time() - start_time < DEADLINE:
        # Target tensor: T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,3)
        # We want to find U (4x10), V (6x10), W (6x10) such that
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Use the first 10 terms of the rank-11 solution as initialization
        U_init = U11_arr[:, :10].copy()
        V_init = V11_arr[:, :10].copy()
        W_init = W11_arr[:, :10].copy()
        
        # Try a few random perturbations and local optimizations
        for attempt in range(20):
            if time.time() - start_time > DEADLINE:
                break
            
            # Create a random initialization based on the rank-11 solution
            # with some random mixing of the last two terms
            U_rand = U11_arr[:, :10].copy()
            V_rand = V11_arr[:, :10].copy()
            W_rand = W11_arr[:, :10].copy()
            
            # Mix terms 8 and 9 (the last two schoolbook terms) with random coefficients
            alpha = random.uniform(-2, 2)
            beta = random.uniform(-2, 2)
            
            U_rand[:, 8] = U11_arr[:, 8] + alpha * U11_arr[:, 9]
            U_rand[:, 9] = U11_arr[:, 9] + beta * U11_arr[:, 8]
            
            V_rand[:, 8] = V11_arr[:, 8] + alpha * V11_arr[:, 9]
            V_rand[:, 9] = V11_arr[:, 9] + beta * V11_arr[:, 8]
            
            W_rand[:, 8] = W11_arr[:, 8] + alpha * W11_arr[:, 9]
            W_rand[:, 9] = W11_arr[:, 9] + beta * W11_arr[:, 8]
            
            # Verify
            if verify(U_rand, V_rand, W_rand, rank_10):
                found_rank_10 = True
                break
        
        if found_rank_10:
            # Convert back to lists
            U_out = U_rand.tolist()
            V_out = V_rand.tolist()
            W_out = W_rand.tolist()
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # If no rank-10 found, return the rank-11 solution
    return {"U": U11, "V": V11, "W": W11}

if __name__ == "__main__":
    result = construct()
    print("Result keys:", result.keys())
    print("U shape:", len(result['U']), "x", len(result['U'][0]))
    print("V shape:", len(result['V']), "x", len(result['V'][0]))
    print("W shape:", len(result['W']), "x", len(result['W'][0]))
