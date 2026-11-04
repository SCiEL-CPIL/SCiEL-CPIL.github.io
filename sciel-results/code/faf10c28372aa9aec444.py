"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Try to find a low-rank decomposition
    # The tensor T[i*m+j, k*n+i] structure
    # T is a 3D tensor of shape (n*m, m*p, p*n) = (12, 24, 18)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    
    # Let's build the tensor
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1
    
    # Try rank from 40 to 72
    best_rank = n*m*p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Use the structure: split m=4 into 2+2
    # A is 3x4 = [A_left (3x2) | A_right (3x2)]
    # B is 4x6 = [B_top (2x6); B_bottom (2x6)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Each A_left @ B_top is 3x2 @ 2x6 = 3x6
    # We can split B_top into two 2x3 blocks: B_top = [BT1 (2x3) | BT2 (2x3)]
    # Similarly B_bottom = [BB1 (2x3) | BB2 (2x3)]
    
    # So C = A_left@BT1 + A_right@BB1, A_left@BT2 + A_right@BB2
    # Each term is 3x2 @ 2x3 = 3x3
    
    # For 3x2 @ 2x3, let's find the minimal rank
    # This is a bilinear map: (3x2, 2x3) -> 3x3
    # The tensor has rank at most 18
    
    # Let me try to find a rank decomposition for 3x2 @ 2x3
    # Using the fact that we can write this as a sum of rank-1 tensors
    
    # Actually, let me just use a direct approach with random search for the full problem
    
    # Initialize with schoolbook
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
    
    # Now try to reduce rank by finding linear dependencies
    # Check if any columns can be eliminated
    
    # For a given rank r, check if the decomposition is valid
    def is_valid(U_mat, V_mat, W_mat):
        # Check that sum_c U[:,c] * V[:,c] * W[:,c] reconstructs the tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = 0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[a,r] * V_mat[b,r] * W_mat[c,r]
                    if abs(val - 1) > 1e-9:
                        return False
        # Also check that T[a,b,c] = 0 for incompatible (a,b,c)
        # This is harder to check exhaustively, but for the standard tensor it should hold
        return True
    
    # Try to find a lower rank by random projection
    # Use the idea that if we have a valid decomposition of rank r,
    # we can try to find a linear combination that gives a valid decomposition of rank r-1
    
    # For now, let's try a few random rank-reduction attempts
    import random
    random.seed(42)
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        
        # Try to eliminate one column by expressing it as a combination of others
        # Pick a random column to try to eliminate
        if current_rank <= 1:
            break
        
        col_to_elim = random.randint(0, current_rank - 1)
        
        # Try to find coefficients alpha_r such that
        # U[:,col_to_elim] = sum_{r != col_to_elim} alpha_r * U[:,r]
        # and similarly for V and W
        # This is a system of equations
        
        # For U: n*m equations, current_rank-1 unknowns
        # For V: m*p equations, current_rank-1 unknowns  
        # For W: p*n equations, current_rank-1 unknowns
        
        # This is generally overdetermined, so we need a different approach
        
        # Instead, let's try to find a random direction in the null space
        # of the combined system
        
        # Actually, let's use a different strategy: try to find a rank-(r-1) 
        # decomposition by random search
        
        new_rank = current_rank - 1
        if new_rank < 20:
            continue
            
        # Random initialization
        U_new = np.random.randn(n*m, new_rank)
        V_new = np.random.randn(m*p, new_rank)
        W_new = np.random.randn(p*n, new_rank)
        
        # Gradient descent to minimize the error
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U_new[a,:] * V_new[b,:] * W_new[c,:])
                    error += (val - 1)**2
        
        # This is too slow for many iterations, so let's just keep the schoolbook
        # for now and try a smarter approach
    
    # Let me try a structured approach: use Strassen for the 4x4 part
    # Split m=4 into 2+2
    # A = [A1 A2] where A1, A2 are 3x2
    # B = [B1; B2] where B1, B2 are 2x6
    # C = A1@B1 + A2@B2
    
    # For each A_i @ B_i (3x2 @ 2x6), split B_i into two 2x3 blocks
    # B_i = [Bi1 Bi2] where Bi1, Bi2 are 2x3
    # A_i @ B_i = [A_i@Bi1, A_i@Bi2]
    
    # So we have 4 subproblems of type 3x2 @ 2x3
    
    # For 3x2 @ 2x3, the minimal rank is at most 18 (schoolbook)
    # But we can do better: 3x2 @ 2x3 can be seen as computing 3x3 from 
    # two rank-2 matrices
    
    # Let me just use the schoolbook for now and return it
    # The rank is 72
    
    U_list = current_U.tolist()
    V_list = current_V.tolist()
    W_list = current_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
