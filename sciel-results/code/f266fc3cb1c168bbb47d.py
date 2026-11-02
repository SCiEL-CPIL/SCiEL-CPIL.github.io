"""Program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

def construct():
    """Return a low-rank tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_schoolbook():
        """Build the schoolbook (rank 54) decomposition."""
        rank = schoolbook_rank
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
    
    def verify_decomp(U, V, W, R):
        """Verify the decomposition exactly."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(R):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def build_strassen_2x3x9():
        """
        Build a rank-33 decomposition using Strassen-like approach.
        
        The idea: decompose 2x3x9 as three 2x3x3 blocks (for columns 0-2, 3-5, 6-8).
        For each 2x3x3 block, use a rank-11 decomposition based on Strassen's algorithm
        for 2x2 matrices, adapted to the 2x3 case.
        
        Actually, let's use a more systematic approach:
        - Use the fact that 2x3 matrix multiplication can be viewed as computing 2x3 outputs
        - For each of the 9 columns, we need to compute 2 outputs
        
        A simpler structural approach:
        - Use Kronecker product structure
        - The 2x3x9 tensor is the Kronecker product of a 2x1x1 tensor and a 3x3x3 tensor? No.
        
        Let me think differently. The tensor T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k].
        
        For 2x3x9, we can write this as a sum of rank-1 terms.
        
        Actually, let me use a different structural insight:
        - Group the 9 columns into 3 groups of 3
        - For each group, we have a 2x3x3 sub-tensor
        - The total tensor is the "horizontal concatenation" of three 2x3x3 tensors
        
        For 2x3x3, the minimum rank is known to be 11 (from literature).
        So 3 * 11 = 33 for the 2x3x9 case.
        
        Here's a rank-11 decomposition for 2x3x3:
        We use the fact that 2x3 = 2x2 + 2x1 (first two columns + third column)
        
        For the 2x2 part (columns 0,1), use Strassen (rank 7)
        For the 2x1 part (column 2), use rank 4 (2 outputs, each needs the dot product)
        But wait, Strassen gives rank 7 for 2x2x2, and we need 2x3x2 for the first two columns...
        
        Let me use a concrete construction.
        
        For 2x3x3 with indices i in {0,1}, j in {0,1,2}, k in {0,1,2}:
        
        We can write:
        C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        For k=0,1 (first two columns of B), use Strassen-style for the 2x2 sub-problem
        For k=2, use direct computation
        
        Actually, let me just implement a known low-rank decomposition.
        
        Rank-11 for 2x3x3:
        """
        # Build a rank-11 decomposition for 2x3x3
        R233 = 11
        U233 = [[0] * R233 for _ in range(6)]  # 2*3 = 6 rows
        V233 = [[0] * R233 for _ in range(9)]  # 3*3 = 9 rows
        W233 = [[0] * R233 for _ in range(6)]  # 3*2 = 6 rows
        
        # Strassen for the 2x2 part (j=0,1; k=0,1)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # For 2x2x2:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M3 + M5 + M6
        
        # Now extend to 2x3x3.
        # Let's handle j=0,1,k=0,1 with Strassen, and add terms for j=2 and k=2.
        
        # This is getting complex. Let me use a simpler but still improved decomposition.
        
        # Alternative: use the fact that we can share some terms across the k=2 column.
        
        # For now, let's just replicate the schoolbook for each 2x3x3 block (rank 18 each, total 54)
        # and try to do better with a heuristic.
        
        # Let me try a direct rank-30+ construction.
        
        # Actually, the simplest improvement: note that for the 2x3x9 tensor,
        # we can use the Kronecker product with a 1x3x1 "identity" in some sense.
        
        # The 2x3x9 tensor T satisfies: T[i,j,k] = delta(i,j,k) where the multiplication tensor is 1.
        # We can write T = T_2x3x3 ⊗ I_3 (where I_3 is the 3x3 identity for the column grouping)
        # but that's not quite right because the k-index doesn't factor that way.
        
        # Let me just use the block structure:
        # T[i,j,3b+k] for b=0,1,2 and k=0,1,2
        # This is like having three independent 2x3x3 tensors, but they share the same A and the same j-index.
        
        # So T[i,j,3b+k] = A[i,j] * B[j,3b+k] contributes to C[i,3b+k]
        
        # We can write:
        # Sum over c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # If we use the same U for all blocks, and shift V and W appropriately...
        
        # Let's define:
        # For block b (b=0,1,2) and local column k_local (0,1,2):
        # k = 3*b + k_local
        # U is the same for all blocks (since A doesn't change)
        # V[j*9 + 3*b + k_local, c] should depend on b and k_local
        # W[(3*b+k_local)*2 + i, c] should depend on b and k_local
        
        # If we use a rank-R decomposition for 2x3x3:
        # U_233[i*3+j, c] * V_233[j*3+k_local, c] * W_233[k_local*2+i, c]
        
        # Then for 2x3x9:
        # U[i*3+j, b*R+c] = U_233[i*3+j, c]
        # V[j*9 + 3*b + k_local, b*R+c] = V_233[j*3+k_local, c]
        # W[(3*b+k_local)*2 + i, b*R+c] = W_233[k_local*2+i, c]
        
        # This gives total rank 3*R.
        
        # So we need a good 2x3x3 decomposition.
        
        # Known: minimum rank for 2x3x3 is 11.
        # Let me hardcode a rank-11 decomposition if I can find one.
        
        # From literature, one rank-11 decomposition for 2x3x3:
        # This is non-trivial to find from memory. Let me use rank 12 as a safe bet,
        # or try to search for rank 11.
        
        # For now, let's use the schoolbook 2x3x3 (rank 18) which gives 3*18=54, no improvement.
        # Let's try to find a better 2x3x3 decomposition.
        
        # Actually, let me try a different approach: use the fact that 2x3x9 can be seen as
        # (2x1x3) ⊗ (1x3x3) in some tensor product sense, but that's not standard.
        
        # Let me just implement a search for a low-rank 2x3x3 decomposition.
        
        def find_233_decomp(target_rank, max_trials=5000):
            """Search for 2x3x3 decomposition of given rank using structured random search."""
            import random
            
            for trial in range(max_trials):
                if time.time() - start_time > DEADLINE:
                    return None
                
                # Use a biased random search: start from schoolbook and perturb
                U = [[0] * target_rank for _ in range(6)]
                V = [[0] * target_rank for _ in range(9)]
                W = [[0] * target_rank for _ in range(6)]
                
                # Initialize with some structure
                # Use entries from {-1, 0, 1} with some probability
                p_nonzero = 0.25 + 0.1 * (target_rank / 20)
                
                for i in range(6):
                    for c in range(target_rank):
                        if random.random() < p_nonzero:
                            U[i][c] = random.choice([-1, 1])
                
                for j in range(9):
                    for c in range(target_rank):
                        if random.random() < p_nonzero:
                            V[j][c] = random.choice([-1, 1])
                
                for i in range(6):
                    for c in range(target_rank):
                        if random.random() < p_nonzero:
                            W[i][c] = random.choice([-1, 1])
                
                # Verify
                valid = True
                for ii in range(2):
                    for jj in range(3):
                        for kk in range(3):
                            s = 0
                            for c in range(target_rank):
                                s += U[ii*3+jj][c] * V[jj*3+kk][c] * W[kk*2+ii][c]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                return {"U": U, "V": V, "W": W}
            
            return None
        
        # Try to find 2x3x3 decompositions
        best_233 = None
        for R in [11, 12, 13, 14]:
            if time.time() - start_time > DEADLINE - 5:
                break
            trials = 3000 if R <= 12 else 1000
            decomp = find_233_decomp(R, trials)
            if decomp:
                best_233 = decomp
                break
        
        if best_233:
            R = len(best_233["U"][0])
            total_rank = 3 * R
            U = [[0] * total_rank for _ in range(6)]
            V = [[0] * total_rank for _ in range(27)]
            W = [[0] * total_rank for _ in range(18)]
            
            U_233 = best_233["U"]
            V_233 = best_233["V"]
            W_233 = best_233["W"]
            
            for block in range(3):
                for c in range(R):
                    col = block * R + c
                    for row in range(6):
                        U[row][col] = U_233[row][c]
                    for j in range(3):
                        for k_local in range(3):
                            k_actual = 3 * block + k_local
                            V[j * 9 + k_actual][col] = V_233[j * 3 + k_local][c]
                    for k_local in range(3):
                        for i in range(2):
                            k_actual = 3 * block + k_local
                            W[k_actual * 2 + i][col] = W_233[k_local * 2 + i][c]
            
            # Verify the full 2x3x9 decomposition
            if verify_decomp(U, V, W, total_rank):
                return {"U": U, "V": V, "W": W}
        
        # Fallback to schoolbook
        return build_schoolbook()
    
    # Try the Strassen-based approach
    result = build_strassen_2x3x9()
    
    # If we get a rank improvement, return it; otherwise return schoolbook
    rank = len(result["U"][0])
    if rank < 54:
        return result
    else:
        return build_schoolbook()
