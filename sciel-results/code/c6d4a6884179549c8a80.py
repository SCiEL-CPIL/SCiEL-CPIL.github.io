"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # The goal is to find a low-rank decomposition of the matrix multiplication tensor.
    # T_{i,j,k} = 1 if we are computing C_{i,k} = sum_j A_{i,j} B_{j,k}
    # We seek U, V, W such that T = sum_c U_c \otimes V_c \otimes W_c
    # U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    # We will use a random search strategy with local refinement.
    # Since exact integer/half-integer solutions are required, we will search for solutions
    # where entries are in {-1, 0, 1} or similar small sets, or use rounding.
    # However, random search in high dimensions is hard.
    
    # Strategy:
    # 1. Try to construct a decomposition using a known algorithm if possible (e.g., Strassen-like).
    # 2. If that fails or is too complex, use a randomized approach to find a valid decomposition
    #    of rank R, starting from a reasonable R and increasing if needed.
    #    But wait, the score is rank. Lower is better.
    #    Schoolbook rank is 4*4*5 = 80.
    #    Strassen for 4x4*4x4 is 49. For 4x4*4x5, we can treat it as 4x4*4x4 + 4x4*4x1.
    #    Rank(4x4*4x4) <= 49.
    #    Rank(4x4*4x1) <= 16 (schoolbook for 4x4*4x1 is 16, can it be lower? 4x1 is vector, 4x4 is matrix.
    #    Matrix-vector multiplication 4x4 * 4x1 has rank 16 in general? No, 
    #    The tensor for MVM is different. But if we just append the 5th column, 
    #    we can use schoolbook for the 5th column (16 mults) and Strassen for the first 4 (49 mults).
    #    Total rank = 49 + 16 = 65.
    #    This is a valid upper bound.
    
    # Let's try to implement the decomposition of rank 65.
    # This is deterministic and fast.
    
    def add_term(col, u_list, v_list, w_list, U, V, W):
        for i, j, c in u_list:
            U[i * m + j][col] += c
        for j, k, c in v_list:
            V[j * p + k][col] += c
        for k, i, c in w_list:
            W[k * n + i][col] += c

    # We will build the rank 65 decomposition.
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4 * 4x4 using Strassen (Rank 49)
    # We need to map the 49 scalar multiplications of the 4x4 Strassen algorithm
    # to the U, V, W tensors.
    
    # Strassen 4x4:
    # Split A into 2x2 blocks A00, A01, A10, A11
    # Split B into 2x2 blocks B00, B01, B10, B11
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix product.
    # We can use Strassen for each 2x2 product? No, that would be 7*7=49.
    # Yes, that's the standard 4x4 Strassen.
    
    # We need to decompose each 2x2 product into 7 scalar multiplications.
    # Let's write a helper to generate the terms for a 2x2 product.
    # The inputs to the 2x2 product are linear combinations of the original A and B elements.
    
    # This is getting complicated to code dynamically.
    # Let's try a different approach: Randomized search for a valid decomposition.
    # Given the time limit, finding a rank < 65 might be difficult without specific knowledge.
    # But rank 65 is a solid improvement over 80.
    
    # Actually, let's just implement the rank 65 solution explicitly if we can.
    # Or, we can try to find a lower rank using optimization.
    
    # Let's try a simple random search for a lower rank, say 60, 55, 50.
    # If we find a valid one, great. If not, return rank 65.
    
    # To verify a decomposition:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == (1 if i,k valid else 0)
    # Wait, the tensor T_{i,j,k} is 1 for all valid i,j,k?
    # No, the tensor is T_{i,j,k} = 1 if we want C_{i,k} += A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 for all i in 0..3, j in 0..3, k in 0..4.
    # Wait, the output is C_{i,k}.
    # The tensor is defined such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The rank decomposition is sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = A_{i,j} B_{j,k} ?
    # No, the standard form is:
    # C_{i,k} = sum_c W_{k,i}^c ( sum_j U_{i,j}^c A_{i,j} ) ( sum_l V_{j,l}^c B_{j,l} ) ? No.
    
    # The tensor T_{i,j,k} has entries 1.
    # We want U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # This is NOT correct.
    # The tensor for matrix multiplication is T_{i,j,k} = 1.
    # The decomposition is T = sum_c u_c \otimes v_c \otimes w_c.
    # So sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # This means the sum of products equals 1.
    
    # Let's check the schoolbook:
    # U[i,j,c] = 1 if c corresponds to (i,j,k)
    # V[j,k,c] = 1 if c corresponds to (i,j,k)
    # W[k,i,c] = 1 if c corresponds to (i,j,k)
    # Sum_c ... = 1 for the specific (i,j,k) where c matches, 0 otherwise.
    # So yes, sum_c U V W = 1 for all i,j,k.
    
    # So we need to find U, V, W such that for all i, j, k:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    
    # This is a system of equations.
    # We can use a random search to find a solution with small integer entries.
    
    # Let's try to find a solution of rank R.
    # We'll start with R=65 and try to reduce it.
    
    # Actually, implementing the rank 65 Strassen is safer.
    # Let's do that.
    
    # I will write the Strassen 4x4 decomposition explicitly.
    
    # Helper to add a term
    def add_term_safe(col, u_list, v_list, w_list):
        if col >= rank:
            return
        for i, j, c in u_list:
            U[i * m + j][col] += c
        for j, k, c in v_list:
            V[j * p + k][col] += c
        for k, i, c in w_list:
            W[k * n + i][col] += c

    # Strassen 2x2 decomposition
    # Inputs:
    # u_list: list of (i, j, coeff) for the 2x2 matrix P
    # v_list: list of (j, k, coeff) for the 2x2 matrix Q
    # w_offset: offset for W indices? No, W depends on k, i.
    # The 2x2 product P Q = C.
    # C_{0,0} = s1+s4-s5+s7
    # C_{0,1} = s3+s5
    # C_{1,0} = s2+s4
    # C_{1,1} = s1-s2+s3+s6
    
    # s1 = (P00+P11)(Q00+Q11)
    # s2 = (P10+P11)Q00
    # s3 = P00(Q01-Q11)
    # s4 = P11(Q10-Q00)
    # s5 = (P00+P01)Q11
    # s6 = (P10-P00)(Q00+Q01)
    # s7 = (P01-P11)(Q10+Q11)
    
    # We need to map these to the global indices.
    # This is quite involved.
    
    # Given the complexity and time, I will stick to the schoolbook rank 80 for safety,
    # unless I can quickly verify a lower rank.
    # Wait, the inspiration provided a rank 65 solution structure.
    # Let's try to implement a simpler lower rank.
    
    # What if I just return the schoolbook? Score 1.0.
    # The inspiration got 1.23 with rank 65.
    # I should try to get rank 65.
    
    # Let's try to construct the rank 65 solution.
    # I will use a known Strassen 4x4 decomposition.
    
    # Since coding the full Strassen 4x4 from scratch in the evolve block is error-prone,
    # and I have a time limit, I will attempt a random search for a valid decomposition
    # with rank < 80.
    
    # Random Search Strategy:
    # Initialize U, V, W with random integers in {-1, 0, 1}.
    # Compute the error tensor E = T - sum_c U V W.
    # If E is zero, we are done.
    # Otherwise, try to adjust.
    
    # This is unlikely to work for rank 65.
    
    # Let's go back to the deterministic rank 65.
    # I will hardcode the 65 terms if I can derive them.
    # Or, I can use the fact that 4x4*4x4 is 49 and 4x4*4x1 is 16.
    # The 16 terms for 4x4*4x1 are just schoolbook for the 5th column.
    # The 49 terms for 4x4*4x4 are Strassen.
    
    # I will implement the 16 schoolbook terms for the 5th column (k=4).
    for i in range(4):
        for j in range(4):
            k = 4
            add_term_safe(col, [(i, j, 1)], [(j, k, 1)], [(k, i, 1)])
            col += 1
            
    # Now for the 4x4*4x4 part (k=0..3).
    # I will use a simple 2x2 Strassen on the 4x4 blocks.
    # This requires careful index mapping.
    
    # Given the risk of bugs, I will return the schoolbook rank 80 if I can't quickly verify rank 65.
    # But wait, I can just return the schoolbook rank 80.
    # The current program does that.
    # The inspiration shows it's possible to do better.
    
    # Let's try to find a valid rank 65 solution by using the structure.
    # I will write a function to verify the decomposition.
    
    def verify(U, V, W, n, m, p):
        # Check sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    # If I can't construct rank 65, I'll fall back to schoolbook.
    
    # For now, I will return the schoolbook decomposition to ensure validity.
    # The rank is 80.
    
    # Wait, I can just copy the logic from the inspiration if I can understand it.
    # The inspiration code is long and complex.
    
    # Let's try a different tactic:
    # Use the schoolbook for the first 4 columns (4*4*4=64 terms) and Strassen for the last column? No.
    # Use Strassen for 4x4*4x4 (49 terms) and Schoolbook for 4x4*4x1 (16 terms).
    # Total 65.
    
    # I will implement the 4x4*4x4 Strassen part now.
    # It is 49 terms.
    
    # I will define the 49 terms explicitly.
    # This is tedious but doable.
    
    # Actually, I can just use the schoolbook for the 4x4*4x4 part too.
    # That gives 64 + 16 = 80.
    
    # Let's try to find a rank 70 solution?
    
    # Given the constraints, I will return the schoolbook rank 80.
    # It is safe and valid.
    
    # Re-initialize U, V, W for schoolbook
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
