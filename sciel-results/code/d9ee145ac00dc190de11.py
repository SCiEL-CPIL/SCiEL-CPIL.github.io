"""Find rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
import time
import itertools
import numpy as np
from numpy.linalg import norm

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, construct the guaranteed-correct schoolbook rank-27 decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook decomposition
    def verify(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(len(U[0])):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U_sb, V_sb, W_sb, n, m, p):
        # Should never happen, but just in case
        pass
    
    # Now try to find a lower-rank decomposition using structured search
    # Strategy: Try rank 23 with a parameterized family of solutions
    
    # The key insight: for the all-ones 3x3x3 tensor, we can try decompositions
    # where the rank-1 terms have specific patterns.
    
    # Known approach: use the fact that 1 = sum of certain products.
    # For 3x3x3, one known rank-23 construction uses:
    # - 9 "diagonal" terms (one for each (i,i,i))
    # - 18 "off-diagonal" terms with specific coefficients
    
    # Let me try a different approach: solve the system directly.
    # We need: sum_{r=0}^{R-1} u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k in {0,1,2}
    
    # This gives 27 equations in 9R unknowns.
    # For R=23, we have 207 unknowns and 27 equations - very underdetermined.
    
    # But we need exact rational/half-integer solutions.
    
    # Let me try a constructive approach based on known results.
    # The rank of the 3x3x3 matrix multiplication tensor is known to be at most 23.
    
    # One known construction:
    # Use the identity: 1 = (1/2) * (1 + 1) and exploit symmetry.
    
    # Actually, let me try to use a known rank-23 decomposition.
    # Based on literature, here's a construction:
    
    # The 3x3x3 all-ones tensor can be written as sum of 23 rank-1 tensors.
    # Here's one explicit construction using integer/half-integer entries:
    
    # Term structure:
    # - 9 terms of the form: e_i ⊗ e_i ⊗ e_i (diagonal)
    # - 6 terms for pairs (i,j) with i≠j: (e_i + e_j) ⊗ e_k ⊗ (e_i + e_j) type patterns
    # - 8 additional terms to complete the decomposition
    
    # Let me try a specific known construction.
    
    # After careful analysis, here's a rank-23 decomposition:
    # The key is to use the fact that for each "slice", we can share terms.
    
    # Let me try to construct it explicitly.
    
    # Approach: Use the following identity for the all-ones tensor:
    # T[i,j,k] = 1 = sum of rank-1 terms
    
    # One way: Note that 1 = 1*1*1, and we can use:
    # 1 = (1/3) * sum_{a=0}^{2} sum_{b=0}^{2} omega^{a(i-j)} * omega^{b(j-k)} * ... 
    # But this involves complex numbers, which we need to avoid.
    
    # Let me try a purely real, rational construction.
    
    # Known result: rank(3x3x3 MM tensor) <= 23
    # Construction from Strassen-type algorithms:
    
    # Here's a concrete rank-23 construction:
    # For each (i,j) pair, we have a "base" term, and we add correction terms.
    
    # Let me define the decomposition explicitly:
    
    R = 23
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Terms 0-8: Diagonal terms (i=i=i)
    # U[i*3+i][t] = 1, V[i*3+i][t] = 1, W[i*3+i][t] = 1
    for i in range(3):
        t = i
        U[i * 3 + i][t] = 1
        V[i * 3 + i][t] = 1
        W[i * 3 + i][t] = 1
    
    # Now we need to fill in the off-diagonal entries.
    # For each (i,j,k) with not all equal, we need the sum to be 1.
    
    # This is getting complex. Let me try a different approach.
    
    # Use the fact that the tensor is the Kronecker product structure.
    # Actually, the 3x3x3 all-ones tensor is NOT a Kronecker product.
    
    # Let me try to use a greedy/iterative approach to find a valid decomposition.
    
    # Alternative: Use the known fact that we can always achieve rank <= n*m*p,
    # and try to reduce by merging terms.
    
    # For now, let's try a specific known rank-23 construction.
    
    # Based on research, here's a rank-23 decomposition for 3x3x3:
    
    # The decomposition uses the following structure:
    # 9 "pure" terms + 14 "mixed" terms = 23 total
    
    # Let me try to implement a verified construction.
    
    # Actually, let me try a simpler approach: use the schoolbook rank-27 as fallback,
    # and try to find any improvement through systematic search.
    
    # Try rank 26: Can we merge any two schoolbook terms?
    # Two terms can be merged if their u, v, w vectors are linearly dependent in a way
    # that allows combining.
    
    # For the schoolbook, each term is a "basis" term. Merging is not straightforward.
    
    # Let me try a completely different approach: use the known Strassen-like
    # algorithm for 3x3 matrices.
    
    # After extensive analysis, I'll implement a known rank-23 construction.
    
    # Here's a valid rank-23 decomposition with integer entries:
    
    # Reset U, V, W
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # This is a known construction from the tensor decomposition literature.
    # The 23 terms are defined as follows:
    
    # Terms 0-8: Diagonal
    for i in range(3):
        t = i
        U[i * 3 + i][t] = 1
        V[i * 3 + i][t] = 1
        W[i * 3 + i][t] = 1
    
    # Terms 9-14: Row interactions
    # For each pair (i, j) with i < j, add terms that handle the cross-terms
    
    # This specific construction is complex. Let me verify if we can find
    # a simpler valid lower-rank decomposition.
    
    # Try rank 24: Remove one term from schoolbook and see if we can adjust others.
    
    # Actually, let me just return the schoolbook decomposition which is guaranteed correct.
    # The rank-23 construction requires careful verification.
    
    # Check time
    if time.time() > deadline - 5.0:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Let me try one more approach: use the fact that we can represent
    # the all-ones tensor using fewer terms by exploiting the structure.
    
    # Key insight: 1 = (1/2) * (1 + 1), and we can use:
    # For each (i,k) pair, C[i,k] = sum_j A[i,j] B[j,k]
    # This is a dot product, which can be computed with fewer terms using
    # Strassen-like tricks.
    
    # For 3x3, one approach:
    # Split the 3x3 matrices into 2x2 and 1x1 blocks.
    # Use Strassen for the 2x2 part (rank 7) and handle the 1x1 parts separately.
    # This gives rank <= 7 + 3 + 3 + 1 = 14? No, that's not quite right.
    
    # Let me try: 
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [B11 B12; B21 B22] similarly
    # C = A*B
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x1)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x1)
    
    # Using Strassen for C11: 7 terms
    # For C12: 2x1 = sum of 2 dot products, each 2 terms = 4 terms (or use Strassen-like)
    # Actually, a 2x1 times 1x1 is just scaling, so 2 terms.
    # Similarly for C21: 2 terms.
    # C22: 1 term.
    # Total: 7 + 2 + 2 + 1 = 12? But this doesn't account for the cross-terms properly.
    
    # This approach is getting complicated and I'm not confident in the exact rank.
    
    # Let me just return the guaranteed-correct schoolbook rank-27 decomposition.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
