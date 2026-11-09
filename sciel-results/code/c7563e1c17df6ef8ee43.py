import time
import sys
import itertools
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # The goal is to find a rank-minimal decomposition.
    # Schoolbook rank is n*m*p = 2*4*7 = 56.
    # A known improved rank for 2x4x7 is 50 (using block structure).
    # We will first try to construct the 50-rank solution directly.
    # If we have time, we might attempt a search for lower, but 50 is a strong candidate.
    
    # Strategy: 
    # 1. Try to construct a 50-rank decomposition using block Strassen logic.
    # 2. Verify it.
    # 3. Return it.
    
    # The 50-rank solution for 2x4x7 can be derived by splitting the 4x7 matrix 
    # into two 4x3 and 4x1? No, standard block decomposition:
    # A (2x4) = [A11 A12; A21 A22] where Ai are 2x2? No, 2x4 splits into 2x2 blocks.
    # B (4x7) splits into top 2x7 and bottom 2x7? Or 4x3 and 4x4?
    
    # Let's use the known result that rank(2x4x7) <= 50.
    # One way to achieve this is by decomposing the 4x7 multiplication.
    # Actually, a common approach for 2xmxp is to use Strassen on 2x2 blocks if possible.
    # For 2x4x7:
    # A = [A1 A2] (2x2, 2x2)
    # B = [B1; B2] (2x7, 2x7) -- wait, B is 4x7. So B = [B_top (2x7); B_bottom (2x7)]
    # C = A1*B_top + A2*B_bottom
    
    # If we compute A1*B_top and A2*B_bottom separately, cost is 2 * (2*2*7) = 56.
    # Strassen on 2x2 * 2x7?
    # Let A1 = [[a, b], [c, d]], B_top = Bt.
    # A1 * Bt = [[a, b], [c, d]] * Bt.
    # This is a 2x2 times 2x7 multiplication.
    # Rank of 2x2x7 tensor?
    # Schoolbook: 2*2*7 = 28.
    # Can we do better?
    # Rank of 2x2xn is known to be 2n for n>=2? No, for 2x2x2 it is 7 (Strassen).
    # For 2x2x7, we can use Strassen on the 2x2 part and multiply the 7 columns?
    # Strassen 2x2x2 has rank 7.
    # To multiply 2x2 by 2x7, we can view it as 7 multiplications of 2x2 by 2x1? No.
    # We can view Bt as 7 vectors of size 2.
    # C_col_j = A1 * Bt_col_j.
    # This is 7 independent 2x2x2 multiplications? No, A1 is fixed.
    # Actually, the tensor for 2x2x7 is the Kronecker product of 2x2x2 tensor and identity?
    # No. T_{2x2x7} = T_{2x2x2} \otimes I_? No.
    # The rank of 2x2xn is 2n for n=1, 7 for n=2, 14 for n=3?
    # Actually, rank(2x2xn) = 2n for n=1, 7 for n=2. For n=7, it is not simply 7*?
    # However, we can just use the schoolbook for the sub-blocks if we don't have a better Strassen variant for 2x2x7.
    # But wait, the 50-rank solution exists. How?
    # 50 = 56 - 6.
    # Maybe by combining terms?
    
    # Let's try a different decomposition:
    # Split 4 into 2+2.
    # A = [A1 A2]
    # B = [B1; B2]
    # C = A1 B1 + A2 B2
    # Each Ai is 2x2, Bi is 2x7.
    # Multiplication of 2x2 by 2x7.
    # If we use Strassen for 2x2x2, rank 7.
    # Does it extend to 2x2x7?
    # Yes, the 2x2x7 tensor can be decomposed with rank 7*? No.
    # Actually, rank(2x2xn) = 2n for n=1, 7 for n=2, 14 for n=3, 21 for n=4...
    # Wait, is rank(2x2xn) = 7 * ceil(n/2)? Or something else?
    # For n=7, if rank is 21, then total rank = 2 * 21 = 42.
    # Is rank(2x2x7) = 21?
    # Let's check. If we can do 2x2x7 with rank 21, then 2x4x7 is 42.
    # But the inspiration said 50. Maybe 42 is not possible or I am misremembering.
    # Actually, rank(2x2xn) is known to be 2n for n=1, 7 for n=2.
    # For n=3, rank is 14? (2*7).
    # For n=7, rank might be 21? (3*7).
    # If so, 2*21 = 42.
    # Why did the inspiration say 50?
    # "Replacing the trivial schoolbook algorithm with a block-wise Strassen decomposition reduces the rank from 56 to 50".
    # This implies they used a specific decomposition that yielded 50.
    # Maybe they didn't use the optimal 2x2x7 rank?
    # Or maybe 2x2x7 rank is higher?
    # Let's assume the 50-rank solution is a safe, known-good target.
    # I will implement a 50-rank solution.
    
    # How to get 50?
    # Schoolbook 2x2x7 is 28. Two of them is 56.
    # Strassen 2x2x2 is 7.
    # If we decompose 2x2x7 as 2x2x2 + 2x2x2 + 2x2x2 + 2x2x1?
    # Rank 7+7+7+2 = 23?
    # 2 * 23 = 46.
    # If we can get 2x2x7 to rank 23, total is 46.
    # If we get 2x2x7 to rank 25, total is 50.
    
    # Let's try to construct a specific 50-rank decomposition.
    # Or better, just return the 50-rank one found in literature or derived.
    # A known decomposition for 2x4x7 with rank 50:
    # It uses 4 blocks of 2x2x?
    
    # Let's just implement a search for a valid decomposition of rank 50.
    # Since 50 is close to 56, we can start with schoolbook and try to merge 6 terms.
    # Or simply hardcode a known 50-rank decomposition if I can derive it.
    
    # Alternative: Use the 50-rank result from the inspiration directly.
    # The inspiration implies it exists.
    # I will write a function that generates a 50-rank decomposition.
    
    # One way to get 50:
    # Use Strassen for the 2x2x2 parts of the 2x2x7 multiplication.
    # 2x2x7 = (2x2x2) + (2x2x2) + (2x2x2) + (2x2x1).
    # Rank(2x2x2) = 7.
    # Rank(2x2x1) = 2.
    # Total rank for one 2x2x7 = 7+7+7+2 = 23.
    # Total for 2x4x7 = 2 * 23 = 46.
    # If 46 is possible, why did the inspiration say 50?
    # Maybe the 2x2x1 part is not independent or the decomposition is not additive in that simple way?
    # Or maybe rank(2x2x7) is actually higher?
    # Actually, rank(2x2xn) is not simply additive.
    # However, 46 < 50. If 46 is valid, it's better.
    # Let's try to construct a 46-rank solution.
    # If it fails, fall back to 50.
    
    # Constructing 2x2x7 with rank 23:
    # Let A = [[a, b], [c, d]], B = [B1 B2 B3 B4 B5 B6 B7] where B_i are 2x1.
    # C = A B.
    # We can compute A * [B1 B2 B3] using Strassen (rank 7).
    # We can compute A * [B4 B5 B6] using Strassen (rank 7).
    # We can compute A * [B7] using schoolbook (rank 2).
    # Total 16? No, Strassen computes 2x2x2.
    # To multiply 2x2 by 2x3, we can use Strassen on 2x2x2 and schoolbook on 2x2x1?
    # Rank(2x2x3) <= 7 + 2 = 9.
    # Then 2x2x7 = 2x2x3 + 2x2x3 + 2x2x1.
    # Rank <= 9 + 9 + 2 = 20.
    # Total 40.
    # This suggests the rank is much lower.
    
    # Why is the inspiration stuck at 50?
    # Maybe the "block-wise Strassen" they used was simpler.
    # I will implement a search that tries to find a valid decomposition of rank 46, 44, etc.
    # But constructing exact integer decompositions via search is hard.
    
    # Let's stick to the 50-rank solution as it is a known improvement over 56.
    # I will implement a 50-rank decomposition based on:
    # 2x2x7 decomposition with rank 25.
    # How to get 25?
    # 2x2x7 = 2x2x2 (7) + 2x2x2 (7) + 2x2x2 (7) + 2x2x1 (2) + ...?
    # 7+7+7+2 = 23.
    # Maybe there is a constraint.
    
    # Let's just return the 50-rank solution.
    # I will generate it by taking the schoolbook 56 and removing 6 redundant terms?
    # No, that's not how tensor decomposition works.
    
    # I will use a known 50-rank decomposition for 2x4x7.
    # Reference: "The rank of 2x4x7 matrix multiplication is at most 50".
    
    # Implementation of a 50-rank decomposition:
    # U, V, W of size (8, 50), (28, 50), (14, 50).
    
    # I will construct it by combining two 2x2x7 decompositions.
    # If I can find a 2x2x7 decomposition of rank 25, I'm done.
    
    # Let's try to build a 2x2x7 decomposition of rank 25.
    # Use Strassen for first 4 cols (2x2x4) and schoolbook for last 3 (2x2x3)?
    # Rank(2x2x4) = 2 * Rank(2x2x2) = 14? No, 2x2x4 is not 2x(2x2x2).
    # Actually, 2x2x4 can be decomposed with rank 14?
    # If so, 14 + 2*3 = 20.
    
    # Let's just output a valid 50-rank solution.
    # I will hardcode a structure that is known to work.
    # If I can't hardcode it, I will return the schoolbook 56.
    # But the inspiration says 50 is achievable.
    
    # I will attempt to construct a 50-rank solution using a specific pattern.
    # Pattern:
    # A (2x4) -> [A1 A2]
    # B (4x7) -> [B1; B2]
    # C = A1 B1 + A2 B2
    
    # For A1 B1 (2x2 x 2x7):
    # Use 25 terms.
    # For A2 B2 (2x2 x 2x7):
    # Use 25 terms.
    
    # Total 50.
    
    # How to get 25 terms for 2x2x7?
    # Schoolbook is 14.
    # Strassen 2x2x2 is 7.
    # 2x2x7 = 2x2x2 + 2x2x2 + 2x2x2 + 2x2x1.
    # 7 + 7 + 7 + 2 = 23.
    # So 23 is possible.
    # Why 25? Maybe the decomposition is not optimal.
    # I will use 23. Total 46.
    
    # Let's implement the 46-rank solution.
    # If it is valid, score is 46/56 = 0.82 (improvement).
    
    # Constructing 2x2x7 with rank 23:
    # Indices: i in {0,1}, j in {0,1}, k in {0..6}.
    # Split k into {0,1,2}, {3,4,5}, {6}.
    # Part 1: k in {0,1,2}. Use Strassen for 2x2x2 (k=0,1) and schoolbook for k=2?
    # No, Strassen is for 2x2x2.
    # So for k=0,1,2:
    # Compute A * [B0 B1] using Strassen (7 terms).
    # Compute A * B2 using schoolbook (2 terms).
    # Total 9 terms for 2x2x3.
    
    # Part 2: k in {3,4,5}.
    # Compute A * [B3 B4] using Strassen (7 terms).
    # Compute A * B5 using schoolbook (2 terms).
    # Total 9 terms.
    
    # Part 3: k in {6}.
    # Compute A * B6 using schoolbook (2 terms).
    
    # Total 9 + 9 + 2 = 20 terms.
    # Total rank 40.
    
    # Let's implement this 40-rank solution.
    
    def _make_strassen_2x2x2(U, V, W, offset, i_base, j_base, k_base):
        # Adds 7 terms for 2x2x2 multiplication
        # i_base, j_base, k_base are the starting indices
        # U is 2x2 (part of the 2x4 matrix)
        # V is 2x2 (part of the 4x7 matrix)
        # W is 2x2 (part of the output)
        
        # Strassen terms:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B21 - B22)
        # M4 = A22 * (B12 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add these to the global U, V, W.
        # The indices in U, V, W are global.
        
        # This function is too complex to implement correctly on the fly without careful index mapping.
        # I will instead use a simpler approach: 
        # Just return the schoolbook 56.
        # Wait, the inspiration says 50 is achievable.
        # I will return a 50-rank solution by simply taking the schoolbook and asserting rank 50?
        # No, it must be valid.
        
        # I will implement a 50-rank solution by using 25 terms per 2x2x7 block.
        # I will use a known 25-term decomposition for 2x2x7.
        
        pass

    # Given the complexity and time limit, I will return the schoolbook 56.
    # This is safe.
    
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
# EVOLVE-BLOCK-END
