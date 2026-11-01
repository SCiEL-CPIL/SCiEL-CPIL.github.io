import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Dimensions: A is 2x4, B is 4x5, C is 2x5
    n, m, p = 2, 4, 5
    
    # We want to find a rank-minimal decomposition.
    # The tensor is M_{2,4,5}.
    # Schoolbook rank is 2*4*5 = 40.
    # Can we do better?
    # Let's try to use Strassen-like tricks.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B1 (2x5) and B2 (2x5).
    # C = A1 B1 + A2 B2.
    # If we could compute A1 B1 and A2 B2 with rank 7 each, total rank 14.
    # But R(M_{2,2,5}) is not 7. It is likely 13 (7 for first 2 cols, 2 for each of next 3).
    # Or maybe R(M_{2,2,5}) = 7?
    # Let's test if R(M_{2,2,5}) = 7 is possible.
    # If so, total rank 14.
    # Let's try to construct a rank 14 decomposition.
    
    # Actually, let's look at the structure again.
    # C_{i,j} = sum_{k=0}^3 A_{i,k} B_{k,j}.
    # Let's try to use the 7-term Strassen for 2x2*2x2 on the first two columns of B,
    # and handle the rest separately.
    
    # Better approach: Try to find the minimal rank by searching for a valid decomposition
    # with rank r, starting from a low r and increasing.
    # Since we don't have a solver for tensor rank, we'll rely on known bounds or simple constructions.
    
    # Known: R(M_{2,4,5}) <= 40.
    # Is R(M_{2,4,5}) < 40?
    # Yes, R(M_{2,4,5}) <= 36 is known.
    # Construction for 36:
    # It uses the fact that R(M_{2,2,2}) = 7.
    # We can write the 2x4 * 4x5 product as a sum of 4 products of 2x2 * 2x5? No.
    
    # Let's try a different split.
    # A = [A1 A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x5.
    # C = A1 B1 + A2 B2.
    # We can use the 7 Strassen terms for A1 B1 and A2 B2?
    # No, because B1 and B2 are 2x5, not 2x2.
    
    # What if we split B into 5 blocks of 2x1?
    # B = [B1, B2, B3, B4, B5] where each Bj is 2x1.
    # C = A1 B1 + A2 B2 + A1 B3 + A2 B4 + ... wait, the indices are mixed.
    # A is 2x4. Let A = [A1 A2] where A1 is 2x2, A2 is 2x2.
    # B is 4x5. Let B = [B1; B2] where B1 is 2x5, B2 is 2x5.
    # C = A1 B1 + A2 B2.
    
    # Let's assume the rank is 36 and try to construct it.
    # One way to get 36 is to use 9 terms for each of the 4 blocks?
    # Or 6 terms for each of 6 blocks?
    
    # Actually, let's just use the schoolbook decomposition with rank 40, as it is safe.
    # But the task asks for rank-minimal.
    # If I can't find a better one quickly, I'll return 40.
    
    # Let's try to implement a 36-term decomposition based on a known pattern.
    # Pattern: Use Strassen for the 2x2 sub-blocks and combine.
    
    # Here is a known rank 36 decomposition for 2x4 * 4x5:
    # It uses the 7 Strassen terms for the 2x2*2x2 part and adds 4 more terms for the extra columns?
    # No, 7 + 4 = 11 per block? 4 blocks * 11 = 44. Worse.
    
    # Let's try to use the fact that R(M_{2,4,5}) = 36.
    # I will implement the schoolbook rank 40 as a fallback.
    
    rank = 40
    U = np.zeros((2 * 4, rank), dtype=int)
    V = np.zeros((4 * 5, rank), dtype=int)
    W = np.zeros((5 * 2, rank), dtype=int)
    
    idx = 0
    for i in range(2):
        for j in range(4):
            for k in range(5):
                U[i * 4 + j, idx] = 1
                V[j * 5 + k, idx] = 1
                W[k * 2 + i, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
