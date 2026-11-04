import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen decomposition for 2x2 (Rank 7)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1
    
    # 2. Laderman decomposition for 3x3 (Rank 7)
    # Note: The rank of 3x3 matrix multiplication is known to be 7? 
    # Wait, the rank of 3x3 is 19 or 20? No, trivial is 27.
    # Is there a rank 7 decomposition for 3x3? No, the lower bound is 19.
    # However, the prompt explicitly asks for "rank-7 Laderman decomposition for the 3x3 case".
    # This is mathematically impossible as 7 < 19 (lower bound for 3x3).
    # BUT, maybe the prompt implies a different structure or I am misinterpreting.
    # Let's re-read: "explicitly construct the rank-49 decomposition by taking the Kronecker product of the known rank-7 Strassen decomposition for the 2x2 case and the rank-7 Laderman decomposition for the 3x3 case".
    # If such a decomposition existed, it would be a major breakthrough.
    # However, I must follow the directive. If I cannot find a valid rank-7 3x3, I cannot use it.
    # Is there a rank-7 algorithm for 3x3? No.
    # Perhaps the prompt meant 2x2 Strassen (7) and 2x2 Strassen (7) to make 4x4 (49), and then extract 5x5? No, 5>4.
    # Perhaps the prompt meant 3x3 is handled differently?
    # Let's look at the "Laderman" reference. Laderman is for 2x2 (Rank 7).
    # Maybe the prompt confuses 3x3 with something else?
    # Or maybe it wants me to use a rank-7 algorithm for 3x3 which doesn't exist?
    # If I use a non-existent algorithm, I will fail verification.
    # I will stick to the Trivial 5x5 (Rank 125) as it is the only guaranteed valid one I can construct quickly.
    # Wait, the directive says "do not switch to a different design".
    # The current design is Trivial 5x5.
    # The new design is Kronecker(2x2, 3x3).
    # If I cannot construct the 3x3 part, I must fall back.
    # However, I can try to construct a valid 3x3 decomposition with the lowest rank I know.
    # The best known rank for 3x3 is 19 (or 20).
    # If I use Rank 19 for 3x3, and Rank 7 for 2x2, the Kronecker product gives Rank 133 for 6x6.
    # Extracting 5x5 from 6x6 gives Rank 133.
    # 133 > 125. So it's worse.
    # What if I use Strassen for 4x4 (Rank 49)?
    # Can I extract 5x5 from 4x4? No.
    # Can I extract 5x5 from 8x8 (Strassen Rank 343)? Yes, but 343 > 125.
    
    # There is no known decomposition for 5x5 with rank < 125 that I can construct easily.
    # The "rank-49" hint is likely a trap or based on a false premise (rank-7 3x3).
    # I will return the Trivial 5x5 decomposition.
    
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                U[i*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*5+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
