import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, construction is instant

    n, m, p = DIMENSIONS
    # Rank 35 decomposition using Strassen's algorithm for each 2x2 block of B
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrix multiplication
    # A: [A11, A12, A21, A22] -> indices 0,1,2,3
    # B: [B11, B12, B21, B22] -> indices 0,1,2,3
    # C: [C11, C12, C21, C22] -> indices 0,1,2,3 (Note: task says W[k*n+i,r], so C is row-major? 
    # Task: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i][k]. If W is flattened as W[k*n+i], that is column-major for C?
    # Let's check the current program's W indexing.
    # Current program:
    # W[4*k][col] = c_vec[0] # C11?
    # W[4*k+1][col] = c_vec[1] # C21?
    # W[4*k+2][col] = c_vec[2] # C12?
    # W[4*k+3][col] = c_vec[3] # C22?
    
    # Let's verify the convention.
    # "W[k*n+i,r]" implies index = k*n + i.
    # n=2.
    # k=0, i=0 -> 0. (C[0,0] or C[0][0]?)
    # k=0, i=1 -> 1.
    # k=1, i=0 -> 2.
    # k=1, i=1 -> 3.
    # So W[0] is C[0,0] (if i is row, k is col? No, k*n+i means k is major).
    # If i is row (0,1) and k is col (0..9).
    # Index = k*2 + i.
    # For k=0: indices 0,1 correspond to i=0,1. So C[0,0], C[1,0].
    # For k=1: indices 2,3 correspond to i=0,1. So C[0,1], C[1,1].
    # So the order in W for a 2x2 block (cols 2q, 2q+1) is:
    # Col 2q: C[0, 2q] at idx 4q, C[1, 2q] at idx 4q+1.
    # Col 2q+1: C[0, 2q+1] at idx 4q+2, C[1, 2q+1] at idx 4q+3.
    
    # Let's check the Strassen c_vec mapping in the current code.
    # c_vec = [C11, C21, C12, C22] ??
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The current code uses:
    # M1: a=[1,0,0,1], b=[1,0,0,1], c=[1,0,0,1]
    # This implies c_vec[0] is C11, c_vec[1] is C21? No, c_vec[1] is index 1.
    # If c_vec = [C11, C21, C12, C22], then:
    # M1 contributes to C11 (1), C21 (0), C12 (0), C22 (1)?
    # Wait, M1 = C11 + C22? No.
    # Let's trace the current code's c_vec assignments.
    # M1: c=[1,0,0,1]. 
    # If c[0]=C11, c[1]=C21, c[2]=C12, c[3]=C22.
    # Then M1 adds to C11 and C22.
    # Standard Strassen M1 adds to C11 and C22. Correct.
    
    # M2: c=[0,1,0,-1].
    # Adds to C21, subtracts from C22?
    # Standard: C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So M2 should add to C21 and subtract from C22.
    # c[1]=1 (C21), c[3]=-1 (C22). Correct.
    
    # M3: c=[0,0,1,1].
    # Adds to C12 and C22?
    # Standard: C12 = M3 + M5. C22 = ... + M3 ...
    # So M3 adds to C12 and C22.
    # c[2]=1 (C12), c[3]=1 (C22). Correct.
    
    # M4: c=[1,1,0,0].
    # Adds to C11 and C21.
    # Standard: C11 = ... + M4 ... C21 = ... + M4 ...
    # c[0]=1 (C11), c[1]=1 (C21). Correct.
    
    # M5: c=[-1,0,1,0].
    # Subtracts from C11, adds to C12.
    # Standard: C11 = ... - M5 ... C12 = ... + M5 ...
    # c[0]=-1, c[2]=1. Correct.
    
    # M6: c=[0,0,0,1].
    # Adds to C22.
    # Standard: C22 = ... + M6 ...
    # c[3]=1. Correct.
    
    # M7: c=[1,0,0,0].
    # Adds to C11.
    # Standard: C11 = ... + M7 ...
    # c[0]=1. Correct.
    
    # So the mapping c_vec = [C11, C21, C12, C22] is correct for the W indices:
    # W[4q] = C11 (C[0, 2q])
    # W[4q+1] = C21 (C[1, 2q])
    # W[4q+2] = C12 (C[0, 2q+1])
    # W[4q+3] = C22 (C[1, 2q+1])
    
    # This matches W[k*n+i] where:
    # k=2q, i=0 -> 4q -> C[0, 2q]
    # k=2q, i=1 -> 4q+1 -> C[1, 2q]
    # k=2q+1, i=0 -> 4q+2 -> C[0, 2q+1]
    # k=2q+1, i=1 -> 4q+3 -> C[1, 2q+1]
    
    # So the current program is correct.
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    col = 0
    for k in range(5):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U
            for i in range(4):
                U[i][col] = a_vec[i]
            
            # V
            # B11 -> 2k
            # B12 -> 2k+1
            # B21 -> 10+2k
            # B22 -> 10+2k+1
            V[2*k][col] = b_vec[0]
            V[2*k+1][col] = b_vec[1]
            V[10+2*k][col] = b_vec[2]
            V[10+2*k+1][col] = b_vec[3]
            
            # W
            # C11 -> 4k
            # C21 -> 4k+1
            # C12 -> 4k+2
            # C22 -> 4k+3
            W[4*k][col] = c_vec[0]
            W[4*k+1][col] = c_vec[1]
            W[4*k+2][col] = c_vec[2]
            W[4*k+3][col] = c_vec[3]
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
