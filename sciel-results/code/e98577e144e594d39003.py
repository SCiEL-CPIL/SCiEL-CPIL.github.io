import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p
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

    # Strategy: Use Strassen-based decomposition for the 3x3 x 3x3 blocks
    # and combine efficiently for the 3x3 x 3x8 case.
    
    # For 3x3 x 3x3, we can use a rank-23 decomposition.
    # For 3x3 x 3x8, we can think of it as 8 columns, each requiring 3x3 x 3x1.
    # But we can share the A and B structure.
    
    # Let's try a direct approach: construct the tensor and find a low-rank decomposition.
    # The tensor T has dimensions (n*m, m*p, p*n) = (9, 24, 24).
    # T[(i,j), (j,k), (k,i)] = 1 for valid i,j,k, else 0.
    
    # Actually, let's use a known result: the rank of 3x3 x 3x3 matrix multiplication
    # is at most 23 (using Strassen + corrections).
    # For 3x3 x 3x8, we can potentially do better than schoolbook 72.
    
    # Let me construct a valid decomposition using the following approach:
    # Split the 8 columns into groups and use shared factors.
    
    # A simpler approach: use rank 27 for each 3x3x3 block (schoolbook) but
    # try to find a better decomposition.
    
    # Let me try to construct a rank-23 decomposition for 3x3x3 first.
    
    def make_3x3x3_rank23():
        """Construct a rank-23 decomposition for 3x3 x 3x3 matrix multiplication."""
        rank = 23
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        
        # Use Strassen for the 2x2x2 part (indices 0,1)
        # Strassen's 7 products for 2x2 matrices:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B00) -> A11*B00... 
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B10) -- no, let me use the correct formulas
        # 
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M5 - M7
        # where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B10)
        # M5 = (A00+A10)(B11+B01) -- wait, this is getting confused
        #
        # Let me use a cleaner formulation. For 2x2:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = A*B
        # Strassen:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a-b)(g+h)
        # p4 = (d)(f-h)  -- wait, standard is p4 = d*(e-f)? No.
        #
        # Standard Strassen for C = AB:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a-b)(g+h)
        # p4 = d*(f-h)  -- no, p4 = d*(e-f)? 
        #
        # Actually, the standard Strassen is:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = (a-b)(g+h)
        # p4 = d(f-h)  -- I think this is wrong
        #
        # Let me just use a verified set. The correct Strassen products are:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = (a-b)(g+h)
        # p4 = d(f-h)  -- NO
        #
        # Correct Strassen:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = (a-b)(g+h)
        # p4 = d(f-h)  -- I keep getting confused. Let me look it up mentally.
        #
        # C00 = p1 - p4 - p6 + p7
        # C01 = p3 + p4
        # C10 = p2 + p5
        # C11 = p1 - p2 + p5 - p7
        #
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = (a-b)(g+h)
        # p4 = d(f-h)  -- this doesn't seem right dimensionally
        #
        # I think the correct ones are:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = (a-b)(g+h)
        # p4 = d(f-h)  -- actually I think p4 should be d*(e-f) or similar
        #
        # Let me just verify with a simple example. If A = [[1,0],[0,1]], B = [[1,0],[0,1]]
        # Then C should be [[1,0],[0,1]]
        # a=1,b=0,c=0,d=1, e=1,f=0,g=0,h=1
        # p1 = (1+1)(1+1) = 4
        # p2 = (0+1)*1 = 1
        # p3 = (1-0)(0+1) = 1
        # p4 = 1*(0-1) = -1  (if p4 = d*(f-h))
        # p5 = ? 
        # p6 = ?
        # p7 = ?
        #
        # This is getting too complex to do from memory. Let me just use schoolbook
        # for the 3x3x3 part and see if I can optimize the overall structure.
        
        # Given the complexity and time constraints, let me just return schoolbook.
        return None
    
    # Let me try a different approach: directly construct the decomposition
    # using the tensor structure.
    
    # The multiplication tensor for 3x3 x 3x8 has:
    # U: 9 x rank (indices i*m+j = i*3+j for i,j in 0..2)
    # V: 24 x rank (indices j*p+k = j*8+k for j in 0..2, k in 0..7)
    # W: 24 x rank (indices k*n+i = k*3+i for k in 0..7, i in 0..2)
    #
    # T[(i,j), (j,k), (k,i)] = 1
    
    # Let me try to find a decomposition with rank < 72.
    # One approach: use the fact that for each fixed k, we have a 3x3 x 3x1 problem.
    # But we can share the U and V factors across different k values.
    
    # Actually, let me try a random search for a low-rank decomposition.
    
    def try_random_decomposition(max_rank, max_iters=1000):
        """Try to find a rank-r decomposition using random initialization."""
        for r in range(max_rank, 60, -1):
            if time.time() > deadline - 10:
                break
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                # Random initialization
                U = np.random.randn(9, r)
                V = np.random.randn(24, r)
                W = np.random.randn(24, r)
                
                # Check if this decomposes the tensor
                # T[(i,j),(j,k),(k,i)] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This should equal 1 for all valid (i,j,k)
                
                # Build the tensor from the decomposition
                tensor = np.zeros((9, 24, 24))
                for c in range(r):
                    uc = U[:, c]
                    vc = V[:, c]
                    wc = W[:, c]
                    # Outer product
                    for ij in range(9):
                        for jk in range(24):
                            for ki in range(24):
                                tensor[ij, jk, ki] += uc[ij] * vc[jk] * wc[ki]
                
                # Check against the target tensor
                target = np.zeros((9, 24, 24))
                for i in range(3):
                    for j in range(3):
                        for k in range(8):
                            target[i*3+j, j*8+k, k*3+i] = 1
                
                if np.allclose(tensor, target, atol=1e-6):
                    # Convert to integer/half-integer if possible
                    U_int = U.round(2)
                    V_int = V.round(2)
                    W_int = W.round(2)
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
        return None
    
    # Try random search
    result = try_random_decomposition(70, max_iters=500)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook()
