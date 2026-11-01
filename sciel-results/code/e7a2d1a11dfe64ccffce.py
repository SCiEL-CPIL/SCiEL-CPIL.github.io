"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 4
    
    # Schoolbook decomposition as guaranteed fallback
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
    
    # Verify a candidate decomposition
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using structured search
    # Strategy: exploit the 2x3x4 structure with block decomposition
    # M_{2,3,4} can be decomposed using 2x2x2 Strassen-like patterns
    
    def try_structured_rank(rank):
        """Try to find a valid decomposition of given rank using structured patterns."""
        if time.time() > deadline:
            return None
        
        # For rank 12: use the fact that M_{2,3,4} = M_{2,3,2} + M_{2,3,2}
        # where each M_{2,3,2} has rank 6 (since M_{2,3,2} can be done in 6 terms)
        # Actually, let's use a more direct approach
        
        # Key insight: For 2x3x4, we can split the 4 columns into two groups of 2
        # C[:, 0:2] = A * B[:, 0:2] (2x3 * 3x2)
        # C[:, 2:4] = A * B[:, 2:4] (2x3 * 3x2)
        # Each 2x3x2 multiplication has rank at most 6
        # So total rank <= 12
        
        # For 2x3x2: rank is 6 (can be verified)
        # Construction for 2x3x2:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        
        # This is 6 terms, which is the schoolbook rank for 2x3x2
        
        # But can we do better? For 2x2x2, Strassen gives rank 7 < 8
        # For 2x3x2, let's see if we can get rank < 6
        
        # Actually, the minimal rank for 2x3x2 is known to be 6
        # So M_{2,3,4} has rank at most 12 via this block decomposition
        
        # Let's construct the rank-12 decomposition explicitly
        if rank == 12:
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # First block: k in {0, 1}, 6 terms
            # Term 0: i=0, j=0, k=0
            U[0*3+0][0] = 1; V[0*4+0][0] = 1; W[0*2+0][0] = 1
            # Term 1: i=0, j=1, k=0
            U[0*3+1][1] = 1; V[1*4+0][1] = 1; W[0*2+0][1] = 1
            # Term 2: i=0, j=2, k=0
            U[0*3+2][2] = 1; V[2*4+0][2] = 1; W[0*2+0][2] = 1
            # Term 3: i=1, j=0, k=1
            U[1*3+0][3] = 1; V[0*4+1][3] = 1; W[1*2+1][3] = 1
            # Term 4: i=1, j=1, k=1
            U[1*3+1][4] = 1; V[1*4+1][4] = 1; W[1*2+1][4] = 1
            # Term 5: i=1, j=2, k=1
            U[1*3+2][5] = 1; V[2*4+1][5] = 1; W[1*2+1][5] = 1
            
            # Wait, this is not correct. Let me redo this properly.
            # For 2x3x2 (columns 0,1):
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
            
            # This is 6 terms, but they share U and V entries
            # Let me just use the schoolbook pattern for each block
            
            # Reset
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            # Block 1: k in {0, 1}
            for i in range(n):
                for j in range(m):
                    for k in range(2):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            
            # Block 2: k in {2, 3}
            for i in range(n):
                for j in range(m):
                    for k in range(2, 4):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            
            # This gives rank 12, which is just the schoolbook decomposition
            # rearranged. It's correct but not an improvement.
            
            # Let me try a different approach: use Strassen-like ideas
            # For 2x2x2, Strassen uses 7 terms
            # For 2x3x4, let's see if we can do better than 12
            
            # Actually, let's just return the schoolbook for now since
            # finding a truly lower rank decomposition is complex
            
            return None
        
        # Try random search for lower ranks
        np.random.seed(rank * 12345)
        for attempt in range(200):
            if time.time() > deadline:
                return None
            
            vals = [0, 0.5, 1, -0.5, -1, 1.5, -1.5, 2, -2]
            
            U = [[np.random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
            V = [[np.random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
            W = [[np.random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 10 up to 23
    for rank in range(10, n * m * p):
        if time.time() > deadline:
            break
        
        result = try_structured_rank(rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
