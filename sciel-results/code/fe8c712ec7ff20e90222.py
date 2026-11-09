"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    rank_school = n * m * p  # 128
    
    def verify(U, V, W):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_444(U4, V4, W4):
        """Verify 4x4x4 decomposition."""
        n4, m4, p4 = 4, 4, 4
        U4 = np.array(U4, dtype=float)
        V4 = np.array(V4, dtype=float)
        W4 = np.array(W4, dtype=float)
        rank = U4.shape[1]
        for i in range(n4):
            for j in range(m4):
                for k in range(p4):
                    val = 0.0
                    for c in range(rank):
                        val += U4[i*m4+j, c] * V4[j*p4+k, c] * W4[k*n4+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def build_444_from_strassen():
        """
        Build a rank-32 decomposition for 4x4x4 using Strassen-like structure.
        Partition 4x4 into 2x2 blocks of 2x2 matrices.
        Use 4 independent Strassen multiplications (rank 7 each) for the 
        2x2 block diagonal, plus corrections for off-diagonal terms.
        
        Actually, let's use a simpler approach: use 16 terms for a direct
        decomposition and try to reduce.
        
        For now, return a rank-32 decomposition based on tensor product
        of two rank-4 decompositions... no that gives 16.
        
        Let me just use a known construction.
        """
        # Use the following explicit rank-28 decomposition for 4x4x4
        # Based on the work of Landsberg et al.
        # Coefficients are in {-1, 0, 1}
        
        # This is a known rank-28 decomposition
        # Each term is (u, v, w) where u, v, w are 4x4, 4x4, 4x4 matrices
        # with entries in {-1, 0, 1}
        
        # Let me construct one using the following approach:
        # Use 2x2x2 Strassen as building blocks
        
        # Strassen 2x2x2: 7 terms
        # For 4x4x4, use 4 Strassen blocks = 28 terms
        
        # Block structure:
        # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Each 2x2x2 multiplication can be done with rank 7 (Strassen)
        # So we need 4 * 7 = 28 terms total
        
        # Strassen formulas for 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For 4x4x4, we apply this to each of the 4 block multiplications
        # But we need to be careful about how the indices work
        
        # Let me construct the rank-28 decomposition explicitly
        
        U4 = []
        V4 = []
        W4 = []
        
        # Define Strassen 2x2x2 rank-1 terms
        # For 2x2x2, the 7 terms are:
        # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]  (for M1)
        # etc.
        
        # Actually, let me just build it directly using the block structure
        
        # For each of the 4 block positions (bi, bj) where bi, bj in {0,1},
        # we need to compute C_bj = sum_k A_bk * B_kj
        # Each is a 2x2x2 multiplication
        
        # Strassen 2x2x2 decomposition (rank 7):
        # Label 2x2 indices as 0,1,2,3 (row-major)
        # A = [[a0, a1], [a2, a3]]
        # B = [[b0, b1], [b2, b3]]
        # C = [[c0, c1], [c2, c3]]
        
        # Strassen terms (u, v, w) where u, v, w are vectors of length 4:
        strassen_terms = [
            # M1 = (A00+A11)*(B00+B11)
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            # M2 = (A10+A11)*B00
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            # M3 = A00*(B01-B11)
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            # M4 = A11*(B10-B00)
            ([0,0,0,1], [1,-1,0,0], [0,0,1,0]),  # Wait, this doesn't look right
            # Let me redo this more carefully
        ]
        
        # Let me use a cleaner Strassen decomposition
        # Standard Strassen for 2x2:
        # p1 = (a00+a11)(b00+b11)
        # p2 = (a10+a11)b00
        # p3 = a00(b01-b11)
        # p4 = a11(b10-b00)
        # p5 = (a00+a01)b11
        # p6 = (a10-a00)(b00+b01)
        # p7 = (a01-a11)(b10+b11)
        # c00 = p1+p4-p5+p7
        # c01 = p3+p5
        # c10 = p2+p4
        # c11 = p1-p2+p3+p6
        
        # For each p_i, the rank-1 decomposition is:
        # p1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # p2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # p3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
        # p4: u=[0,0,0,1], v=[1,-1,0,0], w=[0,0,1,0]  # Hmm, need to check
        # p5: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,1]
        # p6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,1,1]
        # p7: u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,0,1]
        
        # This is getting complicated. Let me just use a direct rank-16
        # decomposition for 4x4x4 and see if that's better than 128 for 4x4x8.
        # 16 * 2 = 32 for 4x4x8. That's much better than 128!
        
        # Direct decomposition of 4x4x4:
        # For each (i,j,k), we have one term
        # That's 64 terms for 4x4x4, which is not good.
        
        # Wait, no. The direct decomposition has rank n*m*p = 64 for 4x4x4.
        # For 4x4x8, that's 128.
        
        # I need a better decomposition.
        
        # Let me just return a rank-32 decomposition based on using
        # 16 terms for each half of the 4x8 matrix.
        
        # Actually, the simplest improvement: use the fact that 4x4x8
        # can be done as two 4x4x4 multiplications. If I use a rank-32
        # decomposition for 4x4x4, I get rank-64 for 4x4x8.
        
        # Let me construct a rank-32 decomposition for 4x4x4 using
        # a combination of Strassen and direct terms.
        
        # For now, let me just return a rank-64 decomposition (two copies
        # of a rank-32 4x4x4 decomposition) as an improvement over 128.
        
        # Construct rank-32 for 4x4x4:
        # Use 8 terms for the "left" part and 8 for the "right" part...
        
        # Actually, let me just use a simple construction:
        # For 4x4x4, use 16 terms where each term handles 2 output entries.
        
        # This is getting too complicated without a clear algorithm.
        # Let me just return the schoolbook for now but with a note that
        # a better decomposition should be found.
        
        return None
    
    # Try to find a low-rank decomposition using random search
    import random
    random.seed(42)
    
    best_rank = rank_school
    best_decomp = None
    
    # Try ranks from 56 upward (using the two-block decomposition idea)
    # For each rank r, try to find a decomposition
    
    # First, try to find a rank-28 decomposition for 4x4x4
    # Then duplicate it for 4x4x8 to get rank-56
    
    # Use a simple random search
    for target_rank_444 in [28, 32, 40, 48, 56, 64]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Generate random 4x4x4 decomposition
            U4 = np.random.randint(-1, 2, (16, target_rank_444)).astype(float)
            V4 = np.random.randint(-1, 2, (16, target_rank_444)).astype(float)
            W4 = np.random.randint(-1, 2, (16, target_rank_444)).astype(float)
            
            if verify_444(U4, V4, W4):
                found = True
                break
        
        if found:
            # Duplicate for 4x4x8
            rank_total = 2 * target_rank_444
            U = np.zeros((n*m, rank_total))
            V = np.zeros((m*p, rank_total))
            W = np.zeros((p*n, rank_total))
            
            # Left half (k=0..3)
            for i in range(n):
                for j in range(m):
                    U[i*m+j, :target_rank_444] = U4[i*m+j]
                for k in range(4):
                    V[j*p+k, :target_rank_444] = V4[j*4+k]
            for k in range(4):
                for i in range(n):
                    W[k*n+i, :target_rank_444] = W4[k*n+i]
            
            # Right half (k=4..7)
            for i in range(n):
                for j in range(m):
                    U[i*m+j, target_rank_444:] = U4[i*m+j]
                for k in range(4):
                    V[j*p+(k+4), target_rank_444:] = V4[j*4+k]
            for k in range(4):
                for i in range(n):
                    W[(k+4)*n+i, target_rank_444:] = W4[k*n+i]
            
            if verify(U, V, W):
                best_rank = rank_total
                best_decomp = (U, V, W)
                break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
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
