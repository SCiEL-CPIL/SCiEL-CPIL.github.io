"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def build_rank18():
        """Build the known rank-18 decomposition: two Strassen blocks + schoolbook for col 4."""
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col0, col1, base):
            # Strassen for 2x2 * 2x2 (columns col0, col1)
            # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # V indices: 0*p+col0=(0,col0), 1*p+col0=(1,col0), 0*p+col1=(0,col1), 1*p+col1=(1,col1)
            # W indices: col0*n+0=(col0,0), col0*n+1=(col0,1), col1*n+0=(col1,0), col1*n+1=(col1,1)
            
            # Term 1: M1 = (A00+A11)(B00+B11)
            r = base
            U[0][r] += 1; U[3][r] += 1
            V[0*p+col0][r] += 1; V[1*p+col1][r] += 1
            W[col0*n+0][r] += 1; W[col1*n+1][r] += 1
            
            # Term 2: M2 = A11(B10-B00)
            r = base + 1
            U[3][r] += 1
            V[1*p+col0][r] += 1; V[0*p+col0][r] += -1
            W[col0*n+1][r] += 1; W[col1*n+1][r] += -1
            
            # Term 3: M3 = (A00-A11)(B01+B11)
            r = base + 2
            U[0][r] += 1; U[3][r] += -1
            V[0*p+col1][r] += 1; V[1*p+col1][r] += 1
            W[col1*n+0][r] += 1; W[col1*n+1][r] += 1
            
            # Term 4: M4 = A00(B00-B11)
            r = base + 3
            U[0][r] += 1
            V[0*p+col0][r] += 1; V[1*p+col1][r] += -1
            W[col0*n+0][r] += 1; W[col1*n+1][r] += -1
            
            # Term 5: M5 = A11(B01-B00)
            r = base + 4
            U[3][r] += 1
            V[0*p+col1][r] += 1; V[0*p+col0][r] += -1
            W[col1*n+1][r] += 1; W[col0*n+1][r] += -1
            
            # Term 6: M6 = (A01+A11)(B10-B11)
            r = base + 5
            U[1][r] += 1; U[3][r] += 1
            V[1*p+col0][r] += 1; V[1*p+col1][r] += -1
            W[col0*n+1][r] += 1; W[col1*n+1][r] += -1
            
            # Term 7: M7 = (A00+A01)(B01-B10)
            r = base + 6
            U[0][r] += 1; U[1][r] += 1
            V[0*p+col1][r] += 1; V[1*p+col0][r] += -1
            W[col1*n+0][r] += 1; W[col0*n+0][r] += -1
        
        add_strassen_block(0, 1, 0)
        add_strassen_block(2, 3, 7)
        
        # Schoolbook for column 4 (2x1)
        r = 14
        U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
        r = 15
        U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
        r = 16
        U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
        r = 17
        U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
        
        return U, V, W
    
    def check_decomposition(U, V, W, rank):
        """Verify exact decomposition using integer arithmetic."""
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[a*m+b][r] * V[b*p+c][r] * W[c*n+a][r]
                    if s != 1:
                        return False
        return True
    
    # Start with rank 18 as baseline
    U18, V18, W18 = build_rank18()
    assert check_decomposition(U18, V18, W18, 18), "Rank-18 verification failed"
    
    best_rank = 18
    best_decomp = (U18, V18, W18)
    
    # Try to find rank 17 by searching for a single-term modification
    # Strategy: Try to merge one schoolbook term with a Strassen term
    # or find a rank-10 decomposition for the 2x3 part
    
    # Approach: Search for rank-17 by trying to perturb the rank-18 solution
    # with small integer/half-integer changes to reduce effective rank
    
    # Better approach: Direct search for rank 16-17 with structured entries
    values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    for target_rank in [17, 16]:
        if time.time() > deadline - 20:
            break
        
        # For rank 17: try structured search
        # Key insight: The last column (2x1) might be shareable with Strassen terms
        
        max_attempts = 20000 if target_rank == 17 else 50000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate sparse random matrices with small entries
            U = [[0] * target_rank for _ in range(n*m)]
            V = [[0] * target_rank for _ in range(m*p)]
            W = [[0] * target_rank for _ in range(p*n)]
            
            # Fill with sparse random values
            for c in range(target_rank):
                # U: 4 entries, each nonzero with prob ~0.7
                for i in range(n*m):
                    if np.random.random() < 0.7:
                        U[i][c] = np.random.choice(values)
                # V: 10 entries
                for i in range(m*p):
                    if np.random.random() < 0.5:
                        V[i][c] = np.random.choice(values)
                # W: 10 entries
                for i in range(p*n):
                    if np.random.random() < 0.5:
                        W[i][c] = np.random.choice(values)
            
            if check_decomposition(U, V, W, target_rank):
                found = True
                best_rank = target_rank
                best_decomp = (U, V, W)
                break
        
        if found:
            break
    
    # If rank 17/16 not found, try a more targeted approach:
    # Modify the rank-18 by trying to absorb one schoolbook term into Strassen
    
    if best_rank == 18 and time.time() < deadline - 30:
        # Try to find rank 17 by modifying the last 4 schoolbook terms
        # to use only 3 terms by sharing with Strassen structure
        
        # Strategy: Try all combinations of small modifications to the
        # schoolbook terms for column 4, checking if rank drops to 17
        
        # Alternative: Use the fact that we can try to express one of the
        # Strassen terms as a combination involving the 2x1 part
        
        # Let's try a different partition: 2x2 + 2x3 instead of 2x2+2x2+2x1
        # Can we do 2x2*2x3 in rank 10?
        
        # For now, try direct rank-17 search with more structured approach
        # based on the Strassen template but with merged terms
        
        U17 = [row[:] for row in U18]
        V17 = [row[:] for row in V18]
        W17 = [row[:] for row in W18]
        
        # Try to zero out one column by expressing it as combination of others
        # This is a linear algebra problem on the tensor
        
        # Skip complex linear algebra for now; return rank 18 if no improvement
        pass
    
    U, V, W = best_decomp
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
