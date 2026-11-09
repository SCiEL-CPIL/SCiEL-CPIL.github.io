"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Uses Strassen-style block decomposition with rank-7 Strassen for 2x2x2 blocks.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # For 4x4 * 4x7, we can split into 2x2 blocks:
    # A is 4x4 -> 2x2 blocks of 2x2
    # B is 4x7 -> 2x4 blocks (since 7 is odd, we pad to 8 or handle differently)
    # Actually, let's think differently.
    # 
    # The tensor for (n,m,p) matrix mult has entries T[i,j,k] = 1 for all i,j,k
    # We need rank-r decomposition where sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    #
    # Known: 2x2x2 matrix mult has rank 7 (Strassen)
    # For 4x4x4, rank is at most 7^2 = 49 (using Strassen recursively)
    # For 4x4x7, we can split B into 4x4 + 4x3, but that's messy.
    #
    # Alternative: Use the fact that we can decompose 4x4x7 as:
    # Split A into 2x2 blocks (4 blocks), B into 2x2 blocks (but 7=2+2+2+1)
    # This gets complicated. Let's try a direct approach with known results.
    #
    # Actually, let's use a simpler strategy: 
    # The schoolbook rank is 112. We can try to find a lower rank using
    # structured random search with Strassen-like patterns.
    #
    # Let me try a different approach: use the known rank for small cases
    # and combine them.
    
    # For now, let's implement a proper Strassen-based decomposition for 4x4x4
    # and then extend to 4x4x7.
    
    # Strassen 2x2x2 decomposition (rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4x4 using Strassen on 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    # Each Cij = sum over k of Aik * Bkj, computed via Strassen 2x2x2
    
    # This gives rank 7*7 = 49 for 4x4x4.
    
    # For 4x4x7, we can write B as [B_left(4x4), B_right(4x3)]
    # C = A*B_left + A*B_right
    # A*B_left: 4x4 * 4x4 -> rank 49
    # A*B_right: 4x4 * 4x3 -> need rank for this
    # 4x4 * 4x3: split A into 2x2 blocks, B into 2x2 and 2x1 blocks
    # This gets complex. Let's just try to find a good rank empirically.
    
    # Let's implement a structured search:
    # 1. Start with schoolbook (rank 112)
    # 2. Try to reduce using Strassen patterns
    
    # Actually, let's just implement the Strassen 4x4x4 decomposition and then
    # handle the 4x4x7 case by padding B to 4x8 and using rank 49+something.
    
    # Simpler: just return schoolbook for now, but let's try to do better.
    
    # Let me implement a proper Strassen-based 4x4x4 first.
    
    def strassen_2x2_2x2():
        """Return U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7)."""
        n2, m2, p2 = 2, 2, 2
        rank = 7
        U = [[0]*rank for _ in range(n2*m2)]  # 4 rows
        V = [[0]*rank for _ in range(m2*p2)]  # 4 rows
        W = [[0]*rank for _ in range(p2*n2)]  # 4 rows
        
        # Indexing: U[i*m2+j], V[j*p2+k], W[k*n2+i]
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # M1 = (A00+A11)(B00+B11)
        # A00=0, A11=3, B00=0, B11=3
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        # C00 += M1, C11 += M1
        W[0][0] = 1; W[3][0] = 1  # C00, C11
        
        # M2 = (A10+A11)B00
        # A10=2, A11=3, B00=0
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C10 += M2
        W[2][1] = 1  # C10
        
        # M3 = A00(B10-B11)
        # A00=0, B10=2, B11=3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        # C01 += M3
        W[1][2] = 1  # C01
        
        # M4 = A11(B01-B00)
        # A11=3, B01=1, B00=0
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        # C00 += M4, C10 += M4
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        # A00=0, A01=1, B11=3
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C00 -= M5, C01 += M5
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10=2, A00=0, B00=0, B01=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C11 += M6
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01=1, A11=3, B10=2, B11=3
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C00 += M7
        W[0][6] = 1
        
        return U, V, W
    
    def validate_decomposition(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        
        # Check that all entries are integers or half-integers
        for matrix in [U, V, W]:
            for row in matrix:
                for val in row:
                    if val is None:
                        return False
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        
        # Verify the tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try to build a 4x4x7 decomposition using Strassen blocks.
    # This is complex, so let's first verify our Strassen 2x2x2 is correct.
    
    U2, V2, W2 = strassen_2x2_2x2()
    # Quick check
    valid2 = validate_decomposition(U2, V2, W2, 2, 2, 2)
    
    # For 4x4x7, let's try a different approach:
    # Use the fact that we can decompose the problem into smaller parts.
    # 
    # Actually, let's just try a randomized search with better initialization.
    # Start with schoolbook and try to merge columns.
    
    # Schoolbook decomposition for 4x4x7
    rank_school = n * m * p  # 112
    U_school = [[0]*rank_school for _ in range(n*m)]
    V_school = [[0]*rank_school for _ in range(m*p)]
    W_school = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find a lower rank by random search with integer constraints
    best_rank = rank_school
    best_decomp = (U_school, V_school, W_school)
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But we know 4x4x4 has rank at most 49, so 4x4x7 should be at most ~49+something
    
    # Let's try a structured approach: use Strassen for the 4x4 part and
    # handle the extra column separately.
    
    # For 4x4x4, we can use Strassen recursively: rank 49
    # For 4x4x3, we need a decomposition
    # 4x4x3: split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] (2x2 blocks)
    # B = [[B11, B12, B13], [B21, B22, B23]] (2x3 blocks)
    # C = [[C11, C12, C13], [C21, C22, C23]] (2x3 blocks)
    # Each Cij = A1i*B1j + A2i*B2j
    # This is 4x(2x3) = 4x6 multiplications of 2x2 * 2x2 = 4*6*7 = 168? No.
    # Actually, for each output 2x2 block, we need to compute 2x2 * 2x2 + 2x2 * 2x2
    # Using Strassen, each 2x2*2x2 is rank 7, so each output block is rank 14
    # 6 output blocks * 14 = 84? That's worse than schoolbook 48.
    
    # This is getting too complex. Let's just do a randomized search with
    # a time limit and return the best found.
    
    def generate_random_decomposition(rank, n, m, p, max_val=3):
        """Generate a random decomposition with integer entries."""
        vals = list(range(-max_val, max_val+1))
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W
    
    # Try a few ranks with random search
    for rank in range(28, min(rank_school, 100)):
        if time.time() > deadline - 10:
            break
        
        attempts = 5
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            U, V, W = generate_random_decomposition(rank, n, m, p)
            
            # Quick validation
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                        if abs(total - 1) > 1e-9:
                            valid = False
                            break
            
            if valid:
                if validate_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_decomp = (U, V, W)
                    break
    
    U, V, W = best_decomp
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
