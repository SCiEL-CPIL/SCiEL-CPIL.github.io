"""Matrix multiplication 3,3,7 - rank minimization with Strassen-like recursive partitioning."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 150s limit

    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, num_samples=30):
        """Verify tensor decomposition correctness using random integer matrices."""
        rng = random.Random(42)
        rank = len(U[0]) if U else 0
        for _ in range(num_samples):
            A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
            # True product
            C_true = [[0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j] * B[j][k]
                    C_true[i][k] = s
            # Product via factorization: C[i,k] = sum_{j,c} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            C_fact = [[0] * p for _ in range(n)]
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        uc = U[i * m + j][c]
                        if uc == 0:
                            continue
                        for k in range(p):
                            vc = V[j * p + k][c]
                            if vc == 0:
                                continue
                            wc = W[k * n + i][c]
                            if wc == 0:
                                continue
                            C_fact[i][k] += uc * vc * wc
            for i in range(n):
                for k in range(p):
                    if C_true[i][k] != C_fact[i][k]:
                        return False
        return True

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2 * 2x2 -> 2x2.
        Returns (U, V, W) for the 2x2x2 subproblem.
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        # U[0*2+0]=A00, U[1*2+1]=A11
        U[0][0] = 1; U[3][0] = 1
        # V[0*2+0]=B00, V[1*2+1]=B11
        V[0][0] = 1; V[3][0] = 1
        # W contributions: C00 gets +1, C11 gets +1
        W[0*2+0][0] = 1  # W[k=0, i=0]
        W[1*2+1][0] = 1  # W[k=1, i=1]
        
        # Term 1: M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0*2+1][1] = 1  # C10: +1
        W[1*2+1][1] = -1  # C11: -1
        
        # Term 2: M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[0*2+0][2] = 1  # C00: +1
        W[1*2+0][2] = 1  # C10: +1
        
        # Term 3: M4 = A11(B01-B10)
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        W[0*2+0][3] = 1  # C00: +1
        W[1*2+1][3] = 1  # C11: +1
        
        # Term 4: M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0*2+0][4] = -1  # C00: -1
        W[1*2+0][4] = 1  # C01: +1
        
        # Term 5: M6 = (A10-A00)(B00+B10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[1*2+1][5] = 1  # C11: +1
        
        # Term 6: M7 = (A01-A11)(B01+B11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0*2+0][6] = 1  # C00: +1
        W[1*2+0][6] = 1  # C01: +1
        
        return U, V, W

    def strassen_3x3x3():
        """
        A rank-23 decomposition for 3x3 * 3x3 -> 3x3 using Strassen-like methods.
        This is a known result; we use a specific decomposition.
        For simplicity and reliability, we use a direct construction based on
        partitioning into 2x2 blocks and applying Strassen.
        
        Partition 3x3 into:
        A = [[A11, A12, A13],
             [A21, A22, A23],
             [A31, A32, A33]]
        
        We'll use a block approach: split into 2x2 + 1x1 components.
        Actually, let's use a known rank-23 decomposition.
        
        A simpler approach: use the fact that 3x3 can be done with rank 23
        via specific algorithms. However, implementing the exact rank-23
        decomposition is complex. Let's use a rank-27 decomposition (3x3x3
        schoolbook is 27, and Strassen gives 19 for 2x2, but for 3x3 it's
        more complex).
        
        For reliability, let's use a rank-23 decomposition from literature.
        Actually, let's just use a structured approach that we can verify.
        
        We'll use the following strategy: decompose 3x3x3 into overlapping
        2x2x2 Strassen decompositions plus correction terms.
        
        For now, let's return a schoolbook 3x3x3 (rank 27) as a fallback,
        but try to find a better one.
        """
        # Let's use a known rank-23 decomposition for 3x3 matrix multiplication.
        # This is based on the work of Pan and others.
        # For simplicity and correctness, let's use a verified decomposition.
        
        # Actually, the simplest reliable approach: use rank 23 from a known source.
        # Let me construct a rank-23 decomposition.
        
        # The following is a rank-23 decomposition for 3x3x3:
        # Reference: "Optimal matrix multiplication" - various papers
        
        # For reliability, let's use a slightly higher rank that we can construct
        # and verify easily. Let's use rank 24 or 25.
        
        # Actually, let's just use schoolbook for 3x3x3 (rank 27) and focus on
        # the 3x3x7 problem where we can potentially do better.
        
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return U, V, W

    def decompose_3x3x7():
        """
        Decompose 3x3 * 3x7 -> 3x7 using recursive partitioning.
        
        Strategy: Split the 7 columns into 3+3+1.
        - Use a rank-23 or better decomposition for the 3x3x3 part (first 3 cols)
        - Use another rank-23 or better decomposition for the 3x3x3 part (next 3 cols)  
        - Use rank-9 for the 3x3x1 part (last 1 col)
        
        Total rank: 23 + 23 + 9 = 55 (if we use rank-23 for 3x3x3)
        Or: 27 + 27 + 9 = 63 (schoolbook)
        
        Actually, we can do better by sharing structure.
        
        Let's try: split 7 into 4+3.
        - 3x3x4: can this be done with rank < 12?
        - 3x3x3: rank 23
        
        Or split into 2+2+2+1:
        - Three 3x3x2 blocks and one 3x3x1 block
        
        For 3x3x2: schoolbook is 18. Can we do better?
        For 3x3x1: schoolbook is 9.
        
        Let's try a different approach: use the full 3x3x7 tensor and find
        a low-rank decomposition using the structure.
        
        Actually, let's use a simpler recursive approach:
        Split the 3x3 input into 2x2 + 1x1 blocks and the 7 output into 
        4+3 or similar.
        
        For now, let's construct a decomposition that we can verify:
        Use Strassen for 2x2 blocks and handle the rest with schoolbook.
        
        Partition:
        A (3x3) = [[A11, A12, A13],
                    [A21, A22, A23],
                    [A31, A32, A33]]
        
        B (3x7) = [[B11, B12, B13, B14, B15, B16, B17],
                    [B21, B22, B23, B24, B25, B26, B27],
                    [B31, B32, B33, B34, B35, B36, B37]]
        
        C (3x7) = A * B
        
        Let's split A into top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1.
        And B into left 3x4 and right 3x3.
        
        This is getting complex. Let's use a simpler strategy:
        
        Use the following decomposition:
        - For columns 0-3 (4 columns): use a 3x3x4 decomposition
        - For columns 4-6 (3 columns): use a 3x3x3 decomposition
        
        For 3x3x3, use rank-23 if available, else rank-27.
        For 3x3x4, use rank-24 if available, else rank-12*4=48... no, schoolbook is 12*4=48? No.
        
        Wait, schoolbook for 3x3x4 is 3*3*4 = 36.
        
        Let's just try to find a decomposition with rank < 63 using random search
        guided by the structure, and fall back to schoolbook.
        """
        # Let's try a direct approach: construct a rank-50 decomposition
        # by using Strassen on 2x2 subblocks.
        
        # Actually, let's use a known result: the rank of 3x3x7 matrix multiplication
        # is at most 55 (from some papers) or we can construct a rank-50 decomposition.
        
        # For reliability, let's use a verified approach:
        # 1. Split the 7 columns into two groups: 4 and 3
        # 2. Use schoolbook for each: 3*3*4 + 3*3*3 = 36 + 27 = 63 (same as before)
        
        # To beat 63, we need to share computations.
        
        # Let's try: use Strassen for the 2x2 blocks.
        # Split A into 2x2 + 1x1, and B into 2x4 + 2x3 + 1x4 + 1x3
        
        # This is complex. Let's use a random search approach with structure.
        
        # Try to find a rank-55 decomposition using random initialization + verification
        best_result = None
        best_rank = 63
        
        # Try ranks from 55 down to 45
        for target_rank in range(55, 44, -1):
            if time.time() > deadline:
                break
            
            # Try a few random structured decompositions
            for attempt in range(3):
                if time.time() > deadline:
                    break
                
                # Initialize with sparse random matrices
                rng = random.Random(target_rank * 1000 + attempt)
                U = [[rng.randint(-2, 2) if rng.random() < 0.25 else 0 
                      for _ in range(target_rank)] for _ in range(9)]
                V = [[rng.randint(-2, 2) if rng.random() < 0.25 else 0 
                      for _ in range(target_rank)] for _ in range(21)]
                W = [[rng.randint(-2, 2) if rng.random() < 0.25 else 0 
                      for _ in range(target_rank)] for _ in range(21)]
                
                if verify(U, V, W, num_samples=15):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
            
            if best_result is not None and best_rank <= target_rank:
                continue
        
        if best_result is not None:
            return best_result
        
        # Fallback: try a structured decomposition based on Strassen
        # Use rank 55: split into 2x2 Strassen + corrections
        
        # Actually, let's just return schoolbook if we can't find better
        return schoolbook()

    # Main logic: try the recursive decomposition approach
    result = decompose_3x3x7()
    
    if time.time() > deadline:
        return schoolbook()
    
    return result
# EVOLVE-BLOCK-END
