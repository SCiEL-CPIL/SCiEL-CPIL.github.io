"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    deadline = start_time + 130

    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # For X(3x3) * Y(3x3) = Z(3x3):
    # The 6 diagonal terms: Z[i][i] += X[i][i]*Y[i][i] for i=0,1,2
    # Plus 7 off-diagonal terms using Laderman's 7 extra multiplications
    # Actually Laderman uses 7 total multiplications for the off-diagonal part
    # Total: 6 + 7 = 13? No, Laderman's algorithm uses 23 multiplications total for 3x3.
    # Let me use the known decomposition.
    
    # Actually, let me use a simpler approach: split B into two 4x3 blocks
    # and use a rank-23 algorithm for each 3x4 x 4x3 multiplication.
    
    # For 3x4 x 4x3: this is a "rectangular" matrix multiplication.
    # We can embed it in 4x4 x 4x4 (rank 49 schoolbook, or use Strassen-like).
    # Or we can directly construct a decomposition.
    
    # Simpler: use the fact that 3x4 x 4x3 can be done with rank 3*4*3 = 36 schoolbook.
    # But we can do better. Let's use a 2x2 block Strassen approach on the 4x4 part.
    
    # Pad A (3x4) to 4x4 by adding a zero row: A_padded (4x4)
    # Pad B (4x3) to 4x4 by adding a zero column: B_padded (4x4)
    # Then A_padded * B_padded gives a 4x4 result, top-left 3x3 is A*B.
    # Use Strassen for 4x4: rank 7 (but Strassen is for 2x2, so 4x4 = 2x2 blocks of 2x2)
    # Strassen for 2x2: rank 7 instead of 8. For 4x4 as 2x2 blocks: rank 7*7=49? No.
    # Actually 4x4 matrix mult can be done with 49 multiplications using Strassen recursively,
    # but we only need the 3x3 subproduct.
    
    # Let me just implement a direct rank reduction.
    # For 3x4 x 4x3, the tensor has 3*4*3 = 36 entries (all 1s in the appropriate positions).
    # We want to express it as sum of rank-1 tensors.
    
    # Known result: the rank of the 3x3 matrix multiplication tensor is 23 (Laderman).
    # For 3x4 x 4x3, we can view it as: take a 3x3 x 3x3 tensor (rank 23) and 
    # add one extra column to the first factor and one extra row to the second factor.
    # The extra terms: X[3][j] * Y[j][3] for j=0,1,2 (3 terms) plus X[i][3]*Y[3][k]... 
    # This gets complicated. Let me just use the schoolbook for now but try to optimize.
    
    # Actually, the cleanest approach: use Laderman's algorithm directly.
    # Laderman's algorithm for 3x3 * 3x3 = 3x3 uses 23 scalar multiplications.
    
    # For our case: A is 3x4, B is 4x3.
    # Split A = [A1 | a] where A1 is 3x3, a is 3x1 (column vector)
    # Split B = [b; B1] where b is 1x3 (row vector), B1 is 3x3
    # Then A*B = A1*B1 + a*b
    # A1*B1 is 3x3 * 3x3 = 3x3, rank 23 (Laderman)
    # a*b is an outer product, rank 3 (each element a[i]*b[j])
    # Total rank: 23 + 3 = 26 for one 3x4 x 4x3 block.
    
    # For the full 3x4 x 4x6: split B into two 4x3 blocks B1, B2
    # C = [A*B1 | A*B2], each 3x4 x 4x3 with rank 26
    # Total rank: 26 + 26 = 52
    
    # Let me implement this.
    
    def laderman_3x3():
        """Return U, V, W factors for 3x3 * 3x3 = 3x3 with rank 23.
        U has shape (3*3, 23), V has shape (3*3, 23), W has shape (3*3, 23).
        Indexing: U[i*3+j, r] for A[i][j], V[i*3+j, r] for B[i][j], W[i*3+j, r] for C[i][j]
        """
        rank = 23
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        
        # Laderman's 23 terms:
        # Terms 0-2: diagonal
        # T0: A[0][0]*B[0][0] -> C[0][0]
        # T1: A[1][1]*B[1][1] -> C[1][1]  
        # T2: A[2][2]*B[2][2] -> C[2][2]
        
        # Terms 3-8: off-diagonal via Laderman
        # Laderman's algorithm uses 7 additional multiplications beyond the 6 diagonal... 
        # Actually no. Let me look up the exact Laderman decomposition.
        
        # Laderman's algorithm: 23 multiplications for 3x3.
        # The decomposition is:
        # m1 = A[0][0] * B[0][0]
        # m2 = A[1][1] * B[1][1]
        # m3 = A[2][2] * B[2][2]
        # m4 = (A[0][1]+A[0][0]) * (B[1][0]-B[0][0])
        # m5 = (A[1][2]+A[1][1]) * (B[2][1]-B[1][1])
        # m6 = (A[2][0]+A[2][2]) * (B[0][2]-B[2][2])
        # m7 = (A[0][2]+A[2][0]+A[2][2]) * (B[2][0]+B[0][2]-B[2][2])
        # m8 = (A[1][0]+A[0][0]+A[0][1]) * (B[0][1]+B[1][0]-B[0][0])
        # ... this is getting complex. Let me use a known explicit decomposition.
        
        # Actually, let me use a simpler known rank-23 decomposition.
        # I'll use the one from the literature.
        
        # For now, let me just use schoolbook for 3x3 (rank 27) to be safe,
        # then total rank = 2*(27+3) = 60, still better than 72.
        
        # Actually, let me just do schoolbook for the whole thing and verify,
        # then try to optimize.
        return None  # placeholder
    
    # Let me just implement the direct construction with the split approach.
    # A (3x4) = [A_left (3x3) | a (3x1)]
    # B (4x6) = [b_top (1x6); B_bottom (3x6)]
    # C = A_left * B_bottom + a * b_top
    # A_left * B_bottom: 3x3 * 3x6, rank 3*3*6 = 54 schoolbook
    # a * b_top: 3x1 * 1x6, rank 3*1*6 = 18 schoolbook (but it's just outer product, rank 6 actually since each C[i][j] = a[i]*b[j])
    # Wait, a*b_top is 3x6, each element C[i][j] = a[i]*b[j]. This is rank 6 (one multiplication per element).
    # Total: 54 + 6 = 60. Better than 72!
    
    # But we can do better on the 3x3 * 3x6 part.
    # 3x3 * 3x6: split B_bottom into two 3x3 blocks.
    # A_left * B_bottom1 (3x3*3x3) + A_left * B_bottom2 (3x3*3x3)
    # Each 3x3*3x3 has rank 23 (Laderman) or 27 (schoolbook).
    # With Laderman: 2*23 = 46, plus 6 for outer product = 52 total.
    # With schoolbook: 2*27 = 54, plus 6 = 60 total.
    
    # Let me implement with schoolbook first to get a valid solution of rank 60,
    # then try to improve to 52 with Laderman.
    
    def make_solution_60():
        """Rank 60 solution using 3x3*3x6 + outer product split."""
        # A (3x4): indices (i,j) where i=0..2, j=0..3
        # B (4x6): indices (j,k) where j=0..3, k=0..5
        # C (3x6): indices (i,k) where i=0..2, k=0..5, output as W[k*3+i]
        
        rank = 60
        U = [[0]*rank for _ in range(n*m)]  # 12 x 60
        V = [[0]*rank for _ in range(m*p)]  # 24 x 60
        W = [[0]*rank for _ in range(p*n)]  # 18 x 60
        
        col = 0
        # Part 1: A_left (3x3, cols 0-2) * B_bottom (3x6, rows 1-3)
        # A_left[i][j] for i=0..2, j=0..2 (original A[i][j])
        # B_bottom[j][k] for j=0..2 (original B[j+1][k]), k=0..5
        # C[i][k] += A_left[i][j] * B_bottom[j][k]
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i*m + j][col] = 1       # A[i][j] where j=0..2
                    V[(j+1)*p + k][col] = 1   # B[j+1][k]
                    W[k*n + i][col] = 1       # C[i][k]
                    col += 1
        # Part 2: a (3x1, col 3 of A) * b_top (1x6, row 0 of B)
        # C[i][k] += A[i][3] * B[0][k]
        for i in range(3):
            for k in range(6):
                U[i*m + 3][col] = 1       # A[i][3]
                V[0*p + k][col] = 1       # B[0][k]
                W[k*n + i][col] = 1       # C[i][k]
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Verify
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    ui = U[i*m + j]
                    vj = V[j*p + k]
                    wk = W[k*n + i]
                    for r in range(rank):
                        s += ui[r] * vj[r] * wk[r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with the rank-60 solution
    result = make_solution_60()
    if verify(result["U"], result["V"], result["W"]):
        best_result = result
        best_rank = 60
    else:
        # Fallback to schoolbook
        def make_schoolbook():
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
        best_result = make_schoolbook()
        best_rank = n * m * p
    
    # Now try to improve using Laderman's algorithm for the 3x3*3x3 parts
    # to get rank 52.
    
    def laderman_decomposition():
        """
        Laderman's algorithm for 3x3 matrix multiplication.
        Returns list of (U_factors, V_factors, W_factors) for each of the 23 terms.
        Each term is a rank-1 tensor: U term is a 3x3 matrix, V is 3x3, W is 3x3.
        """
        # Laderman's 23 multiplications for 3x3 * 3x3 = 3x3
        # Source: Laderman, "The 3x3 matrix multiplication algorithm" (1976)
        # The algorithm uses 23 scalar multiplications.
        
        # I'll use a known explicit decomposition.
        # Let me define the 23 terms.
        
        terms = []
        
        # Helper to add a term
        def add_term(u_mat, v_mat, w_mat):
            """u_mat, v_mat, w_mat are 3x3 lists of coefficients."""
            terms.append((u_mat, v_mat, w_mat))
        
        # Identity terms (6 diagonal)
        for i in range(3):
            u = [[0,0,0],[0,0,0],[0,0,0]]
            v = [[0,0,0],[0,0,0],[0,0,0]]
            w = [[0,0,0],[0,0,0],[0,0,0]]
            u[i][i] = 1
            v[i][i] = 1
            w[i][i] = 1
            add_term(u, v, w)
        
        # Now 17 more terms for the off-diagonal elements.
        # Actually Laderman uses 23 total, so 17 more after 6 diagonal.
        # Let me use a standard reference implementation.
        
        # From various references, Laderman's algorithm:
        # m1 = A00*B00
        # m2 = A11*B11
        # m3 = A22*B22
        # m4 = (A01+A00)*(B10-B00)
        # m5 = (A12+A11)*(B21-B11)
        # m6 = (A20+A22)*(B02-B22)
        # m7 = (A02+A20+A22)*(B20+B02-B22)
        # m8 = (A10+A00+A01)*(B01+B10-B00)
        # m9 = (A02+A20)*(B20+B02)
        # m10 = (A01+A11)*(B10+B01)  -- wait, I need to be more careful.
        
        # Let me use a verified decomposition from the literature.
        # I'll use the one from "Fast Matrix Multiplication" by Coppersmith and Winograd,
        # or a standard reference.
        
        # Actually, let me just use a known correct set of 23 terms.
        # Here's a standard Laderman decomposition:
        
        # Term 1: A[0][0] * B[0][0] -> C[0][0]
        # Term 2: A[1][1] * B[1][1] -> C[1][1]
        # Term 3: A[2][2] * B[2][2] -> C[2][2]
        # Term 4: (A[0][1] + A[0][0]) * (B[1][0] - B[0][0])
        #   -> C[0][1] += term, C[1][0] -= term, C[0][0] += term
        # Term 5: (A[1][2] + A[1][1]) * (B[2][1] - B[1][1])
        #   -> C[1][2] += term, C[2][1] -= term, C[1][1] += term
        # Term 6: (A[2][0] + A[2][2]) * (B[0][2] - B[2][2])
        #   -> C[2][0] += term, C[0][2] -= term, C[2][2] += term
        # Term 7: (A[0][2] + A[2][0] + A[2][2]) * (B[2][0] + B[0][2] - B[2][2])
        #   -> C[0][2] += term, C[2][0] += term, C[2][2] -= term, C[0][0] += term, C[2][2]... 
        # This is getting too complex to get right from memory.
        
        # Let me use a different approach: I'll verify computationally.
        # For now, let me just return the 60-rank solution which is verified.
        return None
    
    # Since implementing Laderman correctly from memory is error-prone,
    # let me try a different optimization: use the rank-60 solution and then
    # try random search to reduce further.
    
    # Actually, let me try another structured approach.
    # Split A (3x4) into 2+2 columns: A = [A1 (3x2) | A2 (3x2)]
    # Split B (4x6) into 2+2 rows: B = [B1 (2x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    # Each term: 3x2 * 2x6 = 3x6, rank 3*2*6 = 36
    # Total: 72. No improvement.
    
    # Try: A (3x4) split as 1+3: A = [a (3x1) | A2 (3x3)]
    # B (4x6) split as 1+3: B = [b (1x6); B2 (3x6)]
    # C = a*b + A2*B2
    # a*b: 3x1 * 1x6 = 3x6, rank 3*1*6 = 18 (but actually rank 6 since it's outer product)
    # A2*B2: 3x3 * 3x6 = 3x6, rank 3*3*6 = 54
    # Total: 18 + 54 = 72. No improvement with schoolbook for the big part.
    
    # The 60-rank solution is already an improvement. Let me try to verify it works
    # and then attempt random search for further improvement.
    
    # Try random search to reduce rank below 60
    for target_rank in range(best_rank - 1, max(40, best_rank - 15)):
        if time.time() > deadline - 30:
            break
        
        attempts = 0
        max_attempts = 200
        while time.time() < deadline - 30 and attempts < max_attempts:
            attempts += 1
            
            U = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify(U, V, W):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result
# EVOLVE-BLOCK-END
