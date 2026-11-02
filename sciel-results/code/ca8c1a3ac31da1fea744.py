"""
Strategy: Use Integer Linear Programming (ILP) to find a rank-minimal decomposition.
We formulate the tensor identity as a system of quadratic equations.
Since the variables are products of three matrices, we can introduce auxiliary variables or solve via iterative refinement.
However, a simpler and robust approach for small dimensions is to fix the rank and use a solver to find U, V, W.
Given the non-convexity, we will use a specific known structure or search.
For 4x4x6, the optimal rank is 84.
We will attempt to construct the solution using the known 4x4x4 (rank 49) + 4x4x2 (rank 35) decomposition.
Wait, 4x4x2 optimal rank is not 32. It is 20? No, 4x4x2 is equivalent to 4x4x4 with half the columns?
Actually, let's use a more direct search strategy.
We know rank(4,4,6) <= 84.
Let's try to construct a rank 84 solution.
Decomposition: 4x4x6 = 4x4x4 + 4x4x2.
Rank(4,4,4) = 49.
Rank(4,4,2) = ?
If we use Strassen for 4x4x4, we get 49.
For 4x4x2, we can use a specific decomposition.
Actually, a simpler structural change: Use the known optimal rank for 4x4x4 (49) and for 4x4x2 (which is 20? No, let's check).
Actually, let's just use the ILP approach to find a valid solution for rank 84, or fall back to 81 if 84 is not found quickly.
But the current program already achieves 81.
The best known rank for 4x4x6 is 84? Or is it lower?
Let's check the literature.
Rank(4,4,4) = 49.
Rank(4,4,6) <= Rank(4,4,4) + Rank(4,4,2).
What is Rank(4,4,2)?
It is the rank of the tensor $T_{ijk} = \delta_{ik}$ for $i,k \in 1..4, j \in 1..2$.
This is equivalent to multiplying a 4x4 matrix by a 4x2 matrix.
A known bound is 4*4*2 = 32.
Strassen-like improvements might exist.
However, the current program gets 81.
If we can find a decomposition of 4x4x2 with rank < 32, we improve.
Or if we can find a 4x4x4 with rank < 49? No, 49 is optimal.
So we need to optimize the 4x4x2 part.
Let's use a randomized search with local optimization for the 4x4x2 part, assuming the 4x4x4 part is fixed to the optimal 49-rank Strassen decomposition.
This is a structural change: Hybrid fixed + optimized.
"""
import time
import random
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}

    # Verify function
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # 1. Construct the optimal 4x4x4 decomposition (Rank 49)
    # This is the standard recursive Strassen decomposition.
    # We hardcode the 49-term decomposition for 4x4x4 to be safe and fast.
    
    # Coefficients for 2x2 Strassen
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will build the 4x4x4 tensor using these 7 blocks, each of which is a 2x2x2 tensor.
    # Total rank = 7 * 7 = 49.
    
    A_coeffs_2x2 = [
        [(1,0), (0,1)],  # M1: A00+A11
        [(1,0), (1,1)],  # M2: A10+A11
        [(0,0)],         # M3: A00
        [(1,1)],         # M4: A11
        [(0,0), (0,1)],  # M5: A00+A01
        [(1,0), (0,0)],  # M6: A10-A00 (note sign)
        [(0,1), (1,1)]   # M7: A01-A11 (note sign)
    ]
    # Correcting signs for M6, M7
    A_coeffs_2x2_signed = [
        { (0,0):1, (1,1):1 },
        { (1,0):1, (1,1):1 },
        { (0,0):1 },
        { (1,1):1 },
        { (0,0):1, (0,1):1 },
        { (1,0):1, (0,0):-1 },
        { (0,1):1, (1,1):-1 }
    ]
    B_coeffs_2x2_signed = [
        { (0,0):1, (1,1):1 },
        { (0,0):1 },
        { (0,1):1, (1,1):-1 },
        { (1,0):1, (0,0):-1 },
        { (1,1):1 },
        { (0,0):1, (0,1):1 },
        { (1,0):1, (1,1):1 }
    ]
    C_coeffs_2x2 = [
        [
            [1, 0, 0, -1, 1, 0, -1],  # C00
            [0, 0, 1, 0, 1, 0, 0]     # C01
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0]    # C11
        ]
    ]
    
    # Block coefficients for the outer 2x2 Strassen
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B01-B11
    # M4: A11, B10-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    block_A = [
        { (0,0):1, (1,1):1 },
        { (1,0):1, (1,1):1 },
        { (0,0):1 },
        { (1,1):1 },
        { (0,0):1, (0,1):1 },
        { (1,0):1, (0,0):-1 },
        { (0,1):1, (1,1):-1 }
    ]
    block_B = [
        { (0,0):1, (1,1):1 },
        { (0,0):1 },
        { (0,1):1, (1,1):-1 },
        { (1,0):1, (0,0):-1 },
        { (1,1):1 },
        { (0,0):1, (0,1):1 },
        { (1,0):1, (1,1):1 }
    ]
    
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    for beta in range(7):
        for alpha in range(7):
            col = beta * 7 + alpha
            
            # U_444
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    pi = i % 2
                    pj = j % 2
                    coeff = 0
                    for (r, s), v_a in block_A[beta].items():
                        if r == bi and s == bj:
                            for (pr, ps), v_inner in A_coeffs_2x2_signed[alpha].items():
                                if pr == pi and ps == pj:
                                    coeff += v_a * v_inner
                    U_444[i*4+j][col] = coeff
            
            # V_444
            for j in range(4):
                for k in range(4):
                    bj = j // 2
                    bk = k // 2
                    pj = j % 2
                    pk = k % 2
                    coeff = 0
                    for (r, s), v_b in block_B[beta].items():
                        if r == bj and s == bk:
                            for (pr, ps), v_inner in B_coeffs_2x2_signed[alpha].items():
                                if pr == pj and ps == pk:
                                    coeff += v_b * v_inner
                    V_444[j*4+k][col] = coeff
            
            # W_444
            for i in range(4):
                for k in range(4):
                    bi = i // 2
                    bk = k // 2
                    pi = i % 2
                    pk = k % 2
                    coeff = C_coeffs_2x2[bi][bk][beta] * C_coeffs_2x2[pi][pk][alpha]
                    W_444[k*4+i][col] = coeff

    # 2. Construct the 4x4x2 part.
    # We start with schoolbook (rank 32) and try to reduce it using local search.
    # The tensor is T[i,j,k] for i in 0..3, j in 0..3, k in 0..1.
    # Identity: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta[i,k]
    
    rank_442_init = 32
    U_442 = [[0]*rank_442_init for _ in range(16)]
    V_442 = [[0]*rank_442_init for _ in range(8)]
    W_442 = [[0]*rank_442_init for _ in range(8)]
    
    col_442 = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_442[i*4+j][col_442] = 1
                V_442[j*2+k][col_442] = 1
                W_442[k*4+i][col_442] = 1
                col_442 += 1
                
    # Verify 4x4x2
    def verify_442(U, V, W, rank):
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    s = 0
                    for c in range(rank):
                        s += U[i*4+j][c] * V[j*2+k][c] * W[k*4+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Attempt to reduce rank of 4x4x2 via random restarts and local optimization
    # We look for a rank 20 or 24 solution?
    # Known: Rank(4,4,2) is likely 20 or 24?
    # Actually, let's just try to find ANY valid decomposition with rank < 32.
    # If we find rank 28, total is 49+28=77.
    
    best_442_rank = 32
    best_U_442 = U_442
    best_V_442 = V_442
    best_W_442 = W_442
    
    # Simple local search: Randomly perturb entries and check validity?
    # Since the space is discrete (integers/half-integers), we can try small changes.
    # But 32 variables * 3 matrices is large.
    # Alternative: Use the structure of 4x4x2.
    # 4x4x2 is equivalent to 2x2x2 blocks? No.
    # Let's try a randomized search for a valid rank 24 decomposition.
    
    def random_search_442(target_rank, iterations=1000):
        for _ in range(iterations):
            if time.time() > deadline - 10:
                break
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(16)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(8)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(8)]
            
            # Normalize? No, just check.
            if verify_442(U, V, W, target_rank):
                return U, V, W
        return None

    # Try ranks 28, 24, 20
    for r in [28, 24, 20]:
        res = random_search_442(r, iterations=500)
        if res:
            best_442_rank = r
            best_U_442, best_V_442, best_W_442 = res
            break

    # Combine 4x4x4 and 4x4x2
    total_rank = 49 + best_442_rank
    U_total = [[0]*total_rank for _ in range(16)]
    V_total = [[0]*total_rank for _ in range(24)]
    W_total = [[0]*total_rank for _ in range(24)]
    
    # Copy 4x4x4
    for i in range(16):
        for c in range(49):
            U_total[i][c] = U_444[i][c]
            V_total[i][c] = V_444[i][c]
            W_total[i][c] = W_444[i][c]
            
    # Copy 4x4x2
    for i in range(16):
        for c in range(best_442_rank):
            U_total[i][49+c] = best_U_442[i][c]
    for j in range(8):
        for c in range(best_442_rank):
            V_total[j][49+c] = best_V_442[j][c]
    for k in range(8):
        for c in range(best_442_rank):
            W_total[k][49+c] = best_W_442[k][c]
            
    # Verify combined
    if verify(U_total, V_total, W_total, total_rank, 4, 4, 6):
        best = {'U': U_total, 'V': V_total, 'W': W_total, 'rank': total_rank}
        
    # Fallback to schoolbook if something went wrong
    if best['rank'] >= 96:
         best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}

    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
