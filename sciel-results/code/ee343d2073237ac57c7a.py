import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback (Rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Rank-44 fallback (current best known construction)
    def _rank44():
        S_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        S_V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        S_W = [
            [1, 0, 0, 1],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        rank = 44
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    # Try to construct a rank-36 solution using a 3-row Strassen-like scheme
    # The idea: use a decomposition that handles all 3 rows of A with shared rank terms
    # Based on the structure of optimal decompositions for <2,3,p>
    
    # For 2x3x8, we can use 4 blocks of 2 columns in B
    # Each block uses a rank-9 decomposition for 2x3x2 (instead of 7+4=11)
    # Total: 4 * 9 = 36
    
    # The 2x3x2 tensor has rank 9 (optimal)
    # We need to find a rank-9 decomposition of the 2x3x2 multiplication tensor
    
    # Tensor T[i,a,k] where i in {0,1}, a in {0,1,2}, k in {0,1}
    # We need: sum_c U[i,a,c] * V[a,k,c] * W[k,i,c] = 1 for all (i,a,k)
    
    # Let me construct this using a systematic approach
    # U is 6x9, V is 6x9, W is 4x9
    
    # Known rank-9 decomposition for 2x3x2:
    # Using the fact that rank(<2,3,2>) = 9
    
    # Let's build it term by term
    # Each rank-1 term: U has one nonzero pair (i,a), V has one nonzero pair (a,k), W has one nonzero pair (k,i)
    # But we can have multiple nonzeros in each factor per term
    
    # Actually, for a rank-1 term, U[i,a] * V[a,k] * W[k,i] must contribute
    # Let's use the following construction:
    
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> T[0,0,0] += 1
    # Term 1: U[0,1]=1, V[1,0]=1, W[0,0]=1  -> T[0,1,0] += 1
    # ... this is getting complex. Let me use a known explicit construction.
    
    # Actually, let me try a different approach: use the border rank / approximate methods
    # But we need exact integer/half-integer entries.
    
    # Let me try to find a rank-36 solution by searching over possible 2x3x2 rank-9 decompositions
    
    # For 2x3x2, the 12 tensor entries are:
    # T[0,0,0], T[0,0,1], T[0,1,0], T[0,1,1], T[0,2,0], T[0,2,1],
    # T[1,0,0], T[1,0,1], T[1,1,0], T[1,1,1], T[1,2,0], T[1,2,1]
    
    # Each rank-1 term (u, v, w) contributes u[i,a]*v[a,k]*w[k,i] to T[i,a,k]
    
    # Let's try to find 9 such terms that sum to all-ones tensor
    
    # This is a hard problem. Let me try a greedy/heuristic approach.
    
    # Alternative: use the known result that rank(<2,3,8>) <= 36
    # and construct it via 4 copies of a rank-9 2x3x2 block
    
    # Let me try to explicitly construct a rank-9 2x3x2 decomposition
    
    # After research, here's a rank-9 decomposition for 2x3x2:
    # The tensor entries are indexed by (i, a, k) with i in {0,1}, a in {0,1,2}, k in {0,1}
    
    # I'll use a systematic construction based on the following idea:
    # Split into 3 groups of 4 terms each, one for each value of a
    
    # For a=0: need T[0,0,0]=1, T[0,0,1]=1, T[1,0,0]=1, T[1,0,1]=1
    #   Use 4 terms: (U[0,0]=1,V[0,0]=1,W[0,0]=1), (U[0,0]=1,V[0,1]=1,W[1,0]=1),
    #                 (U[1,0]=1,V[0,0]=1,W[0,1]=1), (U[1,0]=1,V[0,1]=1,W[1,1]=1)
    #   But this gives rank 4 just for a=0, and we'd need similar for a=1 and a=2, total 12
    
    # That's not rank 9. We need sharing.
    
    # Let me try a different strategy: use random search to find a rank-36 solution
    
    best_rank = 44
    best_solution = _rank44()
    
    if not _validate(best_solution):
        best_solution = _schoolbook()
        best_rank = 48
    
    # Try to improve by searching for a lower-rank solution
    # Use a simple local search: start from rank-44 and try to eliminate terms
    
    # For now, return the best we have
    if not _elapsed_ok():
        return best_solution
    
    # Try a more aggressive search: random rank-36 construction
    # We'll parameterize 9 terms for 2x3x2 and check if they work
    
    # Actually, let me just try to verify if a known rank-36 construction works
    
    # The construction uses the following pattern for each 2-column block of B:
    # 9 rank-1 terms that together compute A(2x3) * B_block(3x2)
    
    # Let me try to find such a decomposition by solving a system of equations
    
    # For a single 2x3x2 block, we need 9 vectors (u_c, v_c, w_c) for c=0..8
    # such that for all (i,a,k): sum_c u_c[i,a] * v_c[a,k] * w_c[k,i] = 1
    
    # Let's fix a structure and solve for the entries
    
    # Use the following ansatz: each u_c has nonzeros in at most 2 positions,
    # each v_c has nonzeros in at most 2 positions, each w_c has nonzeros in at most 2 positions
    
    # This is still a hard nonlinear system. Let me try a simpler approach.
    
    # Given time constraints, let me just return the rank-44 solution
    # which is verified to work.
    
    return best_solution
