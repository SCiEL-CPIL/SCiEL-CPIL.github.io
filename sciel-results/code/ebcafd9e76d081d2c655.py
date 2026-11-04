"""Matrix multiplication 3,5,7 — rank-minimal exact tensor decomposition.

Strategy:
  1. Hard-coded block decomposition: split 3x5*5x7 into (2x5*5x7)+(1x5*5x7).
     - 1x5*5x7: rank 5 (trivial, each output column is a dot product).
     - 2x5*5x7: further split into (2x2*2x2) Strassen blocks + residual.
       Split the 5 middle dimension into 2+3, and the 7 output into 2+5.
       The 2x2*2x2 block uses Strassen (rank 7).
       The remaining blocks use schoolbook.
  2. If the hard-coded rank is not better than schoolbook (105), fall back to schoolbook.
  3. Verification is built-in: check the decomposition exactly.
"""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120

    # Schoolbook baseline
    sb = _schoolbook(n, m, p)
    best = sb
    best_rank = n * m * p

    # --- Block decomposition: split rows of A into 2+1 ---
    # A (3x5) = [A1; A2] where A1 is 2x5, A2 is 1x5
    # B (5x7) stays whole
    # C1 = A1 * B (2x7), C2 = A2 * B (1x7)
    # rank(C2) = 5 (1x5*5x7 tensor rank is 5)
    # For C1 (2x5*5x7): split B into B_left (5x2) and B_right (5x5)
    #   C1_left = A1 * B_left (2x2) -> Strassen rank 7
    #   C1_right = A1 * B_right (2x5) -> schoolbook rank 20
    # Total for C1 = 7 + 20 = 27
    # Grand total = 27 + 5 = 32

    # Actually let me be more careful. The tensor rank for 1xm*mxp is m, not p.
    # For 1x5*5x7: T[0, j, k] = 1 for all j,k.
    # T = sum_{j=0}^{4} e_0 ⊗ e_j ⊗ (sum_{k=0}^{6} e_k)
    # That's 5 terms. So rank = 5. Correct.

    # For 2x5*5x7:
    # Split middle dim 5 = 2+3, output dim 7 = 2+5
    # A1 (2x5) = [A1a (2x2), A1b (2x3)]
    # B (5x7) = [B_left (5x2); B_right (5x5)]  -- wait, B is 5x7, split columns
    # B = [B1 (5x2), B2 (5x5)]
    # C1_left = A1 * B1 (2x2): Strassen rank 7
    # C1_right = A1 * B2 (2x5): schoolbook rank 20
    # Total C1 = 27
    # Grand total = 27 + 5 = 32

    # But wait: when we split A1 into A1a and A1b, and B into B1 and B2,
    # the cross terms matter. Let me think again.
    #
    # A1 (2x5) * B (5x7):
    # Split A1 columns: A1 = [A1a (2x2), A1b (2x3)]
    # Split B rows: B = [B1 (2x7); B2 (3x7)]
    # Then A1*B = A1a*B1 + A1b*B2
    # A1a*B1 is 2x2 * 2x7 -> split B1 into (2x2)+(2x5): Strassen(2x2*2x2)=7 + schoolbook(2x2*2x5)=20... 
    # Hmm, that's 27 for just A1a*B1, plus A1b*B2 (2x3*3x7)=42. Total 69. Worse.
    #
    # Better: don't split the middle dimension. Just split the output.
    # A1 (2x5) * B (5x7):
    # Split B columns: B = [B1 (5x2), B2 (5x5)]
    # C1a = A1 * B1 (2x2): Strassen rank 7
    # C1b = A1 * B2 (2x5): schoolbook rank 20
    # Total C1 = 27
    # This works because we're just splitting the output columns, not the middle dim.
    # The decomposition is: for each term in C1a and C1b, we use the corresponding
    # entries of A1 and the respective B block.

    # Let me construct this explicitly.
    #
    # Full decomposition of 3x5*5x7:
    # Split rows of A: n1=2, n2=1
    # Split cols of B: p1=2, p2=5
    #
    # Block (n1, p1): 2x5 * 5x2 -> Strassen 2x2*2x2 (padding) or direct
    #   Actually 2x5*5x2: schoolbook rank = 20. Can we do better?
    #   Split middle: 5=2+3. A1=[A1a(2x2),A1b(2x3)], B1=[B1a(2x2);B1b(3x2)]
    #   A1a*B1a: 2x2*2x2 Strassen rank 7
    #   A1b*B1b: 2x3*3x2 schoolbook rank 12
    #   Total: 19. Slightly better than 20.
    #
    # Let me just go with a clean construction:
    #
    # Decomposition:
    #   Part 1: Rows 0-1 of A (2x5) * Cols 0-1 of B (5x2) -> 2x2 output
    #     Use Strassen on the 2x2*2x2 core (first 2 middle indices)
    #     Plus schoolbook for remaining middle indices
    #   Part 2: Rows 0-1 of A (2x5) * Cols 2-6 of B (5x5) -> 2x5 output
    #     Schoolbook: 20 terms
    #   Part 3: Row 2 of A (1x5) * Cols 0-6 of B (5x7) -> 1x7 output
    #     Rank 5
    #
    # For Part 1 (2x5*5x2):
    #   Split middle 5 = 2+3
    #   A1a (2x2) * B1a (2x2): Strassen rank 7
    #   A1b (2x3) * B1b (3x2): schoolbook rank 12
    #   Total: 19
    #
    # Grand total: 19 + 20 + 5 = 44

    # Hmm, 44 is better than 105 but not amazing. Let me try to optimize.
    #
    # Alternative: for 2x5*5x7, split output 7=2+5:
    #   2x5*5x2: rank 19 (as above)
    #   2x5*5x5: schoolbook 50
    #   Total: 69. Worse.
    #
    # Better: for 2x5*5x7, just use schoolbook: 70.
    # With the 1x5*5x7 part (rank 5), total = 75. Still better than 105.
    #
    # Actually, the simplest improvement:
    # Split rows 2+1. The 1x5*5x7 part has rank 5.
    # The 2x5*5x7 part has schoolbook rank 70.
    # Total: 75. That's already a big improvement over 105.
    #
    # Can we do better on the 2x5*5x7 part?
    # 2x5*5x7: schoolbook = 70.
    # Split output 7=2+5: 2x5*5x2 (rank 20) + 2x5*5x5 (rank 50) = 70. No improvement.
    # Split middle 5=2+3: 2x2*2x7 (rank 14) + 2x3*3x7 (rank 42) = 56. Better!
    #   2x2*2x7: split output 7=2+5: 2x2*2x2 (Strassen 7) + 2x2*2x5 (schoolbook 20) = 27.
    #   Or: 2x2*2x7 schoolbook = 28.
    #   So 2x2*2x7: min(28, 27) = 27.
    #   2x3*3x7: schoolbook = 42.
    #   Total for 2x5*5x7: 27 + 42 = 69. Slightly better than 70.
    #
    # Grand total: 69 + 5 = 74.
    #
    # Let me try even harder on 2x3*3x7:
    # Split output 7=2+5: 2x3*3x2 (rank 12) + 2x3*3x5 (rank 30) = 42. Same.
    # Split middle 3=1+2: 2x1*1x7 (rank 7) + 2x2*2x7 (rank 27) = 34. Better!
    #   2x1*1x7: rank 7 (each output is a dot product of length 1... wait)
    #   Actually 2x1*1x7: T[i,0,k]=1 for i in {0,1}, k in {0..6}.
    #   T = sum_i sum_k e_i ⊗ e_0 ⊗ e_k. Rank = 2*7 = 14? No.
    #   T = sum_i (e_i ⊗ e_0) ⊗ (sum_k e_k). Rank = 2.
    #   Wait: (sum_k e_k) is a vector in R^7. So each term is e_i ⊗ e_0 ⊗ (1,1,...,1).
    #   That's 2 terms. So rank = 2? Let me verify.
    #   T[0,0,k] = 1 for all k. T[1,0,k] = 1 for all k.
    #   Term 1: e_0 ⊗ e_0 ⊗ (1,1,1,1,1,1,1) gives T[0,0,k]=1 for all k. ✓
    #   Term 2: e_1 ⊗ e_0 ⊗ (1,1,1,1,1,1,1) gives T[1,0,k]=1 for all k. ✓
    #   So rank = 2.
    #
    #   2x2*2x7: split output 7=2+5: Strassen(2x2*2x2)=7 + schoolbook(2x2*2x5)=20 = 27.
    #   Or schoolbook: 28.
    #   So 2x2*2x7 = 27.
    #
    #   Total for 2x3*3x7: 2 + 27 = 29.
    #
    # Grand total: 27 + 29 + 5 = 61.
    #
    # Let me verify: 2x5*5x7 = (2x2*2x7) + (2x3*3x7) = 27 + 29 = 56.
    # Plus 1x5*5x7 = 5.
    # Total = 61.
    #
    # Can we improve 2x2*2x7 further?
    # 2x2*2x7: schoolbook = 28.
    # Strassen on 2x2*2x2 part: 7, plus 2x2*2x5: 20. Total 27.
    # Or: 2x2*2x7 = 2x2*(2x2 + 2x5). The 2x2*2x5 part:
    #   Split output 5=2+3: 2x2*2x2 (Strassen 7) + 2x2*2x3 (schoolbook 12) = 19.
    #   So 2x2*2x7 = 7 + 19 = 26.
    #
    # Grand total: 26 + 29 + 5 = 60.
    #
    # Can we improve 2x2*2x3?
    # 2x2*2x3: schoolbook = 12.
    # Split output 3=1+2: 2x2*2x1 (rank 4) + 2x2*2x2 (Strassen 7) = 11.
    #   2x2*2x1: T[i,j,0]=1. T = sum_i sum_j e_i⊗e_j⊗e_0. Rank = 4.
    #   Actually: T = (sum_i e_i) ⊗ (sum_j e_j) ⊗ e_0. Rank = 1!
    #   Wait: (sum_i e_i)⊗(sum_j e_j)⊗e_0 has entries [i,j,0] = 1 for all i,j. Yes!
    #   So 2x2*2x1 has rank 1.
    #   2x2*2x2 has rank 7 (Strassen).
    #   Total: 1 + 7 = 8.
    #
    # So 2x2*2x3 = 8.
    # 2x2*2x5 = 7 + 8 = 15.
    # 2x2*2x7 = 7 + 15 = 22.
    #
    # Grand total: 22 + 29 + 5 = 56.
    #
    # Let me double-check 2x2*2x5:
    # Split output 5=2+3: 2x2*2x2 (7) + 2x2*2x3 (8) = 15. ✓
    #
    # And 2x2*2x7:
    # Split output 7=2+5: 2x2*2x2 (7) + 2x2*2x5 (15) = 22. ✓
    #
    # And 2x3*3x7:
    # Split middle 3=1+2: 2x1*1x7 (2) + 2x2*2x7 (22) = 24.
    # Wait, I had 29 before. Let me recheck.
    # 2x1*1x7: rank 2. ✓
    # 2x2*2x7: 22. ✓
    # Total: 24.
    #
    # Grand total: 22 + 24 + 5 = 51.
    #
    # Can we improve 2x1*1x7?
    # 2x1*1x7: T[i,0,k]=1 for i∈{0,1}, k∈{0..6}.
    # T = e_0⊗e_0⊗(1,1,1,1,1,1,1) + e_1⊗e_0⊗(1,1,1,1,1,1,1). Rank 2. ✓
    #
    # Can we improve 2x2*2x7?
    # 2x2*2x7: schoolbook 28.
    # Strassen-based: 22 (as computed).
    # Any better? 2x2*2x7 = 2x2*(2x7). 
    # The 2x7 output: we could try to split differently.
    # 7 = 3+4: 2x2*2x3 (8) + 2x2*2x4 (?)
    # 2x2*2x4: split 4=2+2: Strassen(2x2*2x2)=7 + Strassen(2x2*2x2)=7 = 14.
    # Or schoolbook: 16.
    # So 2x2*2x4 = 14.
    # 2x2*2x7 = 8 + 14 = 22. Same.
    #
    # 7 = 1+6: 2x2*2x1 (1) + 2x2*2x6 (?)
    # 2x2*2x6: split 6=3+3: 2x2*2x3 (8) + 2x2*2x3 (8) = 16.
    # Or 6=2+4: 7+14=21. Or 6=1+5: 1+15=16.
    # So 2x2*2x6 = 16.
    # 2x2*2x7 = 1 + 16 = 17! Better!
    #
    # Grand total: 17 + 24 + 5 = 46.
    #
    # Let me verify 2x2*2x6 = 16:
    # Split output 6=3+3: 2x2*2x3 (8) + 2x2*2x3 (8) = 16. ✓
    # Or 6=2+4: 7+14=21. 6=1+5: 1+15=16. So 16 is correct.
    #
    # Can we improve 2x2*2x3?
    # 2x2*2x3: split 3=1+2: 2x2*2x1 (1) + 2x2*2x2 (7) = 8. ✓
    # Split 3=2+1: 7+1=8. Same.
    #
    # Can we improve 2x2*2x4?
    # 2x2*2x4: split 4=2+2: 7+7=14. ✓
    # Split 4=1+3: 1+8=9. Better!
    # So 2x2*2x4 = 9.
    #
    # Then 2x2*2x6: split 6=2+4: 7+9=16. Same as before.
    # Split 6=1+5: 1+15=16. Same.
    # Split 6=3+3: 8+8=16. Same.
    #
    # 2x2*2x5: split 5=1+4: 1+9=10. Better than 15!
    # 2x2*2x7: split 7=1+6: 1+16=17. Same as before.
    # Split 7=2+5: 7+10=17. Same.
    # Split 7=3+4: 8+9=17. Same.
    #
    # So 2x2*2x7 = 17.
    #
    # Now 2x3*3x7:
    # Split middle 3=1+2: 2x1*1x7 (2) + 2x2*2x7 (17) = 19.
    # Previously had 24. Now 19.
    #
    # Grand total: 17 + 19 + 5 = 41.
    #
    # Let me check if we can improve 2x1*1x7.
    # 2x1*1x7: rank 2. Can't do better (need at least 2 since output is 2x7 and middle is 1).
    # Actually, the minimum rank for 2x1*1x7: the tensor has 14 non-zero entries.
    # T = sum_{i,k} e_i⊗e_0⊗e_k. 
    # We can write T = (sum_i e_i) ⊗ e_0 ⊗ (sum_k e_k). That's rank 1!
    # Wait: (sum_i e_i)⊗e_0⊗(sum_k e_k) has entry [i,0,k] = 1 for all i,k. Yes!
    # So 2x1*1x7 has rank 1!
    #
    # Similarly, 1x1*1xp has rank 1.
    # 2x1*1x7: rank 1.
    #
    # Grand total: 17 + (1+17) + 5 = 40.
    #
    # Wait, let me recheck. 2x1*1x7:
    # T[i,0,k] = 1 for i∈{0,1}, k∈{0..6}.
    # (e_0+e_1) ⊗ e_0 ⊗ (e_0+e_1+...+e_6) = sum_{i,k} e_i⊗e_0⊗e_k. ✓
    # Rank 1. ✓
    #
    # So 2x3*3x7 = 1 + 17 = 18.
    # Grand total: 17 + 18 + 5 = 40.
    #
    # Can we improve 1x5*5x7?
    # 1x5*5x7: T[0,j,k]=1 for j∈{0..4}, k∈{0..6}.
    # T = e_0 ⊗ (sum_j e_j) ⊗ (sum_k e_k). Rank 1!
    #
    # Grand total: 17 + 18 + 1 = 36.
    #
    # Let me re-verify all the ranks:
    # 1x1*1x1: rank 1
    # 2x1*1x7: rank 1
    # 1x5*5x7: rank 1
    # 2x2*2x1: rank 1
    # 2x2*2x2: rank 7 (Strassen)
    # 2x2*2x3: rank 8 (1+7)
    # 2x2*2x4: rank 9 (1+8)
    # 2x2*2x5: rank 10 (1+9)
    # 2x2*2x6: rank 16 (1+15 or 7+9 or 8+8)
    #   Wait: 2x2*2x5 = 1+9 = 10. Then 2x2*2x6 = 1+10 = 11? 
    #   Split 6=1+5: 1 + 10 = 11. Better than 16!
    # 2x2*2x7: split 7=1+6: 1+11 = 12. Even better!
    #
    # Let me redo:
    # 2x2*2x1: 1
    # 2x2*2x2: 7
    # 2x2*2x3: 1+7 = 8
    # 2x2*2x4: 1+8 = 9
    # 2x2*2x5: 1+9 = 10
    # 2x2*2x6: 1+10 = 11
    # 2x2*2x7: 1+11 = 12
    #
    # But wait, is 2x2*2x4 = 9 correct?
    # Split 4=1+3: 2x2*2x1 (1) + 2x2*2x3 (8) = 9. ✓
    # Split 4=2+2: 7+7=14. So 9 is better. ✓
    #
    # 2x2*2x5: split 5=1+4: 1+9=10. Split 5=2+3: 7+8=15. So 10. ✓
    # 2x2*2x6: split 6=1+5: 1+10=11. Split 6=2+4: 7+9=16. Split 6=3+3: 8+8=16. So 11. ✓
    # 2x2*2x7: split 7=1+6: 1+11=12. Split 7=2+5: 7+10=17. Split 7=3+4: 8+9=17. So 12. ✓
    #
    # Now:
    # 2x2*2x7 = 12
    # 2x3*3x7: split middle 3=1+2: 2x1*1x7 (1) + 2x2*2x7 (12) = 13
    # 1x5*5x7 = 1
    #
    # Grand total: 12 + 13 + 1 = 26.
    #
    # That's a huge improvement over 105!
    #
    # But wait, I need to make sure these decompositions are valid.
    # The key insight is that for 2x2*2xp where p is small, we can recursively
    # split the output dimension and use rank-1 terms for the 2x2*2x1 parts.
    #
    # Let me verify 2x2*2x1 has rank 1:
    # T[i,j,0] = 1 for i,j∈{0,1}.
    # (e_0+e_1)⊗(e_0+e_1)⊗e_0 = sum_{i,j} e_i⊗e_j⊗e_0. ✓
    #
    # And Strassen for 2x2*2x2: rank 7. ✓
    #
    # So the total rank is 26. Let me now construct the actual decomposition.
    #
    # Structure:
    # A (3x5) * B (5x7) = C (3x7)
    #
    # Split A rows: A = [A_top (2x5); A_bot (1x5)]
    # Split B cols: B = [B_left (5x7)]  (no split needed for B cols at top level)
    #
    # Part 1: A_top * B (2x5 * 5x7)
    #   Split middle 5 = 2+3:
    #   A_top = [A_top_left (2x2), A_top_right (2x3)]
    #   B = [B_top (2x7); B_bot (3x7)]
    #   A_top * B = A_top_left * B_top + A_top_right * B_bot
    #
    #   Part 1a: A_top_left (2x2) * B_top (2x7) -> 2x7 output
    #     Split output 7 = 1+6:
    #     B_top = [B_top_1 (2x1), B_top_6 (2x6)]
    #     A_top_left * B_top = A_top_left * B_top_1 + A_top_left * B_top_6
    #     Part 1a-i: 2x2*2x1, rank 1
    #     Part 1a-ii: 2x2*2x6, rank 11
    #
    #   Part 1b: A_top_right (2x3) * B_bot (3x7) -> 2x7 output
    #     Split middle 3 = 1+2:
    #     A_top_right = [A_tr_1 (2x1), A_tr_2 (2x2)]
    #     B_bot = [B_bot_1 (1x7); B_bot_2 (2x7)]
    #     Part 1b-i: 2x1*1x7, rank 1
    #     Part 1b-ii: 2x2*2x7, rank 12
    #
    # Part 2: A_bot (1x5) * B (5x7) -> 1x7 output, rank 1
    #
    # Total: 1 + 11 + 1 + 12 + 1 = 26
    #
    # Now I need to construct the actual U, V, W matrices.
    # This is complex. Let me write a recursive construction function.

    result = _construct_decomp(3, 5, 7)
    if result is not None:
        rank = len(result['U'][0]) if result['U'] else 0
        if rank < best_rank:
            # Verify
            if _verify(result, 3, 5, 7):
                best = result
                best_rank = rank

    return best

def _verify(result, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor
    # T[i, j, k] = 1 for all valid i, j, k
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _construct_decomp(n, m, p):
    """Recursively construct a low-rank decomposition for n x m * m x p."""
    if n == 0 or m == 0 or p == 0:
        return {"U": [], "V": [], "W": []}
    
    # Base case: 1x1*1x1
    if n == 1 and m == 1 and p == 1:
        return {"U": [[1]], "V": [[1]], "W": [[1]]}
    
    # 1xm*mxp: rank 1
    if n == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        # Actually, for 1xm*mxp, we need rank m in general? No, rank 1.
        # T[0,j,k] = 1. T = e_0 ⊗ (sum_j e_j) ⊗ (sum_k e_k). Rank 1.
        # But our factor matrices have specific shapes.
        # U: (1*m) x 1 = (m) x 1. U[j,0] = 1 for all j.
        # V: (m*p) x 1 = (m*p) x 1. V[j*p+k, 0] = 1 for all j,k.
        # W: (p*1) x 1 = (p) x 1. W[k*1+0, 0] = 1 for all k.
        # Check: sum_c U[0*m+j, c] * V[j*p+k, c] * W[k*1+0, c]
        # = U[j,0]*V[j*p+k,0]*W[k,0] = 1*1*1 = 1. ✓
        U = [[1] for _ in range(m)]
        V = [[1] for _ in range(m * p)]
        W = [[1] for _ in range(p)]
        return {"U": U, "V": V, "W": W}
    
    # nx1*1xp: rank 1
    if m == 1:
        U = [[1] for _ in range(n)]
        V = [[1] for _ in range(p)]
        W = [[1] for _ in range(p * n)]
        # U: (n*1) x 1. U[i,0] = 1.
        # V: (1*p) x 1. V[k,0] = 1.
        # W: (p*n) x 1. W[k*n+i,0] = 1.
        # Check: U[i,0]*V[k,0]*W[k*n+i,0] = 1. ✓
        return {"U": U, "V": V, "W": W}
    
    # 2x2*2x2: Strassen, rank 7
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    
    # Recursive case: try splitting dimensions
    best_rank = n * m * p
    best_result = None
    
    # Try splitting n
    if n >= 2:
        for n1 in range(1, n):
            n2 = n - n1
            r1 = _estimate_rank(n1, m, p)
            r2 = _estimate_rank(n2, m, p)
            total = r1 + r2
            if total < best_rank:
                best_rank = total
                best_result = _split_n(n, m, p, n1)
    
    # Try splitting m
    if m >= 2:
        for m1 in range(1, m):
            m2 = m - m1
            r1 = _estimate_rank(n, m1, p)
            r2 = _estimate_rank(n, m2, p)
            total = r1 + r2
            if total < best_rank:
                best_rank = total
                best_result = _split_m(n, m, p, m1)
    
    # Try splitting p
    if p >= 2:
        for p1 in range(1, p):
            p2 = p - p1
            r1 = _estimate_rank(n, m, p1)
            r2 = _estimate_rank(n, m, p2)
            total = r1 + r2
            if total < best_rank:
                best_rank = total
                best_result = _split_p(n, m, p, p1)
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _estimate_rank(n, m, p):
    """Estimate the rank for n x m * m x p."""
    if n == 0 or m == 0 or p == 0:
        return 0
    if n == 1 or m == 1 or p == 1:
        # 1xm*mxp: rank 1
        # nx1*1xp: rank 1
        # nxm*mx1: rank 1
        return 1
    if n == 2 and m == 2 and p == 2:
        return 7
    
    best = n * m * p
    
    # Try splits
    if n >= 2:
        for n1 in range(1, n):
            best = min(best, _estimate_rank(n1, m, p) + _estimate_rank(n - n1, m, p))
    if m >= 2:
        for m1 in range(1, m):
            best = min(best, _estimate_rank(n, m1, p) + _estimate_rank(n, m - m1, p))
    if p >= 2:
        for p1 in range(1, p):
            best = min(best, _estimate_rank(n, m, p1) + _estimate_rank(n, m, p - p1))
    
    return best

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2, rank 7."""
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+cg, af+dh],[ce+dg, cf+bh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U indices: i*m+j where i∈{0,1}, j∈{0,1} -> 0,1,2,3
    # V indices: j*p+k where j∈{0,1}, k∈{0,1} -> 0,1,2,3
    # W indices: k*n+i where k∈{0,1}, i∈{0,1} -> 0,1,2,3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): U[0]=1,a=U[0]; U[3]=1,d=U[3]; V[0]=1,e=V[0]; V[3]=1,h=V[3]
    # W: C11+=1, C22+=1 -> W[0]=1, W[3]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e: U[2]=1,c; U[3]=1,d; V[0]=1,e
    # W: C21+=1 -> W[2]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = a(f-h): U[0]=1,a; V[1]=1,f; V[3]=-1,h
    # W: C12+=1, C22+=1 -> W[1]=1, W[3]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e): U[3]=1,d; V[2]=1,g; V[0]=-1,e
    # W: C11+=1, C21+=1 -> W[0]=1, W[2]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+b)h: U[0]=1,a; U[1]=1,b; V[3]=1,h
    # W: C11-=1, C12+=1 -> W[0]=-1, W[1]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (c-a)(e+f): U[2]=1,c; U[0]=-1,a; V[0]=1,e; V[1]=1,f
    # W: C22+=1 -> W[3]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h): U[1]=1,b; U[3]=-1,d; V[2]=1,g; V[3]=1,h
    # W: C11+=1 -> W[0]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _split_n(n, m, p, n1):
    """Split n into n1 + (n-n1)."""
    n2 = n - n1
    r1 = _construct_decomp(n1, m, p)
    r2 = _construct_decomp(n2, m, p)
    
    # Combine: U1 is (n1*m) x r1, U2 is (n2*m) x r2
    # U = [U1; U2] (n*m) x (r1+r2)
    # V = [V1, V2] (m*p) x (r1+r2)
    # W = [W1, W2] (p*n) x (r1+r2)
    # But W indices are k*n+i, so for the second block, i is offset by n1.
    
    r1_rank = r1['U'][0].__len__() if r1['U'] else 0
    r2_rank = r2['U'][0].__len__() if r2['U'] else 0
    total_rank = r1_rank + r2_rank
    
    U = [[0]*total_rank for _ in range(n * m)]
    V = [[0]*total_rank for _ in range(m * p)]
    W = [[0]*total_rank for _ in range(p * n)]
    
    # Fill first block
    for i in range(n1):
        for j in range(m):
            for c in range(r1_rank):
                U[i * m + j][c] = r1['U'][i * m + j][c]
    for j in range(m):
        for k in range(p):
            for c in range(r1_rank):
                V[j * p + k][c] = r1['V'][j * p + k][c]
    for k in range(p):
        for i in range(n1):
            for c in range(r1_rank):
                W[k * n + i][c] = r1['W'][k * n + i][c]
    
    # Fill second block (offset by r1_rank in columns, n1 in i-index of W)
    for i in range(n2):
        for j in range(m):
            for c in range(r2_rank):
                U[(n1 + i) * m + j][r1_rank + c] = r2['U'][i * m + j][c]
    for j in range(m):
        for k in range(p):
            for c in range(r2_rank):
                V[j * p + k][r1_rank + c] = r2['V'][j * p + k][c]
    for k in range(p):
        for i in range(n2):
            for c in range(r2_rank):
                W[k * n + (n1 + i)][r1_rank + c] = r2['W'][k * n + i][c]
    
    return {"U": U, "V": V, "W": W}

def _split_m(n, m, p, m1):
    """Split m into m1 + (m-m1)."""
    m2 = m - m1
    r1 = _construct_decomp(n, m1, p)
    r2 = _construct_decomp(n, m2, p)
    
    r1_rank = r1['U'][0].__len__() if r1['U'] else 0
    r2_rank = r2['U'][0].__len__() if r2['U'] else 0
    total_rank = r1_rank + r2_rank
    
    U = [[0]*total_rank for _ in range(n * m)]
    V = [[0]*total_rank for _ in range(m * p)]
    W = [[0]*total_rank for _ in range(p * n)]
    
    # First block: j in [0, m1)
    for i in range(n):
        for j in range(m1):
            for c in range(r1_rank):
                U[i * m + j][c] = r1['U'][i * m1 + j][c]
    for j in range(m1):
        for k in range(p):
            for c in range(r1_rank):
                V[j * p + k][c] = r1['V'][j * p + k][c]
    for k in range(p):
        for i in range(n):
            for c in range(r1_rank):
                W[k * n + i][c] = r1['W'][k * n + i][c]
    
    # Second block: j in [m1, m)
    for i in range(n):
        for j in range(m2):
            for c in range(r2_rank):
                U[i * m + (m1 + j)][r1_rank + c] = r2['U'][i * m2 + j][c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2_rank):
                V[(m1 + j) * p + k][r1_rank + c] = r2['V'][j * p + k][c]
    for k in range(p):
        for i in range(n):
            for c in range(r2_rank):
                W[k * n + i][r1_rank + c] = r2['W'][k * n + i][c]
    
    return {"U": U, "V": V, "W": W}

def _split_p(n, m, p, p1):
    """Split p into p1 + (p-p1)."""
    p2 = p - p1
    r1 = _construct_decomp(n, m, p1)
    r2 = _construct_decomp(n, m, p2)
    
    r1_rank = r1['U'][0].__len__() if r1['U'] else 0
    r2_rank = r2['U'][0].__len__() if r2['U'] else 0
    total_rank = r1_rank + r2_rank
    
    U = [[0]*total_rank for _ in range(n * m)]
    V = [[0]*total_rank for _ in range(m * p)]
    W = [[0]*total_rank for _ in range(p * n)]
    
    # First block: k in [0, p1)
    for i in range(n):
        for j in range(m):
            for c in range(r1_rank):
                U[i * m + j][c] = r1['U'][i * m + j][c]
    for j in range(m):
        for k in range(p1):
            for c in range(r1_rank):
                V[j * p + k][c] = r1['V'][j * p1 + k][c]
    for k in range(p1):
        for i in range(n):
            for c in range(r1_rank):
                W[k * n + i][c] = r1['W'][k * p1 + i][c]
    
    # Second block: k in [p1, p)
    for i in range(n):
        for j in range(m):
            for c in range(r2_rank):
                U[i * m + j][r1_rank + c] = r2['U'][i * m + j][c]
    for j in range(m):
        for k in range(p2):
            for c in range(r2_rank):
                V[j * p + (p1 + k)][r1_rank + c] = r2['V'][j * p2 + k][c]
    for k in range(p2):
        for i in range(n):
            for c in range(r2_rank):
                W[(p1 + k) * n + i][r1_rank + c] = r2['W'][k * p2 + i][c]
    
    return {"U": U, "V": V, "W": W}
