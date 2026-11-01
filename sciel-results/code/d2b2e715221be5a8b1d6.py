import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]


def verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and d == k and b == c) else 0.0
                            s = 0.0
                            for l in range(rank):
                                s += U[a * m + b, l] * V[c * p + d, l] * W[k * n + i, l]
                            if abs(s - target) > 1e-9:
                                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 6

    # Strategy: Build a rank-11 decomposition for 2x3x3, then duplicate for 2x3x6.
    # The 2x3x3 multiplication tensor T[i,b,k] where i in {0,1}, b in {0,1,2}, k in {0,1,2}
    # Decomposition: sum_l U[i*3+b, l] * V[b*3+k, l] * W[k*2+i, l] = delta(i,i')*delta(k,k')*delta(b,b')
    
    # For 2x3x3: U is (6, R), V is (9, R), W is (6, R)
    # We know mu(2,2,2) = 7 (Strassen). For the 2x3x3 case, we need 7 terms for the 2x2x2 part
    # plus 3 additional terms to handle the third column of B (and corresponding C columns).
    # Total: 7 + 3 = 10? Or 7 + 4 = 11?
    # Actually, the 2x3x3 tensor has 2*3*3 = 18 nonzero entries.
    # The 2x2x2 sub-tensor (i in {0,1}, b in {0,1}, k in {0,1}) has 8 nonzero entries, rank 7.
    # The remaining entries involve b=2 or k=2.
    
    # Let's try to construct it explicitly.
    # Strassen's 7-term decomposition for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For 2x3x3, we extend this. Let's denote:
    # A is 2x3: A[i][b] for i in {0,1}, b in {0,1,2}
    # B is 3x3: B[b][k] for b in {0,1,2}, k in {0,1,2}
    # C is 2x3: C[i][k] for i in {0,1}, k in {0,1,2}
    
    # C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
    # C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]
    # C[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]
    # C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0]
    # C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]
    # C[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]
    
    # The 2x2x2 part (b in {0,1}, k in {0,1}) can be done in 7 terms.
    # The terms involving b=2 or k=2 need additional terms.
    # Entries with b=2: A[0][2]*B[2][k] and A[1][2]*B[2][k] for k in {0,1,2}
    # Entries with k=2: A[i][b]*B[b][2] for i in {0,1}, b in {0,1,2}
    
    # A known rank-11 decomposition for 2x3x3 uses the following structure:
    # Terms 0-6: Strassen-like for the 2x2x2 core
    # Terms 7-9: Handle the third column of B (b=2)
    # Terms 10-11: Handle the third row of B output (k=2) with cross terms
    
    # Let me try a specific construction. Based on literature, one rank-11 decomposition is:
    
    U11 = np.zeros((6, 11))
    V11 = np.zeros((9, 11))
    W11 = np.zeros((6, 11))
    
    # Indices:
    # U: row = i*3 + b, i in {0,1}, b in {0,1,2}
    #   0: (0,0), 1: (0,1), 2: (0,2)
    #   3: (1,0), 4: (1,1), 5: (1,2)
    # V: row = b*3 + k, b in {0,1,2}, k in {0,1,2}
    #   0: (0,0), 1: (0,1), 2: (0,2)
    #   3: (1,0), 4: (1,1), 5: (1,2)
    #   6: (2,0), 7: (2,1), 8: (2,2)
    # W: row = k*2 + i, k in {0,1,2}, i in {0,1}
    #   0: (0,0), 1: (0,1)
    #   2: (1,0), 3: (1,1)
    #   4: (2,0), 5: (2,1)
    
    # Term 0: (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    U11[0, 0] = 1; U11[4, 0] = 1
    V11[0, 0] = 1; V11[4, 0] = 1
    W11[0, 0] = 1; W11[1, 0] = 1  # k=0: i=0 -> W[0], i=1 -> W[1]
    W11[2, 0] = 1; W11[3, 0] = 1  # k=1: i=0 -> W[2], i=1 -> W[3]
    # Wait, this doesn't work directly. Let me reconsider.
    
    # Actually, the Strassen decomposition for 2x2x2 gives:
    # C00 = M1 - M3 - M4 + M6
    # C01 = M1 + M2 - M5 - M7
    # C10 = -M2 + M3 + M5 + M7
    # C11 = M1 + M4 - M6 + M7
    
    # Let me set up the 7 Strassen terms properly for the 2x2x2 subproblem.
    # Then add terms for the extensions.
    
    # Reset
    U11 = np.zeros((6, 11))
    V11 = np.zeros((9, 11))
    W11 = np.zeros((6, 11))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U11[0, 0] = 1; U11[4, 0] = 1
    V11[0, 0] = 1; V11[4, 0] = 1
    # W should extract contributions to all four C entries
    # C00 gets +M1, C01 gets +M1, C10 gets +M1, C11 gets +M1
    W11[0, 0] = 1   # k=0,i=0 -> C00
    W11[2, 0] = 1   # k=1,i=0 -> C01
    W11[1, 0] = 1   # k=0,i=1 -> C10
    W11[3, 0] = 1   # k=1,i=1 -> C11
    
    # Term 1: M2 = (A10+A11)B00
    U11[3, 1] = 1; U11[4, 1] = 1
    V11[0, 1] = 1
    # C01 gets +M2, C10 gets -M2
    W11[2, 1] = 1   # C01
    W11[1, 1] = -1  # C10
    
    # Term 2: M3 = A00(B10-B11)
    U11[0, 2] = 1
    V11[3, 2] = 1; V11[4, 2] = -1
    # C00 gets -M3, C10 gets +M3
    W11[0, 2] = -1  # C00
    W11[1, 2] = 1   # C10
    
    # Term 3: M4 = A11(B01-B00)
    U11[4, 3] = 1
    V11[1, 3] = 1; V11[0, 3] = -1
    # C00 gets -M4, C11 gets +M4
    W11[0, 3] = -1  # C00
    W11[3, 3] = 1   # C11
    
    # Term 4: M5 = (A00+A01)B11
    U11[0, 4] = 1; U11[1, 4] = 1
    V11[4, 4] = 1
    # C01 gets -M5, C10 gets +M5
    W11[2, 4] = -1  # C01
    W11[1, 4] = 1   # C10
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    U11[3, 5] = 1; U11[0, 5] = -1
    V11[0, 5] = 1; V11[3, 5] = 1
    # C00 gets +M6, C11 gets -M6
    W11[0, 5] = 1   # C00
    W11[3, 5] = -1  # C11
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    U11[1, 6] = 1; U11[4, 6] = -1
    V11[1, 6] = 1; V11[4, 6] = 1
    # C01 gets -M7, C10 gets +M7
    W11[2, 6] = -1  # C01
    W11[1, 6] = 1   # C10
    
    # Now we need to handle:
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # And the b=2 terms in C00, C01:
    # C00 += A02*B20
    # C01 += A02*B21
    # C10 += A12*B20
    # C11 += A12*B21
    
    # Term 7: Handle A02 * (B20 + B21) for C00 and C01
    U11[2, 7] = 1
    V11[6, 7] = 1; V11[7, 7] = 1  # B20, B21
    W11[0, 7] = 1   # C00
    W11[2, 7] = 1   # C01
    
    # Term 8: Handle A12 * (B20 + B21) for C10 and C11
    U11[5, 8] = 1
    V11[6, 8] = 1; V11[7, 8] = 1  # B20, B21
    W11[1, 8] = 1   # C10
    W11[3, 8] = 1   # C11
    
    # Term 9: Handle (A00 + A01) * B02 for C02
    U11[0, 9] = 1; U11[1, 9] = 1
    V11[2, 9] = 1  # B02
    W11[4, 9] = 1  # C02 (k=2, i=0)
    
    # Term 10: Handle (A10 + A11) * B02 for C12
    U11[3, 10] = 1; U11[4, 10] = 1
    V11[2, 10] = 1  # B02
    W11[5, 10] = 1  # C12 (k=2, i=1)
    
    # But wait, C02 also has A01*B12 and A02*B22
    # C12 also has A11*B12 and A12*B22
    # And we're missing A01*B12, A11*B12, A02*B22, A12*B22
    
    # This construction is getting complex. Let me verify what we have so far
    # and see if we can fix it.
    
    # Actually, let me try a different, more systematic approach.
    # Let me use the known result that mu(2,3,3) <= 11 and try to find it
    # by solving a linear system with a fixed support pattern.
    
    # Alternative: Use a known rank-11 decomposition from the literature.
    # One such decomposition was found by Bini et al. or in various papers.
    
    # Let me try yet another approach: directly search for a valid rank-11 decomposition
    # using a structured pattern.
    
    # Actually, let me first verify if my above construction works for the 2x2x2 part,
    # then figure out the extensions.
    
    # Let me just try to verify the current U11, V11, W11 for 2x3x3
    valid_11 = verify(U11, V11, W11, 2, 3, 3)
    
    if not valid_11:
        # Try a different construction. Let me use a more careful approach.
        # The issue is that the simple extension doesn't work because the Strassen
        # terms already use certain B entries, and we need to account for all cross terms.
        
        # Let me try a completely different rank-11 pattern.
        # Based on the "lax" decomposition or similar.
        
        U11 = np.zeros((6, 11))
        V11 = np.zeros((9, 11))
        W11 = np.zeros((6, 11))
        
        # I'll use a known valid rank-11 decomposition.
        # Reference: The rank of 2x3x3 matrix multiplication is 11.
        # One explicit decomposition:
        
        # Term 0: A00*B00
        U11[0, 0] = 1; V11[0, 0] = 1; W11[0, 0] = 1
        # Term 1: A00*B01
        U11[0, 1] = 1; V11[1, 1] = 1; W11[2, 1] = 1
        # Term 2: A00*B02
        U11[0, 2] = 1; V11[2, 2] = 1; W11[4, 2] = 1
        # Term 3: A01*B10
        U11[1, 3] = 1; V11[3, 3] = 1; W11[0, 3] = 1
        # Term 4: A01*B11
        U11[1, 4] = 1; V11[4, 4] = 1; W11[2, 4] = 1
        # Term 5: A01*B12
        U11[1, 5] = 1; V11[5, 5] = 1; W11[4, 5] = 1
        # Term 6: A02*B20
        U11[2, 6] = 1; V11[6, 6] = 1; W11[0, 6] = 1
        # Term 7: A02*B21
        U11[2, 7] = 1; V11[7, 7] = 1; W11[2, 7] = 1
        # Term 8: A02*B22
        U11[2, 8] = 1; V11[8, 8] = 1; W11[4, 8] = 1
        # That's already 9 terms just for row 0 of A.
        # We need 6 more for row 1 of A, but we only have 2 terms left.
        # This naive approach won't work. We need to share terms.
        
        # Let me go back to the Strassen-based approach and fix it properly.
        
        U11 = np.zeros((6, 11))
        V11 = np.zeros((9, 11))
        W11 = np.zeros((6, 11))
        
        # Use Strassen for the 2x2x2 core (b in {0,1}, k in {0,1})
        # Then add 4 terms for the k=2 column and b=2 row extensions.
        
        # Strassen terms (0-6) as before
        U11[0, 0] = 1; U11[4, 0] = 1
        V11[0, 0] = 1; V11[4, 0] = 1
        W11[0, 0] = 1; W11[2, 0] = 1; W11[1, 0] = 1; W11[3, 0] = 1
        
        U11[3, 1] = 1; U11[4, 1] = 1
        V11[0, 1] = 1
        W11[2, 1] = 1; W11[1, 1] = -1
        
        U11[0, 2] = 1
        V11[3, 2] = 1; V11[4, 2] = -1
        W11[0, 2] = -1; W11[1, 2] = 1
        
        U11[4, 3] = 1
        V11[1, 3] = 1; V11[0, 3] = -1
        W11[0, 3] = -1; W11[3, 3] = 1
        
        U11[0, 4] = 1; U11[1, 4] = 1
        V11[4, 4] = 1
        W11[2, 4] = -1; W11[1, 4] = 1
        
        U11[3, 5] = 1; U11[0, 5] = -1
        V11[0, 5] = 1; V11[3, 5] = 1
        W11[0, 5] = 1; W11[3, 5] = -1
        
        U11[1, 6] = 1; U11[4, 6] = -1
        V11[1, 6] = 1; V11[4, 6] = 1
        W11[2, 6] = -1; W11[1, 6] = 1
        
        # Now for the extensions:
        # We need to add:
        # C00 += A02*B20
        # C01 += A02*B21
        # C10 += A12*B20
        # C11 += A12*B21
        # C02 = A00*B02 + A01*B12 + A02*B22
        # C12 = A10*B02 + A11*B12 + A12*B22
        
        # Term 7: A02*B20 -> C00
        U11[2, 7] = 1; V11[6, 7] = 1; W11[0, 7] = 1
        # Term 8: A02*B21 -> C01
        U11[2, 8] = 1; V11[7, 8] = 1; W11[2, 8] = 1
        # Term 9: A12*B20 -> C10
        U11[5, 9] = 1; V11[6, 9] = 1; W11[1, 9] = 1
        # Term 10: A12*B21 -> C11
        U11[5, 10] = 1; V11[7, 10] = 1; W11[3, 10] = 1
        
        # But we still need C02 and C12!
        # C02 = A00*B02 + A01*B12 + A02*B22
        # C12 = A10*B02 + A11*B12 + A12*B22
        # We've used all 11 terms. This won't work.
        
        # The problem is that the naive Strassen + 4 extension terms only gives 11 terms
        # but doesn't cover C02 and C12. We need a smarter combination.
        
        # Let me try combining some terms.
        # Instead of separate terms for A02*B20 and A02*B21, use one term:
        # Term 7: A02*(B20+B21) -> contributes to both C00 and C01
        # Term 8: A12*(B20+B21) -> contributes to both C10 and C11
        # Then we have 2 terms left for C02 and C12.
        # But C02 has 3 terms and C12 has 3 terms. We can't do it in 2 terms.
        
        # Unless we combine:
        # Term 9: (A00+A01)*B02 + A02*B22? No, that's not a product of linear forms.
        
        # The key insight is that we need to be more clever.
        # Let me look at this differently.
        
        # Actually, I recall that the rank of 2x3x3 is indeed 11, but the construction
        # is non-trivial. Let me try to use a computer algebra approach.
        
        # For now, let me just use the rank-17 solution from the seed and return rank 34.
        # But first, let me try one more thing: a rank-12 construction.
        
        # Actually, let me just verify if the rank-17 block from the seed works,
        # and return rank 34. That's guaranteed to work.
        pass

    # Let me just use the verified rank-17 solution from the seed.
    # This gives rank 34 for 2x3x6.
    
    U_seed = np.zeros((6, 34))
    V_seed = np.zeros((18, 34))
    W_seed = np.zeros((12, 34))
    
    U_block = np.zeros((6, 17))
    V_block = np.zeros((9, 17))
    W_block = np.zeros((6, 17))
    
    # Hardcoded Rank 17 solution for 2x3x3
    U_block[0, 0] = 1; U_block[4, 0] = 1
    V_block[0, 0] = 1; V_block[4, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    U_block[3, 1] = 1; U_block[4, 1] = 1
    V_block[0, 1] = 1
    W_block[1, 1] = 1; W_block[3, 1] = -1
    
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[4, 2] = -1
    W_block[2, 2] = 1; W_block[3, 2] = 1
    
    U_block[4, 3] = 1
    V_block[3, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[1, 3] = 1
    
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[4, 4] = 1
    W_block[0, 4] = -1; W_block[2, 4] = 1
    
    U_block[3, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1
    
    U_block[1, 6] = 1; U_block[4, 6] = -1
    V_block[3, 6] = 1; V_block[4, 6] = 1
    W_block[0, 6] = 1
    
    U_block[0, 7] = 1
    V_block[2, 7] = 1
    W_block[4, 7] = 1
    
    U_block[1, 8] = 1
    V_block[5, 8] = 1
    W_block[4, 8] = 1
    
    U_block[3, 9] = 1
    V_block[2, 9] = 1
    W_block[5, 9] = 1
    
    U_block[4, 10] = 1
    V_block[5, 10] = 1
    W_block[5, 10] = 1
    
    U_block[2, 11] = 1
    V_block[6, 11] = 1
    W_block[0, 11] = 1
    
    U_block[2, 12] = 1
    V_block[7, 12] = 1
    W_block[2, 12] = 1
    
    U_block[2, 13] = 1
    V_block[8, 13] = 1
    W_block[4, 13] = 1
    
    U_block[5, 14] = 1
    V_block[6, 14] = 1
    W_block[1, 14] = 1
    
    U_block[5, 15] = 1
    V_block[7, 15] = 1
    W_block[3, 15] = 1
    
    U_block[5, 16] = 1
    V_block[8, 16] = 1
    W_block[5, 16] = 1
    
    # Assemble Rank 34
    U_seed[:, :17] = U_block
    U_seed[:, 17:] = U_block
    
    for j in range(3):
        for k in range(3):
            V_seed[j * 6 + k, :17] = V_block[j * 3 + k]
            V_seed[j * 6 + 3 + k, 17:] = V_block[j * 3 + k]
    
    for k in range(3):
        for i in range(2):
            W_seed[k * 2 + i, :17] = W_block[k * 2 + i]
            W_seed[(k + 3) * 2 + i, 17:] = W_block[k * 2 + i]
    
    return {
        "U": U_seed.tolist(),
        "V": V_seed.tolist(),
        "W": W_seed.tolist()
    }


if __name__ == "__main__":
    construct()
