import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]


def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    start_time = time.time()
    deadline = 130  # safety margin before 150s kill

    # Target tensor
    T_target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1

    # Schoolbook rank 18 as baseline
    rank_sb = n * m * p  # 18
    U_sb = np.zeros((nm, rank_sb), dtype=int)
    V_sb = np.zeros((mp, rank_sb), dtype=int)
    W_sb = np.zeros((pn, rank_sb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_lists(mat):
        return mat.tolist()

    def verify_int(U, V, W, T):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
        r = U.shape[1]
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    total = 0
                    for rr in range(r):
                        total += U[a, rr] * V[b, rr] * W[c, rr]
                    if total != T[a, b, c]:
                        return False
        return True

    # Try to find rank-17 via systematic search with small integer values
    # Values allowed: -2, -1, 0, 1, 2 (covers Strassen-style coefficients)
    # Use a smarter approach: fix the structure based on known Strassen for 2x2x2
    # and search for the remaining terms.

    # Strategy: 
    # The 2x2x2 sub-tensor (i in {0,1}, j in {0,1}, k in {0,1}) can be done in rank 7 (Strassen)
    # The remaining terms involve j=2 or k=2 or both.
    # Total (i,j,k) triples: 18. 
    # Strassen handles 8 triples (2*2*2) in 7 terms.
    # Remaining: 10 triples. If we can do them in 10 terms, total = 17.
    # But some Strassen terms might overlap with the "remaining" structure.
    
    # Actually, let's try a different decomposition:
    # Split by j: 
    #   j=0: A[:,0] * B[0,:] -> 2x3 terms (i,k pairs) = 6 terms
    #   j=1: A[:,1] * B[1,:] -> 2x3 terms = 6 terms  
    #   j=2: A[:,2] * B[2,:] -> 2x3 terms = 6 terms
    # Total schoolbook: 18
    
    # Alternative: For each j, we're doing outer product of A[:,j] (2x1) and B[j,:] (1x3)
    # which gives a 2x3 matrix. Rank of that "slice" is at most 6 (schoolbook) but
    # could be less if we combine across j's.
    
    # Let me try: use rank 7 for the 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1})
    # and handle the rest with minimal additional terms.
    
    # The 8 (i,j,k) triples with j in {0,1}, k in {0,1}:
    # (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # Strassen does these in 7 multiplications.
    
    # The remaining 10 triples:
    # j=0,k=2: (0,0,2),(1,0,2)
    # j=1,k=2: (0,1,2),(1,1,2)
    # j=2,k=0: (0,2,0),(1,2,0)
    # j=2,k=1: (0,2,1),(1,2,1)
    # j=2,k=2: (0,2,2),(1,2,2)
    
    # Each of these 10 requires at least... well, they're independent in the sense that
    # no Strassen term covers them. So we need at least 10 more? That gives 17.
    # But wait - can we share terms? A single rank-1 term contributes to exactly one
    # (i,j,k) position in the tensor (since T[a,b,c] is 1 only at specific positions).
    # Actually no - a rank-1 term U[:,r] * V[:,r] * W[:,r] contributes to ALL (a,b,c)
    # where U[a,r]*V[b,r]*W[c,r] != 0. So it can contribute to multiple positions!
    
    # This is the key insight: a single term can cover multiple (i,j,k) positions if
    # the U, V, W vectors have multiple non-zero entries.
    
    # Let's try rank 17 with a structured search.
    # Given time constraints, let me try a targeted approach:
    # Build Strassen for 2x2x2 block (7 terms) + 10 schoolbook terms for the rest = 17.
    # But verify it's actually correct (the Strassen terms should only affect the 2x2x2 block).
    
    U17 = np.zeros((nm, 17), dtype=int)
    V17 = np.zeros((mp, 17), dtype=int)
    W17 = np.zeros((pn, 17), dtype=int)
    
    col = 0
    
    # Strassen 2x2x2: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # In global indices:
    # A[i,j] = U index i*m+j = i*3+j for i in {0,1}, j in {0,1} -> {0,1,3,4}
    # B[j,k] = V index j*p+k = j*3+k for j in {0,1}, k in {0,1} -> {0,1,3,4}
    # C[i,k] = W index k*n+i = k*2+i for i in {0,1}, k in {0,1} -> {0,1,2,3}
    
    # Standard Strassen for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 - M6
    # C11 = M1 - M2 + M5
    
    # Map to global:
    # A00 = a=0, A01 = a=1, A10 = a=3, A11 = a=4
    # B00 = b=0, B01 = b=1, B10 = b=3, B11 = b=4
    # C00 = c=0, C01 = c=1, C10 = c=2, C11 = c=3
    
    # M1 = (A00+A11)(B00+B11)
    U17[0, col] = 1; U17[4, col] = 1
    V17[0, col] = 1; V17[4, col] = 1
    W17[0, col] = 1; W17[3, col] = 1
    col += 1  # col=1
    
    # M2 = (A10+A11)B00
    U17[3, col] = 1; U17[4, col] = 1
    V17[0, col] = 1
    W17[2, col] = 1; W17[3, col] = -1
    col += 1  # col=2
    
    # M3 = A00(B01-B11)
    U17[0, col] = 1
    V17[1, col] = 1; V17[4, col] = -1
    W17[1, col] = 1
    col += 1  # col=3
    
    # M4 = A11(B10-B00)
    U17[4, col] = 1
    V17[3, col] = 1; V17[0, col] = -1
    W17[0, col] = -1; W17[2, col] = 1
    col += 1  # col=4
    
    # M5 = (A00+A01)B11
    U17[0, col] = 1; U17[1, col] = 1
    V17[4, col] = 1
    W17[1, col] = 1; W17[3, col] = 1
    col += 1  # col=5
    
    # M6 = (A10-A00)(B00+B01)
    U17[3, col] = 1; U17[0, col] = -1
    V17[0, col] = 1; V17[1, col] = 1
    W17[2, col] = -1
    col += 1  # col=6
    
    # M7 = (A01-A11)(B10+B11)
    U17[1, col] = 1; U17[4, col] = -1
    V17[3, col] = 1; V17[4, col] = 1
    W17[1, col] = 1; W17[3, col] = -1
    col += 1  # col=7
    
    # Now 10 schoolbook terms for the remaining (i,j,k) with j=2 or k=2 (but not both in {0,1}x{0,1})
    remaining = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in [0, 1] and k in [0, 1]:
                    continue  # already handled by Strassen
                remaining.append((i, j, k))
    
    # remaining should have 10 elements
    for (i, j, k) in remaining:
        a = i * m + j
        b = j * p + k
        c = k * n + i
        U17[a, col] = 1
        V17[b, col] = 1
        W17[c, col] = 1
        col += 1
    
    # Verify rank-17
    if col == 17 and verify_int(U17, V17, W17, T_target):
        return {
            "U": to_lists(U17),
            "V": to_lists(V17),
            "W": to_lists(W17)
        }
    
    # If rank-17 failed verification, try rank-16 with a search
    # This is harder. Let's try to see if any two of the 10 schoolbook terms
    # can be merged into one term.
    
    # Actually, let me first check if the Strassen construction above is correct
    # by verifying just the 2x2x2 part
    print(f"Rank-17 verification: {'PASS' if verify_int(U17, V17, W17, T_target) else 'FAIL'}")
    print(f"Col count: {col}")
    
    # Debug: check each term
    for r in range(17):
        contrib = np.einsum('a,b,c->abc', U17[:,r], V17[:,r], W17[:,r])
        nonzero = np.nonzero(contrib)
        if len(nonzero[0]) > 0:
            positions = [(nonzero[0][i], nonzero[1][i], nonzero[2][i]) for i in range(len(nonzero[0]))]
            values = [contrib[nonzero[0][i], nonzero[1][i], nonzero[2][i]] for i in range(len(nonzero[0]))]
            # print(f"Term {r}: {list(zip(positions, values))}")
    
    # Check where errors are
    if not verify_int(U17, V17, W17, T_target):
        errors = []
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    total = 0
                    for r in range(17):
                        total += U17[a, r] * V17[b, r] * W17[c, r]
                    if total != T_target[a, b, c]:
                        errors.append((a, b, c, T_target[a, b, c], total))
        print(f"Number of errors: {len(errors)}")
        for e in errors[:20]:
            print(f"  T[{e[0]},{e[1]},{e[2]}] expected {e[3]}, got {e[4]}")
    
    # Return schoolbook rank 18 as fallback
    return {
        "U": to_lists(U_sb),
        "V": to_lists(V_sb),
        "W": to_lists(W_sb)
    }
