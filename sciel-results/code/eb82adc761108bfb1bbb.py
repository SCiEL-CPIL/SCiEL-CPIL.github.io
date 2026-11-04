def construct():
    n, m, p = 2, 3, 10
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A[:, 0] * B[0, :] — 20 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: A[:, 1:3] * B[1:3, :] using 32 terms (6.4 per k-pair on average)
    # For each pair (k, k+1) for k = 0,2,4,6,8 (5 pairs), use 32 total terms
    # We use a more efficient decomposition than 7 per pair

    # A' indices in U:
    # A00 (i=0, j=1) -> idx 1
    # A01 (i=0, j=2) -> idx 2
    # A10 (i=1, j=1) -> idx 4
    # A11 (i=1, j=2) -> idx 5

    # B' indices in V:
    # B00 (j=1, k) -> idx 10+k
    # B01 (j=1, k+1) -> idx 10+k+1
    # B10 (j=2, k) -> idx 20+k
    # B11 (j=2, k+1) -> idx 20+k+1

    # C indices in W:
    # C00 (i=0, k) -> idx 2k
    # C01 (i=0, k+1) -> idx 2(k+1)
    # C10 (i=1, k) -> idx 2k+1
    # C11 (i=1, k+1) -> idx 2(k+1)+1

    # Use 32 terms for 5 k-pairs = 6.4 per pair
    # We use a clever decomposition with 32 total terms

    for b in range(5):
        k0 = 2 * b

        # M1 = (A00 + A11)(B00 + B11)
        U[1][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1       # C00
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1

        # M2 = (A10 + A11)B00
        U[4][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        W[2 * k0 + 1][col] += 1       # C10
        W[2 * (k0 + 1) + 1][col] -= 1 # C11
        col += 1

        # M3 = A00(B01 - B11)
        U[1][col] += 1
        V[10 + k0 + 1][col] += 1
        V[20 + k0 + 1][col] -= 1
        W[2 * (k0 + 1)][col] += 1     # C01
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1

        # M4 = A11(B10 - B00)
        U[5][col] += 1
        V[20 + k0][col] += 1
        V[10 + k0][col] -= 1
        W[2 * k0][col] += 1           # C00
        W[2 * k0 + 1][col] += 1       # C10
        col += 1

        # M5 = (A00 + A01)B11
        U[1][col] += 1
        U[2][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] -= 1           # C00
        W[2 * (k0 + 1)][col] += 1     # C01
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[4][col] += 1
        U[1][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[2][col] += 1
        U[5][col] -= 1
        V[20 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1           # C00
        col += 1

    # We have 20 + 35 = 55 terms so far. We need to reduce to 52.
    # Let's remove the last 3 terms by combining them into the existing structure
    # Actually, let's try a different approach: use 32 terms for the second part

    # Reset and rebuild with 32 terms for part 2
    col = 0

    # Part 1: A[:, 0] * B[0, :] — 20 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: Use 32 terms total for A[:,1:3] * B[1:3,:]
    # 5 k-pairs, need 32 terms = 6.4 per pair
    # Let's use 7 terms for 4 pairs and 4 terms for 1 pair, or distribute differently
    # Actually, let's use 6 terms for 2 pairs and 4 terms for 3 pairs, etc.
    # Better: use a uniform approach with 32 terms

    # For each k-pair, we need to compute:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11

    # That's 8 products per pair. With Strassen we can do 7.
    # For 5 pairs: 35 terms. To get 32, we need to save 3 terms.

    # Let's try: for the first 3 pairs use 7 terms (21), for last 2 pairs use 5.5 each (11) = 32
    # Or: use 7 for 2 pairs (14), 6 for 2 pairs (12), 5 for 1 pair (5) = 31... not quite

    # Let's just use 35 terms and accept rank 55, but that's what we had before.
    # Instead, let's try to be smarter.

    # Actually, let's use 32 terms by being more creative.
    # Use 7 terms for each of 4 pairs (28) + 4 terms for 1 pair (4) = 32

    # For the 5th pair, we can use a simpler decomposition
    for b in range(4):
        k0 = 2 * b

        # M1 = (A00 + A11)(B00 + B11)
        U[1][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1       # C00
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1

        # M2 = (A10 + A11)B00
        U[4][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        W[2 * k0 + 1][col] += 1       # C10
        W[2 * (k0 + 1) + 1][col] -= 1 # C11
        col += 1

        # M3 = A00(B01 - B11)
        U[1][col] += 1
        V[10 + k0 + 1][col] += 1
        V[20 + k0 + 1][col] -= 1
        W[2 * (k0 + 1)][col] += 1     # C01
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1

        # M4 = A11(B10 - B00)
        U[5][col] += 1
        V[20 + k0][col] += 1
        V[10 + k0][col] -= 1
        W[2 * k0][col] += 1           # C00
        W[2 * k0 + 1][col] += 1       # C10
        col += 1

        # M5 = (A00 + A01)B11
        U[1][col] += 1
        U[2][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] -= 1           # C00
        W[2 * (k0 + 1)][col] += 1     # C01
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[4][col] += 1
        U[1][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[2][col] += 1
        U[5][col] -= 1
        V[20 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1           # C00
        col += 1

    # For the 5th pair (k=8, k=9), use 4 terms
    k0 = 8

    # Direct 4-term decomposition for this pair
    # C00 = A00*B00
    U[1][col] += 1
    V[10 + k0][col] += 1
    W[2 * k0][col] += 1
    col += 1

    # C01 = A00*B01
    U[1][col] += 1
    V[10 + k0 + 1][col] += 1
    W[2 * (k0 + 1)][col] += 1
    col += 1

    # C10 = A10*B00
    U[4][col] += 1
    V[10 + k0][col] += 1
    W[2 * k0 + 1][col] += 1
    col += 1

    # C11 = A11*B11
    U[5][col] += 1
    V[20 + k0 + 1][col] += 1
    W[2 * (k0 + 1) + 1][col] += 1
    col += 1

    # Wait, this is 4 terms but misses A01*B10 and A10*B01 and A11*B10 and A01*B11
    # This won't work. Let me reconsider.

    # Actually, for the 5th pair we still need all 8 products.
    # Let's just use 7 terms for all 5 pairs = 35, total rank 55.
    # Or find a way to share terms across pairs.

    # Let me try a different approach: use 32 terms by combining some products
    # across the 5 pairs.

    # Reset
    col = 0

    # Part 1: 20 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: 32 terms
    # Use 7 terms for pairs 0-3 (28 terms) and 4 terms for pair 4
    # But 4 terms is not enough for pair 4.
    # Let's use 6 terms for pair 4 and 7 for pairs 0-3 = 6+28 = 34. Still 54 total.
    # Let's use 5 terms for pair 4: 5+28 = 33. Total 53.
    # Let's use 4 terms for pair 4 and 7 for others: 4+28=32. Total 52.

    # For pair 4, use 4 terms (won't be complete, so this won't work)

    # I think the best I can do is 55. Let me just return the original 55-term solution.
    col = 0

    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    for b in range(5):
        k0 = 2 * b

        U[1][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1

        U[4][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        W[2 * k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] -= 1
        col += 1

        U[1][col] += 1
        V[10 + k0 + 1][col] += 1
        V[20 + k0 + 1][col] -= 1
        W[2 * (k0 + 1)][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1

        U[5][col] += 1
        V[20 + k0][col] += 1
        V[10 + k0][col] -= 1
        W[2 * k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1

        U[1][col] += 1
        U[2][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] -= 1
        W[2 * (k0 + 1)][col] += 1
        col += 1

        U[4][col] += 1
        U[1][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1

        U[2][col] += 1
        U[5][col] -= 1
        V[20 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
