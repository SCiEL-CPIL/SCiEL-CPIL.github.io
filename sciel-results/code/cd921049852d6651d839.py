import numpy as np
import time
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58

    # Precompute masks for blocking
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])

    # Helper to generate cyclic code from generator polynomial coefficients
    def gen_cyclic_code(g_coeffs):
        # g_coeffs is a list of 0/1, index 0 is x^0, index -1 is highest degree?
        # Standard: g(x) = g_0 + g_1 x + ... + g_{n-k} x^{n-k}
        # Generator matrix rows are shifts of g.
        # Row i is x^i g(x).
        # We need k rows.
        k = N - len(g_coeffs)
        G = []
        for i in range(k):
            row = 0
            for j, c in enumerate(g_coeffs):
                if c:
                    row |= (1 << (i + j))
            G.append(row)
        
        # Generate codewords
        codewords = [0]
        for i in range(k):
            new_codewords = [c ^ G[i] for c in codewords]
            codewords.extend(new_codewords)
        return codewords

    def get_min_dist(codewords):
        min_d = N + 1
        for c in codewords:
            if c == 0: continue
            w = bin(c).count('1')
            if w < min_d:
                min_d = w
        return min_d

    # Construct [17,8,6] code
    # Compute g1(x) and g3(x)
    # Roots of g1: C1 = {1, 2, 4, 8, 16, 15, 13, 9}
    # Roots of g3: C3 = {3, 6, 12, 7, 14, 11, 5, 10}
    
    def poly_mul(p1, p2):
        res = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            if c1:
                for j, c2 in enumerate(p2):
                    if c2:
                        res[i+j] ^= 1
        return res

    def poly_from_roots(roots):
        # roots are integers in GF(2^m) represented as powers of 2?
        # We work in GF(2). The roots are elements of GF(2^8)? No, GF(2) is not enough for degree 8.
        # We need to compute the minimal polynomial over GF(2).
        # The roots are in GF(2^8) or similar?
        # Actually, we can just compute the product of (x + alpha) where alpha are the roots.
        # But we don't have a field representation easily.
        # Alternative: Use the property that g(x) is the LCM of minimal polynomials.
        # Or just hardcode the generator matrix?
        # Let's try to find the generator matrix online or derive it.
        # A known generator matrix for [17,8,6] QR code:
        # It is the circulant matrix with first row:
        # 1 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 1 ?
        # Let's use a reliable source or a simple construction.
        # Actually, the [17,8,6] code is the **BCH code** with designed distance 6?
        # BCH code with n=17, d=6.
        # Roots: 1, 2, 4, 8, 16, 15, 13, 9 (C1) and 3, 6, 12, 7, 14, 11, 5, 10 (C3)?
        # If we take roots 1, 2, 3, 4, 5, 6, 7, 8?
        # The BCH code with roots 1, 2, 3, 4, 5, 6 has degree?
        # C1 (8) + C3 (8) = 16.
        # If we take C1, degree 8, k=9.
        # If we take C1 U C3, degree 16, k=1.
        # Wait, the [17,8,6] code has k=8.
        # This implies degree 9.
        # So it's not a standard BCH code with consecutive roots.
        
        # Let's use the fact that the [18,9,6] QR code is well known.
        # Shortening it gives [17,8,6].
        # The [18,9,6] code is the QR code of length 18? No, 18 is not prime.
        # The QR code is for p=17.
        # The extended QR code is [18,9,6].
        # We can construct the [17,8,6] code by taking the [17,9,5] QR code? No.
        
        # Let's just use a random search for a [17,8,6] code but with a better heuristic?
        # Or just use the greedy from 0 which seemed to work (score 8.0).
        # If greedy from 0 gives 256, and we can't easily construct the linear code,
        # maybe we should stick with greedy from 0 but improve the local search?
        
        # However, the linear code is "structured".
        # Let's try to construct the generator matrix by finding a basis for the dual code?
        # Too complex.
        
        # Let's assume the Greedy from 0 is the bottleneck or the starting point is suboptimal.
        # But if Greedy from 0 gives 256, and A(17,6) >= 258,
        # maybe the Greedy code is NOT the one that can be extended.
        # So we MUST find the linear code.
        
        # I will hardcode the generator matrix rows for the [17,8,6] code.
        # Source: Standard tables.
        # The generator matrix G is 8x17.
        # One common representation:
        # Row 0: 1 0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 1
        # ...
        # I don't have the exact rows memorized.
        
        # Alternative: Use the **binary Golay code** [23,12,7]?
        # Shorten to 17?
        # Shortening [23,12,7] by 6 coordinates gives [17,12-?, 7-?].
        # If we shorten by 6, k becomes 12-6=6? No, shortening reduces k by 1 per coordinate if the coordinate is non-zero in all?
        # Actually, shortening a code C at coordinate i gives a code of length n-1 and dimension k-1 if the projection is surjective.
        # So [23,12,7] -> [22,11,7] -> ... -> [17,6,7].
        # This is a [17,6,7] code. Size 64.
        # This is too small.
        
        # So we really need the [17,8,6] code.
        
        # Let's try the random search again but with more attempts or better initialization?
        # The current code tries 200 times.
        # Maybe it just needs more time?
        # But 200 random matrices is not enough.
        
        # I will implement a fallback:
        # 1. Try to find linear code (more attempts, e.g., 1000).
        # 2. If failed, use Greedy from 0.
        # 3. Run Greedy Expansion.
        # 4. Run Local Search.
        
        # To improve the chance of finding the linear code, we can use a **structured** random search?
        # Or just increase attempts to 1000.
        # 1000 attempts * (256 words * 17 bits) is fast.
        
        pass

    # Step 1: Find a [17,8,6] linear code
    k = 8
    linear_code = None

    for attempt in range(1000):
        if time.time() - start_time > 5.0:
            break
        G = np.random.randint(0, 2, size=(k, N))
        
        G_ints = []
        for i in range(k):
            val = 0
            for j in range(N):
                if G[i, j] == 1:
                    val |= (1 << j)
            G_ints.append(val)
        
        codewords = [0]
        for i in range(k):
            new_codewords = [c ^ G_ints[i] for c in codewords]
            codewords.extend(new_codewords)
        
        min_d = N + 1
        for c in codewords:
            if c == 0:
                continue
            w = bin(c).count('1')
            if w < min_d:
                min_d = w
                if min_d < MIN_DIST:
                    break
        
        if min_d >= MIN_DIST:
            linear_code = codewords
            break

    # Step 2: Fallback
    if linear_code is None:
        blocked = bytearray(1 << N)
        code = []
        
        def add_word(w):
            if blocked[w]:
                return
            code.append(w)
            for mask in all_masks:
                blocked[w ^ mask] = 1
        
        add_word(0)
        for w in range(1, 1 << N):
            if time.time() - start_time > time_limit:
                break
            if not blocked[w]:
                add_word(w)
        return code

    # Step 3: Expand
    blocked = bytearray(1 << N)
    code = list(linear_code)
    
    for w in code:
        for mask in all_masks:
            blocked[w ^ mask] = 1

    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            code.append(w)
            for mask in all_masks:
                blocked[w ^ mask] = 1

    # Step 4: Local Search
    if time.time() - start_time < time_limit - 10:
        # Try to remove and re-add
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for i in range(len(code)):
                if time.time() - start_time > time_limit - 5:
                    break
                removed = code[i]
                code.remove(removed)
                
                for mask in all_masks:
                    blocked[removed ^ mask] = 0
                
                old_len = len(code)
                for w in range(1 << N):
                    if time.time() - start_time > time_limit - 5:
                        break
                    if not blocked[w]:
                        code.append(w)
                        for mask in all_masks:
                            blocked[w ^ mask] = 1
                
                if len(code) > old_len:
                    improved = True
                else:
                    # Undo
                    while len(code) > old_len:
                        w = code.pop()
                        for mask in all_masks:
                            blocked[w ^ mask] = 0
                    for mask in all_masks:
                        blocked[removed ^ mask] = 1
                    code.append(removed)

    return code
