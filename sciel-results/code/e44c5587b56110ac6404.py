import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 17
    MIN_DIST = 6
    FULL_MASK = (1 << N) - 1

    # Step 1: Construct the base [17,9,6] code via GF(256)
    poly = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1 (primitive)

    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    # Find primitive 17th root of unity in GF(256)
    beta = None
    for candidate in range(2, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            beta = candidate
            break

    if beta is None:
        # Fallback: return a small valid code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Build generator polynomial G(x) = prod_{i=0}^{8} (x - beta^(2^i))
    # This gives a [17,9,6] BCH code
    g_coeffs = [1]  # Start with 1
    for i in range(9):
        root = gf_pow(beta, 1 << i)
        # Multiply g_coeffs by (x - root) = (x + root) in GF(2)
        new_len = len(g_coeffs) + 1
        new_coeffs = [0] * new_len
        for j in range(len(g_coeffs)):
            new_coeffs[j + 1] ^= g_coeffs[j]          # x * g_coeffs[j]
            new_coeffs[j] ^= gf_mul(g_coeffs[j], root)  # root * g_coeffs[j]
        g_coeffs = new_coeffs

    # g_coeffs has length 10 (degree 9), representing g(x) = sum g_coeffs[i] x^i
    # The generator matrix rows are shifts of g
    # For a [17,9] code, generator polynomial degree is 17-9=8
    # Wait, let me reconsider. For [n,k,d] BCH, the generator polynomial has degree n-k.
    # For [17,9,6], degree should be 8.
    # The roots are beta^1, beta^2, ..., beta^8 (consecutive roots)
    # Let me redo this properly.

    # Actually for the quadratic residue code of length 17:
    # The QR code has roots beta, beta^3, beta^9, beta^10, beta^13, beta^5, beta^15, beta^14, beta^8
    # where beta is a primitive 17th root. The defining set is the QR residues mod 17.
    # QR residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16, 1} -> {1,2,4,8,9,13,15,16}
    # Wait, 17 is prime. QR residues mod 17 are squares: 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
    # So QR = {1,2,4,8,9,13,15,16}
    # The QR code has defining set = QR, so generator poly = prod_{a in QR} (x - beta^a)
    # This gives degree 8, so [17,9] code.

    qr_residues = [1, 2, 4, 8, 9, 13, 15, 16]
    g_coeffs = [1]
    for a in qr_residues:
        root = gf_pow(beta, a)
        new_len = len(g_coeffs) + 1
        new_coeffs = [0] * new_len
        for j in range(len(g_coeffs)):
            new_coeffs[j + 1] ^= g_coeffs[j]
            new_coeffs[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_coeffs

    # g_coeffs should now have length 9 (degree 8)
    # Generate all 2^9 = 512 codewords, but the code is [17,9] so we get 512 codewords
    # Each message u (9 bits) gives codeword c = u * G(x) mod (x^17 - 1)
    # But for systematic form, we can use: c(x) = x^9 * m(x) + r(x) where r = x^9*m mod g

    # Simpler: just compute c(x) = m(x) * g(x) mod (x^17 - 1)
    # Since deg(g) = 8 and deg(m) <= 8, deg(c) <= 16, so no wrap-around needed
    # c(x) = m(x) * g(x) directly (no mod needed since deg < 17)

    codewords = []
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                # Multiply g by x^i
                for j in range(9):
                    if g_coeffs[j]:
                        c ^= 1 << (i + j)
        codewords.append(c)

    # This gives 512 codewords of length 17
    # But wait, the [17,9,6] code has 2^9 = 512 codewords? No, it has 2^9 = 512 codewords.
    # Actually [n,k] means 2^k codewords. So [17,9] has 512 codewords.
    # But the task says the linear reference is 256 words. Let me re-read.
    # "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate"
    # So the [18,9,6] code has 2^9 = 512 codewords, extended from [17,9,5] or something?
    # Actually, the QR code of length 17 is [17,9,5] typically. The extended version is [18,9,6].
    # Let me just use the 512 codewords we have and check their minimum distance.

    # Check min distance
    min_d = 99
    for i in range(len(codewords)):
        for j in range(i+1, len(codewords)):
            d = (codewords[i] ^ codewords[j]).bit_count()
            min_d = min(min_d, d)

    # If min distance is >= 6, we can use all 512
    # If not, we need to select a subset
    if min_d >= MIN_DIST:
        base_code = list(set(codewords))
    else:
        # Greedily select codewords with min distance >= 6
        base_code = []
        used = set()
        for c in codewords:
            ok = True
            for b in base_code:
                if (c ^ b).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                base_code.append(c)
                used.add(c)

    # Ensure we have at least the seed
    if len(base_code) < 32:
        # Fallback
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    code_list = list(set(base_code))
    code_set = set(code_list)

    # Step 2: Greedy expansion
    # Strategy: generate candidates by flipping bits of existing codewords
    # A new word w must be at distance >= 6 from all existing codewords.
    # Candidates likely to work are at distance 6 from some existing codeword.

    # Precompute for speed
    used_arr = np.zeros(1 << N, dtype=bool)
    for w in code_list:
        used_arr[w] = True

    code_arr = np.array(code_list, dtype=np.uint32)

    # Generate candidate words: all words at distance exactly 6 from any codeword
    # There are C(17,6) = 12376 words at distance 6 from a given word.
    # With 256+ codewords, that's ~3M candidates, but many duplicates.
    # Use a set to deduplicate.

    # Precompute all 17-bit masks with weight 6
    weight6_masks = []
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    for m in range(l+1, 17):
                        for n in range(m+1, 17):
                            mask = (1<<i)|(1<<j)|(1<<k)|(1<<l)|(1<<m)|(1<<n)
                            weight6_masks.append(mask)
    weight6_masks = np.array(weight6_masks, dtype=np.uint32)

    # Generate all candidates
    candidates_set = set()
    for c in code_list:
        c_np = np.uint32(c)
        cands = c_np ^ weight6_masks
        for cand in cands:
            cand_int = int(cand)
            if cand_int not in code_set:
                candidates_set.add(cand_int)

    # Sort candidates by some heuristic (e.g., weight)
    candidates_list = list(candidates_set)
    # Sort by weight, then by value
    candidates_list.sort(key=lambda x: (x.bit_count(), x))

    # Greedy add
    added_count = 0
    for w in candidates_list:
        if time.time() > deadline - 5:
            break

        # Check distance to all existing codewords
        w_np = np.uint32(w)
        dists = np.bit_count(code_arr ^ w_np)
        if np.min(dists) >= MIN_DIST:
            code_list.append(w)
            code_set.add(w)
            code_arr = np.append(code_arr, w_np)
            used_arr[w] = True
            added_count += 1

    # Step 3: Try more rounds with distance 7 candidates
    if time.time() < deadline - 10:
        weight7_masks = []
        for i in range(17):
            for j in range(i+1, 17):
                for k in range(j+1, 17):
                    for l in range(k+1, 17):
                        for m in range(l+1, 17):
                            for n in range(m+1, 17):
                                for o in range(n+1, 17):
                                    mask = (1<<i)|(1<<j)|(1<<k)|(1<<l)|(1<<m)|(1<<n)|(1<<o)
                                    weight7_masks.append(mask)
        weight7_masks = np.array(weight7_masks, dtype=np.uint32)

        candidates_set2 = set()
        for c in code_list:
            c_np = np.uint32(c)
            cands = c_np ^ weight7_masks
            for cand in cands:
                cand_int = int(cand)
                if cand_int not in code_set:
                    candidates_set2.add(cand_int)

        candidates_list2 = list(candidates_set2)
        candidates_list2.sort(key=lambda x: (x.bit_count(), x))

        for w in candidates_list2:
            if time.time() > deadline - 5:
                break
            w_np = np.uint32(w)
            dists = np.bit_count(code_arr ^ w_np)
            if np.min(dists) >= MIN_DIST:
                code_list.append(w)
                code_set.add(w)
                code_arr = np.append(code_arr, w_np)
                used_arr[w] = True

    # Step 4: Final pass - try random perturbations
    if time.time() < deadline - 5:
        import random
        random.seed(42)
        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            w = random.randint(0, (1 << N) - 1)
            if w in code_set:
                continue
            w_np = np.uint32(w)
            dists = np.bit_count(code_arr ^ w_np)
            if np.min(dists) >= MIN_DIST:
                code_list.append(w)
                code_set.add(w)
                code_arr = np.append(code_arr, w_np)

    return code_list
