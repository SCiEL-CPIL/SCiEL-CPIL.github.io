import time
import numpy as np


def _bit_count_u8(x):
    """Popcount for 18-bit values using a 256-entry lookup for the low byte."""
    lut = np.array(
        [bin(i).count("1") for i in range(256)],
        dtype=np.uint8,
    )
    return lut[(x & 0xFF).astype(np.uint8)].astype(np.int32) + (
        lut[(x >> 8).astype(np.uint8)].astype(np.int32)
    )


def _make_code_from_generator(gen_poly, n, k):
    """Build all 2^k codewords of a binary cyclic code from its generator polynomial.

    gen_poly: integer whose bit i is 1 iff x^i appears.
    Returns list of length-2^k integers (n-bit codewords, low bit = position 0).
    """
    m = n - k  # parity-check degree
    mask_n = (1 << n) - 1
    words = []
    for msg in range(1 << k):
        # c(x) = m(x) * g(x), truncated to degree < n
        c = 0
        for i in range(k):
            if (msg >> i) & 1:
                c ^= gen_poly << i
        c &= mask_n
        words.append(c)
    return words


def _make_linear_code(gens, n):
    """All 2^k XOR combinations of the given generator vectors."""
    k = len(gens)
    words = [0]
    for g in gens:
        new_words = [w ^ g for w in words]
        words = words + new_words
    return words


def _verify_code(words, min_dist):
    """Return True iff every pair has Hamming distance >= min_dist. Vectorized."""
    if len(words) < 2:
        return True
    arr = np.array(words, dtype=np.uint32)
    n = len(arr)
    # Check in chunks to avoid OOM
    chunk = 4096
    for i in range(0, n, chunk):
        a = arr[i : i + chunk]
        for j in range(0, n, chunk):
            b = arr[j : j + chunk]
            # Only check pairs where global index i-chunk < j-chunk, or within chunk i<j
            xor = a[:, None] ^ b[None, :]
            d = _bit_count_u8(xor)
            # For the diagonal block (i == j region), zero distance is expected for self
            # We need to ensure no off-diagonal pair is < min_dist
            # Mask out the self-comparison
            rows = np.arange(a.shape[0])[:, None]
            cols = np.arange(b.shape[1])[None, :]
            if i == j:
                mask = rows != cols
            else:
                mask = np.ones((a.shape[0], b.shape[1]), dtype=bool)
            if not np.all(d[mask] >= min_dist):
                return False
    return True


def _vectorized_greedy_extend(base_words, min_dist, deadline, n_bits, order="weight"):
    """Extend a valid code by greedily adding words from a candidate pool.

    Uses vectorized NumPy conflict checks. Returns the extended list.
    """
    words = list(base_words)
    word_set = set(words)
    total = 1 << n_bits
    arr = np.array(words, dtype=np.uint32)

    # Candidate order: by weight (popcount), then value.
    # Precompute all weights vectorized.
    all_idx = np.arange(total, dtype=np.uint32)
    weights = _bit_count_u8(all_idx)

    if order == "weight":
        # Sort indices by (weight, value)
        # Use a stable argsort on a combined key: weight * total + value
        keys = (weights.astype(np.int64) * total) + all_idx.astype(np.int64)
        order_idx = np.argsort(keys, kind="stable")
    else:
        order_idx = np.arange(total)

    elapsed = time.time()
    deadline_check_interval = 20000  # check deadline every N candidates

    for count, idx in enumerate(order_idx):
        if count % deadline_check_interval == 0:
            if time.time() > deadline - 2.0:
                break

        cand = int(idx)
        if cand in word_set:
            continue

        # Vectorized conflict check against all current words
        xor = arr ^ cand
        d = _bit_count_u8(xor)
        if np.all(d >= min_dist):
            words.append(cand)
            word_set.add(cand)
            arr = np.append(arr, np.array([cand], dtype=np.uint32))
            # Optionally shrink the candidate pool? No, keep iterating.

    return words


def _greedy_add_random_candidates(base_words, min_dist, deadline, n_bits, seed=123):
    """Additional search: try random candidates not in the code."""
    words = list(base_words)
    word_set = set(words)
    arr = np.array(words, dtype=np.uint32)
    rng = np.random.default_rng(seed)

    while time.time() < deadline - 2.0:
        # Generate a batch of random candidates
        batch_size = 20000
        batch = rng.integers(0, 1 << n_bits, size=batch_size, dtype=np.uint32)

        for cand in batch:
            cand = int(cand)
            if cand in word_set:
                continue
            xor = arr ^ cand
            d = _bit_count_u8(xor)
            if np.all(d >= min_dist):
                words.append(cand)
                word_set.add(cand)
                arr = np.append(arr, np.array([cand], dtype=np.uint32))

    return words


def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # --- Construction 1: Extended QR(17) code via generator polynomial ---
    # QR(17) has generator polynomial (in standard cyclic code convention,
    # g(x) is the product of minimal polynomials of alpha and alpha^2, etc.)
    # A known generator for the [17,7,5] QR code: g(x) = x^10 + x^9 + x^8 + x^7 + x^4 + x^3 + x^2 + 1
    # Let's try several candidate generator polynomials and verify.

    candidate_gen_polys = [
        # Various candidate generator polynomials of degree 10 for the [17,7,5] QR code
        0b1001011111,  # x^10 + x^8 + x^6 + x^4 + x^3 + x^2 + x + 1
        0b1000000001,  # x^10 + 1 (unlikely)
        0b1111100011,  # some other
    ]

    # Actually, let me use a more reliable construction.
    # The [18, 8, 6] code can be built from the [16, 5, 6] Reed-Muller-like code
    # or from known constructions. Let me use a verified approach:
    # Build a [18, 8, 6] code by extending a [17, 7, 5] code with a parity bit,
    # but I need the actual [17,7,5] code first.

    # Alternative: use the shortened/extended BCH code.
    # For n=17, the BCH code with designed distance 5 has parameters [17, 9, 5] or [17,7,5].
    # Let me just try to find a valid generator polynomial by search.

    # Better approach: use the fact that we can build a valid [18,8,6] code
    # from the quadratic residue construction more carefully.

    # Let me use a different, verified construction:
    # The [18, 8, 6] code is the dual of the [18, 10, 5] code.
    # Or: take the [16, 11, 4] code and extend... no.

    # Most reliable: use the extended Hamming code structure.
    # Actually, let me just use a known valid 256-word code.
    # I'll construct it as follows:
    # The code is the set of all vectors (f(0), f(1), ..., f(17)) where f is a
    # polynomial of degree <= 2 over GF(2) evaluated at points... no, that's too few.

    # Let me use the BCH code approach properly.
    # For the binary BCH code of length 17 with designed distance 5:
    # The roots are alpha, alpha^2, alpha^3, alpha^4 where alpha is a primitive 17th root.
    # The generator is lcm(M1(x), M3(x)) where M_i is the minimal polynomial of alpha^i.
    # For p=17, the cyclotomic cosets mod 17: {0}, {1,2,4,8,16,15,13,9}, {3,6,12,7,14,11,5,10}
    # M1(x) has degree 8, M3(x) has degree 8.
    # g(x) = lcm(M1, M3) = M1 * M3 (since they're distinct) = degree 16? That gives [17,1,?]
    # Hmm, that's too large. For designed distance 5, roots alpha^1..alpha^4.
    # alpha^2 is in coset of alpha^1, alpha^4 is in coset of alpha^1.
    # So roots in coset {1,2,4,8,...} cover alpha^1, alpha^2, alpha^4, alpha^8, alpha^16, alpha^15, alpha^13, alpha^9
    # And we need alpha^3: coset {3,6,12,7,14,11,5,10}
    # So g = M1 * M3, degree 16. [17, 1, ?] - that's the repetition code essentially.
    #
    # For [17,7,5]: dimension 7, so g has degree 10.
    # That means only cosets of total degree 10.
    # M1 has degree 8. We need 2 more degrees. That doesn't match a cyclotomic coset.
    #
    # Let me just use a different, simpler verified construction.

    # SIMPLEST RELIABLE APPROACH: Use the known [18, 8, 6] code from the
    # extended quadratic residue code. I'll build it using the fact that
    # it's equivalent to a specific linear code with known generators.

    # Let me use the following verified generators for an [18, 8, 6] code:
    # These are from the known construction. I'll verify them.
    #
    # Actually, the most robust approach: build the code by extending RM(1,4)
    # with 2 additional coordinates chosen to maintain distance 6, then verify.

    # RM(1,4): 32 words of length 16, min distance 8.
    # Extend to 18 bits by adding 2 bits per word.
    # The challenge: choosing the 2 extra bits to maintain distance >= 6.
    # Since original distance is 8, adding 2 bits can reduce distance by at most 2,
    # so we need the added bits to have distance >= 2 for pairs at distance 8,
    # and >= 0 for pairs at distance 6 or more.
    # Actually, for pairs at distance 8 in the 16-bit part, we need the 2-bit extension
    # to have distance >= 6 - 8 = -2, which is always satisfied.
    # For pairs at distance 6 in the 16-bit part... but RM(1,4) has min distance 8,
    # so all pairs are at distance >= 8. Adding 2 bits: worst case distance becomes 8-2=6.
    # So we just need to ensure that no pair drops below 6, which means the 2-bit
    # extensions should not both be 0 for any pair that's at distance 8.
    # Actually, if both bits are the same for two words, the distance is unchanged (8).
    # If they differ in 1 bit, distance is 7. If 2 bits, distance is 6.
    # All are >= 6, so ANY extension works!

    # So: take RM(1,4) (32 words, 16 bits), extend to 18 bits by adding 2 zero bits.
    # This gives 32 words of length 18 with min distance >= 6.
    # Then greedily extend to get more words.

    # But we can do better: use a larger base. Let me try to find the 256-word QR code.

    # Let me try a different approach: build the [18,8,6] code by taking
    # all linear combinations of 8 generator vectors, and verify.
    # I'll use generators from the known extended QR code.

    # For now, let me use the RM(1,4) base (32 words) and greedy extend,
    # plus try a second construction with a different base.

    # Base 1: RM(1,4) extended to 18 bits
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
    # Extend to 18 bits: add 2 zero bits (bits 16 and 17)
    base1 = [w for w in rm_words]  # already < 2^16, so bits 16,17 are 0

    # Verify base1
    if not _verify_code(base1, MIN_DIST):
        # Should not happen, but just in case
        base1 = [0]

    # Base 2: Try to construct a larger base using a linear code with known generators
    # I'll use the [18, 8, 6] code from the extended QR construction.
    # Let me try to find valid generators by using the structure.
    #
    # Alternative base 2: use a [18, 8, 6] code built from the [17, 7, 5] QR code.
    # I'll search for a generator polynomial that gives a valid [17,7,5] code.

    base2 = None
    for gen_poly in range(1, 1 << 11):
        # Only consider polynomials with constant term 1 and degree exactly 10
        if gen_poly & 1 == 0:
            continue
        if bin(gen_poly).count("1") == 0 or (gen_poly >> 10) == 0:
            continue
        if gen_poly >= (1 << 11):
            continue

        # Build the [17, 7, 5] code
        try:
            qr_words = _make_code_from_generator(gen_poly, 17, 7)
        except Exception:
            continue

        # Extend to 18 bits by adding a parity bit
        # The parity bit for each word: sum of bits mod 2
        extended_words = []
        for w in qr_words:
            parity = bin(w).count("1") % 2
            ew = w | (parity << 17)
            extended_words.append(ew)

        # Verify the extended code
        if _verify_code(extended_words, MIN_DIST):
            base2 = extended_words
            break

    if base2 is None:
        # Fallback: use base1 for both
        base2 = base1

    # Choose the larger base for greedy extension
    if len(base2) >= len(base1):
        base = base2
    else:
        base = base1

    # Verify the chosen base
    if not _verify_code(base, MIN_DIST):
        base = base1 if _verify_code(base1, MIN_DIST) else [0]

    # Greedy extension
    extended = _vectorized_greedy_extend(base, MIN_DIST, deadline, N, order="weight")

    # If time permits, do a second pass with random candidates
    if time.time() < deadline - 5.0:
        extended2 = _greedy_add_random_candidates(
            extended, MIN_DIST, deadline, N, seed=456
        )
        if len(extended2) > len(extended):
            extended = extended2

    # Final verification
    if not _verify_code(extended, MIN_DIST):
        # Fallback to the base
        extended = base

    return extended
