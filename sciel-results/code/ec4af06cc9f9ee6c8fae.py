"""Construct a binary code of length 17 with minimum distance 6."""
import time
import random

def construct():
    deadline = time.time() + 75  # safety margin

    # Step 1: Build the [17,8,6] shortened extended quadratic-residue code (256 words).
    # The [17,9,5] QR code has generator polynomial g(x) = x^9 + x^6 + x^5 + x^4 + x + 1.
    g = (1 << 0) | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 9)

    # Generate all 512 codewords: m(x)*g(x) mod (x^17 - 1) for m in F_2[x], deg < 9
    qr17 = []
    for m in range(256):
        word = 0
        for i in range(9):
            if (m >> i) & 1:
                shifted = g << i
                # Reduce modulo x^17 - 1 (x^17 = 1 in F_2[x]/(x^17-1))
                for j in range(35, 16, -1):
                    if (shifted >> j) & 1:
                        shifted ^= (1 << j)
                        shifted ^= (1 << (j - 17))
                word ^= shifted & ((1 << 17) - 1)
        qr17.append(word)

    # Shorten by coordinate 0: keep words with bit 0 = 0, drop that bit
    base = list(set(w >> 1 for w in qr17 if (w & 1) == 0))

    def dist(a, b):
        return (a ^ b).bit_count()

    # Step 2: Greedy extension over the full 2^17 space
    words = list(base)
    word_set = set(words)

    # Candidates: all words with weight >= 6 (necessary since 0 is in the code)
    cands = [x for x in range(1 << 17) if x.bit_count() >= 6]
    cands.sort(key=lambda x: x.bit_count())

    for c in cands:
        if time.time() > deadline - 25:
            break
        if c in word_set:
            continue
        ok = True
        for w in words:
            if dist(c, w) < 6:
                ok = False
                break
        if ok:
            words.append(c)
            word_set.add(c)

    best = list(words)
    best_n = len(words)

    # Step 3: Random-restart greedy to find better orderings
    rng = random.Random(2024)
    for _ in range(15):
        if time.time() > deadline - 25:
            break
        rng.shuffle(cands)
        tw = list(base)
        ts = set(tw)
        for c in cands:
            if c in ts:
                continue
            ok = True
            for w in tw:
                if dist(c, w) < 6:
                    ok = False
                    break
            if ok:
                tw.append(c)
                ts.add(c)
        if len(tw) > best_n:
            best_n = len(tw)
            best = tw

    return best
