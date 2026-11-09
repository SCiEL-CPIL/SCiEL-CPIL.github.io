import time
import random
import itertools

BASE = 51975
DEADLINE = 140.0

# divisors of 51975 = 3^3 * 5^2 * 7 * 11
DIVS = [3, 5, 7, 9, 11, 15, 21, 25, 33, 35, 45, 55, 63, 75, 77, 99, 105, 135, 165, 175,
        231, 275, 315, 385, 441, 495, 525, 693, 825, 945, 1155, 1375, 1575, 2205, 2771,
        3465, 4725, 5390, 5544, 6930, 8925, 13850, 17850, 23100, 34650, 46200, 47250,
        69300, 94500, 138500, 231000, 346500, 472500, 693000, 945000, 1385000, 2310000,
        3465000, 4725000, 6930000, 9450000, 13850000, 23100000, 34650000, 47250000,
        69300000, 94500000, 138500000, 231000000, 346500000, 472500000, 693000000,
        945000000, 1385000000, 2310000000, 3465000000, 4725000000, 6930000000,
        9450000000, 13850000000, 23100000000, 34650000000, 47250000000, 69300000000,
        94500000000, 138500000000, 231000000000, 346500000000, 472500000000, 693000000000,
        945000000000, 1385000000000, 2310000000000, 3465000000000, 4725000000000,
        6930000000000, 9450000000000, 13850000000000, 23100000000000, 34650000000000,
        47250000000000, 69300000000000, 94500000000000, 138500000000000, 231000000000000,
        346500000000000, 472500000000000, 693000000000000, 945000000000000,
        1385000000000000, 2310000000000000, 3465000000000000, 4725000000000000,
        6930000000000000, 9450000000000000, 13850000000000000, 23100000000000000,
        34650000000000000, 47250000000000000, 69300000000000000, 94500000000000000,
        138500000000000000, 231000000000000000, 346500000000000000, 472500000000000000,
        693000000000000000, 945000000000000000, 1385000000000000000, 2310000000000000000]
# Correct the divisor list
DIVS = []
for a in range(4):      # 3^a
    for b in range(3):  # 5^b
        for c in range(2):  # 7^c
            for d in range(2):  # 11^d
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    DIVS.append(v)
DIVS.sort()

# bitsets
BITSETS = {}
for m in DIVS:
    bits = 0
    for r in range(m):
        for x in range(r, BASE, m):
            bits |= (1 << x)
    BITSETS[m] = bits

# order moduli by ascending size (most valuable first)
ORDER = DIVS  # already ascending
ORDER_IDX = {m: i for i, m in enumerate(ORDER)}
NMOD = len(ORDER)


def score_of(covered_mask):
    return BASE - covered_mask.bit_count()


def greedy_pick(m, covered_mask, covered_by):
    # pick residue r for modulus m maximizing newly covered
    best_r = 0
    best_gain = -1
    for r in range(m):
        bits = BITSETS[m] & ~(covered_mask)
        # shift to compute per-residue
        # actually need per-residue bitset
        pass
    # simpler: compute per residue
    best_r = 0
    best_gain = -1
    for r in range(m):
        mask = 0
        for x in range(r, BASE, m):
            mask |= (1 << x)
        gain = (mask & ~covered_mask).bit_count()
        if gain > best_gain:
            best_gain = gain
            best_r = r
    return best_r, best_gain


def local_search(classes, covered_mask, covered_by, deadline):
    # classes: dict m -> r
    used = set(classes.keys())
    cur = score_of(covered_mask)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # 1-move: change residue of a used modulus
        for m in list(classes.keys()):
            old_r = classes[m]
            old_bits = 0
            for x in range(old_r, BASE, m):
                old_bits |= (1 << x)
            for r in range(m):
                if r == old_r:
                    continue
                new_bits = 0
                for x in range(r, BASE, m):
                    new_bits |= (1 << x)
                new_mask = (covered_mask & ~old_bits) | new_bits
                s = score_of(new_mask)
                if s < cur:
                    covered_mask = new_mask
                    classes[m] = r
                    cur = s
                    improved = True
                    break
            if improved and time.time() > deadline - 2:
                break
        if time.time() > deadline - 2:
            break
        # 2-swap: change residue of used m1 AND add/swap unused m2
        # try swapping residue of one used with a fresh unused modulus
        for m1 in list(classes.keys()):
            old_r1 = classes[m1]
            old_bits1 = 0
            for x in range(old_r1, BASE, m1):
                old_bits1 |= (1 << x)
            for m2 in ORDER:
                if m2 in used:
                    continue
                for r2 in range(m2):
                    new_bits2 = 0
                    for x in range(r2, BASE, m2):
                        new_bits2 |= (1 << x)
                    new_mask = (covered_mask & ~old_bits1) | new_bits2
                    s = score_of(new_mask)
                    if s < cur:
                        covered_mask = new_mask
                        classes[m2] = r2
                        used.add(m2)
                        cur = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if time.time() > deadline - 2:
            break
    return classes, covered_mask, cur


def one_greedy(rng, deadline):
    # random order of moduli (ascending bias), pick best residue
    covered_mask = 0
    covered_by = {}
    classes = {}
    order = list(ORDER)
    # shuffle but keep small moduli early with high probability
    rng.shuffle(order)
    for m in order:
        if time.time() > deadline - 3:
            break
        # pick residue
        best_r = 0
        best_gain = -1
        for r in range(m):
            mask = 0
            for x in range(r, BASE, m):
                mask |= (1 << x)
            gain = (mask & ~covered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_r = r
        # add
        for x in range(best_r, BASE, m):
            covered_mask |= (1 << x)
        classes[m] = best_r
    return classes, covered_mask


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(12345)
    best_classes = None
    best_score = BASE + 1

    # deterministic greedy first
    classes, covered_mask = one_greedy(rng, deadline)
    s = score_of(covered_mask)
    if s < best_score:
        best_score = s
        best_classes = dict(classes)
    # local search
    classes2, covered_mask2, s2 = local_search(classes, covered_mask, {}, deadline)
    if s2 < best_score:
        best_score = s2
        best_classes = dict(classes2)

    # randomized restarts
    it = 0
    while time.time() < deadline - 5:
        it += 1
        classes, covered_mask = one_greedy(rng, deadline)
        s = score_of(covered_mask)
        if s < best_score:
            best_score = s
            best_classes = dict(classes)
        # local search on this
        classes2, covered_mask2, s2 = local_search(classes, covered_mask, {}, deadline)
        if s2 < best_score:
            best_score = s2
            best_classes = dict(classes2)
        if best_score == 0:
            break

    # build result
    if best_classes is None:
        # fallback: just the smallest modulus
        best_classes = {ORDER[0]: 0}
    return [[m, r] for m, r in best_classes.items()]
