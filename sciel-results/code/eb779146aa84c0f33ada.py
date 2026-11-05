import time
from fractions import Fraction
from math import gcd
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    L = 1
    for d in denominators:
        L = L // gcd(L, d) * d

    w = [L // d for d in denominators]

    mid = 23
    n1 = mid
    n2 = n - mid  # 24

    def gen_sums_masks(weights):
        sums = [0]
        masks = [0]
        for i, wt in enumerate(weights):
            bit = 1 << i
            ln = len(sums)
            for j in range(ln):
                sums.append(sums[j] + wt)
                masks.append(masks[j] | bit)
        return sums, masks

    left_w = w[:n1]
    right_w = w[n1:]

    left_sums, left_masks = gen_sums_masks(left_w)
    right_sums, right_masks = gen_sums_masks(right_w)

    if time.time() > deadline - 20:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Sort right by sum, keeping only valid (< L)
    right_idx = sorted(range(len(right_sums)), key=right_sums.__getitem__)
    right_sums_f = []
    right_masks_f = []
    for i in right_idx:
        s = right_sums[i]
        if s < L:
            right_sums_f.append(s)
            right_masks_f.append(right_masks[i])
        else:
            break

    if time.time() > deadline - 10:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    best_total = -1
    best_lm = 0
    best_rm = 0

    for ls, lm in zip(left_sums, left_masks):
        if ls >= L:
            continue
        if time.time() > deadline - 3:
            break
        bound = L - ls
        idx = bisect.bisect_left(right_sums_f, bound) - 1
        if idx < 0:
            continue
        rs = right_sums_f[idx]
        rm = right_masks_f[idx]
        total = ls + rs
        if total > best_total and (lm | rm) != 0:
            best_total = total
            best_lm = lm
            best_rm = rm

    selected = []
    for i in range(n1):
        if best_lm & (1 << i):
            selected.append(i)
    for i in range(n2):
        if best_rm & (1 << i):
            selected.append(n1 + i)

    if not selected:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    total_frac = Fraction(0)
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])

    if total_frac >= 1:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    return selected
