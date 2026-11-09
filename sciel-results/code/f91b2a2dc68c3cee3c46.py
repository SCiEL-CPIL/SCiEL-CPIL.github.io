"""Rewritten: bounded ascending DFS with exact fractions to minimize largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

_TARGET = Fraction(5, 31)
_DEADLINE_MARGIN = 12.0  # stop this many seconds before the 90s kill

def _deadline():
    return time.time() + _DEADLINE_MARGIN

def _greedy(target):
    r = target
    vals = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        vals.append(d)
        r -= Fraction(1, d)
    return vals

def _valid(vals, target):
    if not vals or len(vals) > 256:
        return False
    if len(set(vals)) != len(vals):
        return False
    if any(d < 1 or d > 10_000_000 for d in vals):
        return False
    s = Fraction(0, 1)
    for d in vals:
        s += Fraction(1, d)
    return s == target

def _max_sum_from(start, count):
    total = Fraction(0, 1)
    for i in range(start + 1, start + 1 + count):
        total += Fraction(1, i)
    return total

def _dfs(target, last, max_terms, max_den, chosen, best):
    if target == 0:
        return list(chosen)
    if max_terms <= 0 or target < 0:
        return None
    lo = last + 1
    if target.numerator > 0:
        need = (target.denominator + target.numerator - 1) // target.numerator
        if need > lo:
            lo = need
    if lo > max_den:
        return None
    ub = _max_sum_from(lo - 1, max_terms)
    if ub < target:
        return None
    if target.denominator % target.numerator == 0:
        d = target.denominator // target.numerator
        if last < d <= max_den:
            cand = list(chosen) + [d]
            if max(cand) < best[0]:
                best[0] = max(cand)
                best[1] = list(cand)
            return cand
    for d in range(lo, max_den + 1):
        if time.time() > _deadline():
            return None
        rem = target - Fraction(1, d)
        if rem < 0:
            continue
        if rem == 0:
            cand = list(chosen) + [d]
            if max(cand) < best[0]:
                best[0] = max(cand)
                best[1] = list(cand)
            return cand
        if max_terms - 1 <= 0:
            continue
        ub2 = _max_sum_from(d, max_terms - 1)
        if ub2 < rem:
            continue
        chosen.append(d)
        res = _dfs(rem, d, max_terms - 1, max_den, chosen, best)
        chosen.pop()
        if res is not None:
            m = max(res)
            if m < best[0]:
                best[0] = m
                best[1] = list(res)
            return res
    return None

def construct():
    target = _TARGET
    g = _greedy(target)
    best_vals = g
    best_max = max(g)

    term_budgets = [6, 8, 10, 12, 16, 24, 40]
    step = max(1, best_max // 200)
    C = best_max - 1
    attempts = 0
    max_attempts = 400000
    while C > 1 and attempts < max_attempts:
        if time.time() > _deadline():
            break
        for tb in term_budgets:
            if time.time() > _deadline():
                break
            best = [C, None]
            res = _dfs(target, 0, tb, C, [], best)
            attempts += 1
            if res is not None and max(res) <= C and _valid(res, target):
                if max(res) < best_max:
                    best_max = max(res)
                    best_vals = res
                break
        C -= step

    if not _valid(best_vals, target):
        best_vals = _greedy(target)
    if len(best_vals) > 256:
        best_vals = _greedy(target)
    return best_vals
# EVOLVE-BLOCK-END
