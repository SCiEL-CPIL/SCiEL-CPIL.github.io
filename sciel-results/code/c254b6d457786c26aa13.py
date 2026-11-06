"""Rewritten program for divisibility-antichain smooth number representations."""
# EVOLVE-BLOCK-START
import time

def _is_smooth(x):
    """Check if x is a 2,3,5-smooth number."""
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _precompute_smooth(max_val):
    """Precompute all smooth numbers up to max_val."""
    smooth = []
    for x in range(1, max_val + 1):
        if _is_smooth(x):
            smooth.append(x)
    return smooth

def _is_antichain(terms):
    """Check if no term divides another."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _find_1_term(n, smooth_set):
    """Try 1-term solution."""
    if n in smooth_set:
        return [n]
    return None

def _find_2_terms(n, smooth, smooth_set):
    """Try 2-term solution."""
    for a in smooth:
        if a >= n:
            break
        b = n - a
        if b <= 0 or b == a:
            continue
        if b not in smooth_set:
            continue
        # Check antichain: a ∤ b and b ∤ a
        if b % a == 0 or a % b == 0:
            continue
        return [a, b]
    return None

def _find_3_terms(n, smooth, smooth_set, deadline):
    """Try 3-term solution."""
    smooth_leq_n = [x for x in smooth if x < n]
    m = len(smooth_leq_n)
    for i in range(m):
        if time.time() > deadline:
            return None
        a = smooth_leq_n[i]
        for j in range(i + 1, m):
            b = smooth_leq_n[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b:
                continue
            if c not in smooth_set:
                continue
            # Check antichain for all pairs
            if c % a == 0 or a % c == 0:
                continue
            if c % b == 0 or b % c == 0:
                continue
            if b % a == 0 or a % b == 0:
                continue
            return [a, b, c]
    return None

def _find_4_terms(n, smooth, smooth_set, deadline):
    """Try 4-term solution."""
    smooth_leq_n = [x for x in smooth if x < n]
    m = len(smooth_leq_n)
    for i in range(m):
        if time.time() > deadline:
            return None
        a = smooth_leq_n[i]
        for j in range(i + 1, m):
            b = smooth_leq_n[j]
            if a + b >= n:
                break
            for k in range(j + 1, m):
                c = smooth_leq_n[k]
                if a + b + c >= n:
                    break
                d = n - a - b - c
                if d <= c:
                    continue
                if d not in smooth_set:
                    continue
                terms = [a, b, c, d]
                if _is_antichain(terms):
                    return terms
    return None

def _fallback_solution(n, smooth, smooth_set):
    """Find any valid antichain representation for n (less optimal)."""
    for count in range(1, min(31, n + 1)):
        smooth_leq_n = [x for x in smooth if x <= n]
        if len(smooth_leq_n) < count:
            continue
        from itertools import combinations
        for terms in combinations(smooth_leq_n, count):
            if sum(terms) == n and _is_antichain(terms):
                return list(terms)
    # Last resort
    if n in smooth_set:
        return [n]
    return [1, n - 1] if (n - 1) in smooth_set else [n]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    smooth = _precompute_smooth(128)
    smooth_set = set(smooth)
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            answer = _fallback_solution(n, smooth, smooth_set)
            answers.append(answer)
            continue
        
        answer = _find_1_term(n, smooth_set)
        if answer is None:
            answer = _find_2_terms(n, smooth, smooth_set)
        if answer is None:
            answer = _find_3_terms(n, smooth, smooth_set, deadline)
        if answer is None:
            answer = _find_4_terms(n, smooth, smooth_set, deadline)
        if answer is None:
            answer = _fallback_solution(n, smooth, smooth_set)
        answers.append(answer)
    
    return answers
# EVOLVE-BLOCK-END
