"""Local-search engine for pairwise-coprime selection maximizing sum 1/(512-a).
Uses prime-factor-set conflict detection for speed, enabling deeper search."""
import time
from math import gcd

N = 512
DEADLINE = 140.0  # safety margin below the 150 s kill


def _precompute_prime_factors():
    """Precompute prime factor sets for 1..511."""
    pf = [set() for _ in range(N)]
    for i in range(1, N):
        n = i
        d = 2
        while d * d <= n:
            while n % d == 0:
                pf[i].add(d)
                n //= d
            d += 1
        if n > 1:
            pf[i].add(n)
    return pf


# Precompute once at module level
_PF = _precompute_prime_factors()


def _is_coprime_with_all(a, chosen_set):
    """Check if a is coprime with all elements in chosen_set using prime factor sets."""
    fa = _PF[a]
    if not fa:
        return True
    for b in chosen_set:
        if fa & _PF[b]:
            return False
    return True


def _score(chosen):
    return sum(1.0 / (N - a) for a in chosen)


def _greedy_from(chosen, blocked, deadline):
    """Greedy fill from the current chosen set, skipping `blocked`,
    adding any coprime candidate with positive weight. Returns new list."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    for a in range(N - 1, 0, -1):
        if time.time() > deadline:
            break
        if a in chosen_set or a in blocked:
            continue
        if _is_coprime_with_all(a, chosen_set):
            chosen.append(a)
            chosen_set.add(a)
    return chosen


def _multi_swap_search(best, best_score, deadline):
    """Try removing 2 elements and re-greedy, which can escape 1-swap local optima."""
    improved = False
    n = len(best)
    for i in range(n):
        if time.time() > deadline:
            return best, best_score, improved
        for j in range(i + 1, n):
            if time.time() > deadline:
                return best, best_score, improved
            removed = {best[i], best[j]}
            rest = [x for k, x in enumerate(best) if k != i and k != j]
            cand = _greedy_from(rest, removed, deadline)
            s = _score(cand)
            if s > best_score + 1e-15:
                best, best_score = cand, s
                improved = True
                return best, best_score, improved
    return best, best_score, improved


def construct():
    deadline = time.time() + DEADLINE

    # --- seed: greedy from high a (matches the reference, but as a starting point) ---
    chosen = []
    chosen_set = set()
    for a in range(N - 1, 0, -1):
        if _is_coprime_with_all(a, chosen_set):
            chosen.append(a)
            chosen_set.add(a)

    best = chosen
    best_score = _score(best)

    # --- 1-move add: try adding any coprime candidate ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        chosen_set = set(best)
        for a in range(N - 1, 0, -1):
            if a in chosen_set:
                continue
            if _is_coprime_with_all(a, chosen_set):
                cand = best + [a]
                s = _score(cand)
                if s > best_score + 1e-15:
                    best, best_score = cand, s
                    improved = True
                    break

    # --- 1-move remove + re-greedy: freeing a chosen element may unlock
    #     multiples whose combined weight beats the removed element ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in best:
            if time.time() > deadline:
                break
            rest = [x for x in best if x != c]
            cand = _greedy_from(rest, {c}, deadline)
            s = _score(cand)
            if s > best_score + 1e-15:
                best, best_score = cand, s
                improved = True
                break

    # --- 2-move swap: remove a chosen element, re-greedy, then try adding a
    #     previously-blocked candidate (escapes greedy's myopic skips) ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        base_set = set(best)
        for c in best:
            if time.time() > deadline:
                break
            rest = [x for x in best if x != c]
            rest_set = set(rest)
            # candidates that were blocked by c (not coprime to c) and not in rest
            candidates = [a for a in range(N - 1, 0, -1)
                          if a not in rest_set and _PF[a] & _PF[c]]
            for a in candidates:
                if time.time() > deadline:
                    break
                if not _is_coprime_with_all(a, rest_set):
                    continue
                seed = rest + [a]
                seed_set = set(seed)
                cand = _greedy_from(seed, set(), deadline)
                s = _score(cand)
                if s > best_score + 1e-15:
                    best, best_score = cand, s
                    improved = True
                    break
            if improved:
                break

    # --- multi-swap: remove 2 elements, re-greedy ---
    improved = True
    while improved and time.time() < deadline:
        best, best_score, improved = _multi_swap_search(best, best_score, deadline)

    # --- final polish: one more combined add/remove sweep ---
    for _ in range(5):
        if time.time() > deadline:
            break
        # add
        chosen_set = set(best)
        for a in range(N - 1, 0, -1):
            if a in chosen_set:
                continue
            if _is_coprime_with_all(a, chosen_set):
                cand = best + [a]
                s = _score(cand)
                if s > best_score + 1e-15:
                    best, best_score = cand, s
        # remove
        for c in best:
            if time.time() > deadline:
                break
            rest = [x for x in best if x != c]
            cand = _greedy_from(rest, {c}, deadline)
            s = _score(cand)
            if s > best_score + 1e-15:
                best, best_score = cand, s
        # multi-swap
        if time.time() < deadline:
            best, best_score, _ = _multi_swap_search(best, best_score, deadline)

    return best
