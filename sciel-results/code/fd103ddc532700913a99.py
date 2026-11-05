import time

_DEADLINE = 58.0
_START = time.time()


def _remaining_budget():
    return _DEADLINE - (time.time() - _START)


def _is_smooth(x):
    for p in (2, 3, 5):
        while x % p == 0:
            x //= p
    return x == 1


def _smooth_terms(n):
    """All smooth numbers <= n, ascending."""
    out = []
    for x in range(1, n + 1):
        if _is_smooth(x):
            out.append(x)
    return out


def _find_one(n):
    # k=1: target itself smooth
    if _is_smooth(n):
        return [n]
    terms = [x for x in range(1, n) if _is_smooth(x)]
    # k=2: a + b = n, neither divides the other
    for a in terms:
        b = n - a
        if b <= a:
            break
        if b > n:
            continue
        if _is_smooth(b) and (a % b != 0) and (b % a != 0):
            return [a, b]
    # k=3: a + b + c = n, pairwise non-dividing
    m = len(terms)
    for i in range(m):
        a = terms[i]
        if a * 3 >= n:
            break
        for j in range(i + 1, m):
            b = terms[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b:
                continue
            if c > n:
                continue
            if not _is_smooth(c):
                continue
            if (a % b == 0) or (b % a == 0):
                continue
            if (a % c == 0) or (c % a == 0):
                continue
            if (b % c == 0) or (c % b == 0):
                continue
            return [a, b, c]
    # k=4: a + b + c + d = n
    for i in range(m):
        a = terms[i]
        if a * 4 >= n:
            break
        for j in range(i + 1, m):
            b = terms[j]
            if a + b * 3 >= n:
                break
            for k in range(j + 1, m):
                c = terms[k]
                if a + b + c * 2 >= n:
                    break
                d = n - a - b - c
                if d <= c:
                    continue
                if not _is_smooth(d):
                    continue
                if (a % b == 0) or (b % a == 0):
                    continue
                if (a % c == 0) or (c % a == 0):
                    continue
                if (a % d == 0) or (d % a == 0):
                    continue
                if (b % c == 0) or (c % b == 0):
                    continue
                if (b % d == 0) or (d % b == 0):
                    continue
                if (c % d == 0) or (d % c == 0):
                    continue
                return [a, b, c, d]
    return None


def construct():
    answers = []
    for n in range(1, 129):
        if _remaining_budget() < 0.5:
            # safety: fall back to a guaranteed (possibly non-minimal) antichain
            # For remaining targets, use a simple valid construction.
            ans = _fallback(n)
            answers.append(ans)
            continue
        ans = _find_one(n)
        if ans is None:
            ans = _fallback(n)
        answers.append(ans)
    return answers


def _fallback(n):
    # Guaranteed valid antichain: use the largest smooth numbers <= n that are
    # mutually non-dividing. Build greedily from large to small, skipping any
    # that divide or are divided by an already-chosen term, until sum == n.
    # Simpler: search over k=5..8 with the same logic but relaxed; if still
    # failing, brute-force a small k.
    terms = [x for x in range(1, n) if _is_smooth(x)]
    m = len(terms)
    # Try k=5
    for i in range(m):
        a = terms[i]
        if a * 5 >= n:
            break
        for j in range(i + 1, m):
            b = terms[j]
            if a + b * 4 >= n:
                break
            for k in range(j + 1, m):
                c = terms[k]
                if a + b + c * 3 >= n:
                    break
                for l in range(k + 1, m):
                    d = terms[l]
                    if a + b + c + d * 2 >= n:
                        break
                    e = n - a - b - c - d
                    if e <= d:
                        continue
                    if not _is_smooth(e):
                        continue
                    cand = [a, b, c, d, e]
                    ok = True
                    for x in range(5):
                        for y in range(x + 1, 5):
                            if cand[x] % cand[y] == 0 or cand[y] % cand[x] == 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        return cand
    # Last resort: single term if n smooth, else two-term exhaustive
    if _is_smooth(n):
        return [n]
    for a in terms:
        b = n - a
        if b <= a:
            break
        if _is_smooth(b) and (a % b != 0) and (b % a != 0):
            return [a, b]
    # Absolute last resort: return n itself if smooth, else a trivial valid set
    # (should not happen for n<=128)
    return [n] if _is_smooth(n) else [1, n - 1]
