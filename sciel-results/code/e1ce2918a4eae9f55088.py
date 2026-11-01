"""Suite task: ternary covering code K_3(6,1). Tetracode-based init + aggressive reduction + ILS."""
import time
import itertools
import numpy as np

_TIME_LIMIT = 140.0
_START = time.time()
_cache = {}


def construct(q, n, radius):
    key = (q, n, radius)
    if key in _cache:
        return [list(w) for w in _cache[key]]
    if (q, n, radius) == (3, 6, 1):
        res = _solve_q3n6r1()
    else:
        res = _generic(q, n, radius)
    _cache[key] = res
    return [list(w) for w in res]


def _generic(q, n, radius):
    if n >= 4:
        base = [[a, b, c, d, (a + b + c + d) % q] for a in range(q)
                for b in range(q) for c in range(q) for d in range(q)]
        base = base[: 2 * (q - 1) * 4 + 1]
        return [h[:4] + list(tail) for h in base
                for tail in itertools.product(range(q), repeat=n - 4)]
    return [list(t) for t in itertools.product(range(q), repeat=n)]


def _solve_q3n6r1():
    q, n = 3, 6
    N = q ** n

    # Precompute the 19-ball (self + 18 single-coordinate neighbours) as flat indices.
    ball = [0]
    for i in range(n):
        for d in range(1, q):
            ball.append(d * (q ** i))
    ball = np.array(ball, dtype=np.int64)

    # Base-3 digits of each flat index, shape (N, n).
    digits = np.zeros((N, n), dtype=np.int64)
    for i in range(n):
        digits[:, i] = (np.arange(N, dtype=np.int64) // (q ** i)) % q

    # Coverage helper: cov[j] = number of selected codewords within the ball of j.
    def add(cov, c):
        idx = c + ball
        cov[idx] += 1

    def remove(cov, c):
        idx = c + ball
        cov[idx] -= 1

    def uncovered_count(cov):
        return int(np.count_nonzero(cov == 0))

    def is_covered(cov):
        return bool(np.all(cov >= 1))

    # Tetracode: the (3,4) ternary code with 9 words (all words of length 4 with sum ≡ 0 mod 3)
    # Actually the ternary Golay-like code: length 4, radius 1, 9 words
    # The ternary Hamming code of length (3^2-1)/2 = 4 has 3^2 = 9 codewords
    # Parity check: sum of digits ≡ 0 mod 3
    tetracode = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                d = (-a - b - c) % 3
                tetracode.append([a, b, c, d])

    # Build the initial code: for each tetracode word t (length 4) and each tail (length 2),
    # place codeword at position (t, tail)
    # This gives 9 * 9 = 81 codewords
    initial = []
    for t in tetracode:
        for x in range(3):
            for y in range(3):
                idx = t[0] * (3**0) + t[1] * (3**1) + t[2] * (3**2) + t[3] * (3**3) + x * (3**4) + y * (3**5)
                initial.append(idx)

    # Greedy repair: add codewords to cover any uncovered positions
    def greedy_repair(cov, selected, max_add=30):
        for _ in range(max_add):
            if time.time() - _START > _TIME_LIMIT - 2:
                break
            uc = uncovered_count(cov)
            if uc == 0:
                break
            gain = np.count_nonzero(cov[ball] == 0)
            best = int(np.argmax(gain))
            if gain[best] <= 0:
                break
            add(cov, best)
            selected.append(best)
        return selected, cov

    def reduce_code(selected, cov):
        """Aggressively remove redundant codewords, multiple passes."""
        for _ in range(10):
            improved = False
            for c in list(selected):
                if time.time() - _START > _TIME_LIMIT - 2:
                    break
                remove(cov, c)
                if is_covered(cov):
                    selected.remove(c)
                    improved = True
                else:
                    add(cov, c)
            if not improved:
                break
        return selected, cov

    def swap_search(selected, cov, max_iter=200):
        """Try 1-for-1 swaps to reduce size."""
        for _ in range(max_iter):
            if time.time() - _START > _TIME_LIMIT - 2:
                break
            improved = False
            for c in list(selected):
                if time.time() - _START > _TIME_LIMIT - 2:
                    break
                remove(cov, c)
                before = uncovered_count(cov)
                # Find best replacement
                g = np.count_nonzero(cov[ball] == 0)
                x = int(np.argmax(g))
                if g[x] > before:
                    add(cov, x)
                    after = uncovered_count(cov)
                    if after == 0:
                        # Commit swap: remove c, add x
                        selected.remove(c)
                        selected.append(x)
                        improved = True
                        break
                    else:
                        remove(cov, x)
                add(cov, c)
            if not improved:
                break
        return selected, cov

    # Main search
    budget = _START + _TIME_LIMIT - 5.0

    # Start with tetracode initialization
    cov = np.zeros(N, dtype=np.int32)
    selected = list(initial)
    for c in selected:
        add(cov, c)

    # Repair if not fully covered
    selected, cov = greedy_repair(cov, selected)

    # Reduce
    selected, cov = reduce_code(selected, cov)

    best = list(selected)
    best_size = len(best)

    # ILS: perturb and re-reduce
    rng = np.random.default_rng(42)
    while time.time() - _START < budget - 3:
        # Perturb: remove 2-4 random codewords
        sel = list(best)
        cov2 = np.zeros(N, dtype=np.int32)
        for c in sel:
            add(cov2, c)

        n_remove = rng.integers(2, 5)
        for _ in range(n_remove):
            if sel:
                idx = int(rng.integers(len(sel)))
                c = sel[idx]
                remove(cov2, c)
                sel.remove(c)

        # Repair
        sel, cov2 = greedy_repair(cov2, sel, max_add=20)

        # Reduce
        sel, cov2 = reduce_code(sel, cov2)

        # Swap
        sel, cov2 = swap_search(sel, cov2, max_iter=50)

        if len(sel) < best_size:
            best = list(sel)
            best_size = len(best)
            # Rebuild cov for best
            cov = np.zeros(N, dtype=np.int32)
            for c in best:
                add(cov, c)

        if best_size <= 73:
            break

    # Final polish: tight reduction
    for c in list(best):
        remove(cov, c)
        if is_covered(cov):
            best.remove(c)
        else:
            add(cov, c)

    return best


if __name__ == "__main__":
    res = construct(3, 6, 1)
    print(len(res))
