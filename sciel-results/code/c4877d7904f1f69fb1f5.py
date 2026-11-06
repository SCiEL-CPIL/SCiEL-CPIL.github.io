"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 82.0  # safety margin before 90s kill
_best = None
_best_score = None

def _square_nonzero_count(exponents, coeffs):
    sq = {}
    n = len(exponents)
    for i in range(n):
        ei = exponents[i]
        ci = coeffs[i]
        for j in range(i, n):
            ej = exponents[j]
            cj = coeffs[j]
            d = ei + ej
            if i == j:
                sq[d] = sq.get(d, 0) + ci * ci
            else:
                sq[d] = sq.get(d, 0) + 2 * ci * cj
    cnt = 0
    for v in sq.values():
        if v != 0:
            cnt += 1
    return cnt

def _validate(exponents, coeffs):
    if len(exponents) != 12 or len(coeffs) != 12:
        return False
    for e, c in zip(exponents, coeffs):
        if c == 0 or not (-64 <= c <= 64) or e < 0 or e > 40:
            return False
    if len(set(exponents)) != 12:
        return False
    return True

def _build_candidate(exponents, coeffs):
    max_deg = max(exponents)
    lst = [0] * (max_deg + 1)
    for e, c in zip(exponents, coeffs):
        lst[e] = c
    while len(lst) > 0 and lst[-1] == 0:
        lst.pop()
    if len(lst) > 41 or not lst or lst[-1] == 0:
        return None
    return lst

def _try_candidate(exponents, coeffs):
    global _best, _best_score
    if not _validate(exponents, coeffs):
        return None
    cnt = _square_nonzero_count(exponents, coeffs)
    if _best_score is None or cnt < _best_score:
        cand = _build_candidate(exponents, coeffs)
        if cand is not None:
            _best = cand
            _best_score = cnt
    return cnt

def _local_search(exponents, coeffs, rng, deadline):
    exponents = list(exponents)
    coeffs = list(coeffs)
    cur_cnt = _square_nonzero_count(exponents, coeffs)
    n = len(exponents)
    while time.time() < deadline:
        improved = False
        moves = []
        for i in range(n):
            for j in range(i + 1, n):
                moves.append(('swap', i, j))
            for m in range(1, 9):
                for s in [-1, 1]:
                    nc = s * m
                    if nc != coeffs[i]:
                        moves.append(('mag', i, nc))
        rng.shuffle(moves)
        for move in moves:
            if time.time() >= deadline:
                return exponents, coeffs, cur_cnt
            if move[0] == 'swap':
                _, i, j = move
                exponents[i], exponents[j] = exponents[j], exponents[i]
                coeffs[i], coeffs[j] = coeffs[j], coeffs[i]
                new_cnt = _square_nonzero_count(exponents, coeffs)
                if new_cnt < cur_cnt:
                    cur_cnt = new_cnt
                    improved = True
                else:
                    exponents[i], exponents[j] = exponents[j], exponents[i]
                    coeffs[i], coeffs[j] = coeffs[j], coeffs[i]
            elif move[0] == 'mag':
                _, i, nc = move
                old = coeffs[i]
                coeffs[i] = nc
                new_cnt = _square_nonzero_count(exponents, coeffs)
                if new_cnt < cur_cnt:
                    cur_cnt = new_cnt
                    improved = True
                else:
                    coeffs[i] = old
        if not improved:
            break
    return exponents, coeffs, cur_cnt

def construct():
    global _best, _best_score
    rng = random.Random(42)
    deadline = _DEADLINE
    # Seed with known structure: consecutive exponents
    base_exps = list(range(12))
    base_coeffs = [1] * 12
    _try_candidate(base_exps, base_coeffs)
    # Random search with local refinement
    trials = 0
    while time.time() < deadline - 8.0:
        exponents = sorted(rng.sample(range(0, 41), 12))
        coeffs = []
        for _ in range(12):
            m = rng.randint(1, 6)
            coeffs.append(rng.choice([-1, 1]) * m)
        cnt = _try_candidate(exponents, coeffs)
        trials += 1
        if trials % 30 == 0 and time.time() < deadline - 5.0:
            exponents, coeffs, lc = _local_search(exponents, coeffs, rng, deadline - 3.0)
            _try_candidate(exponents, coeffs)
    # Try structured exponent patterns
    if time.time() < deadline - 4.0:
        patterns = [
            list(range(12)),
            list(range(0, 24, 2)),
            [0,1,2,3,4,5,35,36,37,38,39,40],
            [0,1,3,4,6,7,9,10,12,13,15,16],
            [0,2,4,6,8,10,11,13,15,17,19,21],
            [0,1,2,4,5,7,8,10,11,13,14,16],
        ]
        for exps in patterns:
            if time.time() >= deadline - 4.0:
                break
            if len(exps) != 12:
                continue
            for trial in range(30):
                if time.time() >= deadline - 4.0:
                    break
                coeffs = []
                for _ in range(12):
                    m = rng.randint(1, 6)
                    coeffs.append(rng.choice([-1, 1]) * m)
                _try_candidate(exps, coeffs)
                if trial % 10 == 0 and time.time() < deadline - 2.0:
                    exponents, coeffs, lc = _local_search(exps, coeffs, rng, deadline - 1.0)
                    _try_candidate(exponents, coeffs)
    if _best is None:
        return [1] * 12
    return _best
# EVOLVE-BLOCK-END
