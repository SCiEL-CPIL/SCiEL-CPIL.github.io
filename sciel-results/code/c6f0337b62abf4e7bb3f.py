# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_T1 = time.time()
_HARD = _T1 + 150.0


def _deadline(margin):
    return (time.time() + margin) < _HARD


def _fallback_base(q, n, radius):
    """Direct-sum of tetracode: 9 * q^(n-4) words, radius-1 for n>=4, q=3."""
    if q != 3 or n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tail_len = n - 4
    out = []
    for tail in itertools.product(range(q), repeat=tail_len):
        for h in tetracode:
            out.append(h + list(tail))
    return out


def _neighbors_int(q, n):
    """Return (N, powers, neigh) where neigh[w] are all words within Hamming dist <=1 of w,
    each encoded as a base-q integer in 0..N-1."""
    N = q ** n
    powers = [q ** i for i in range(n)]
    neigh_size = 1 + n * (q - 1)
    neigh = np.zeros((N, neigh_size), dtype=np.int32)
    for w in range(N):
        neigh[w, 0] = w
        idx = 1
        for i in range(n):
            d = (w // powers[i]) % q
            for delta in range(1, q):
                new_d = (d + delta) % q
                neigh[w, idx] = w + (new_d - d) * powers[i]
                idx += 1
    return N, powers, neigh


def _int_to_list(w, q, n):
    out = []
    for _ in range(n):
        out.append(w % q)
        w //= q
    return out


def _list_to_int(cw, q, n):
    v = 0
    for i in range(n):
        v += cw[i] * (q ** i)
    return v


def _greedy_init(N, neigh, rng):
    """Greedy set cover over the whole space, vectorized."""
    uncovered = np.ones(N, dtype=np.bool_)
    code = set()
    while np.any(uncovered):
        gains = uncovered[neigh].sum(axis=1)
        max_gain = int(np.max(gains))
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        w = int(rng.choice(candidates))
        code.add(w)
        uncovered[neigh[w, :]] = False
    return code


def _local_search(code_set, N, neigh, rng, deadline):
    """Deletion + swap local search using vectorized coverage."""
    code_set = set(code_set)

    while time.time() < deadline:
        # --- deletion pass ---
        cov = np.zeros(N, dtype=np.int32)
        for c in code_set:
            cov[neigh[c, :]] += 1

        redundant = [c for c in code_set if int(np.min(cov[neigh[c, :]])) > 1]
        if redundant:
            for c in redundant:
                code_set.remove(c)
            continue  # coverage changed, recompute

        # --- swap pass ---
        improved = False
        c_list = list(code_set)
        rng.shuffle(c_list)

        for c in c_list:
            if time.time() > deadline:
                break

            c_neighbors = neigh[c, :]
            essential = c_neighbors[cov[c_neighbors] == 1]
            if len(essential) == 0:
                continue

            if len(essential) == 1:
                cand_set = set(neigh[int(essential[0]), :].tolist())
            else:
                cand_set = set(neigh[int(essential[0]), :].tolist())
                for ew in essential[1:]:
                    cand_set.intersection_update(neigh[int(ew), :].tolist())

            valid = [d for d in cand_set if d not in code_set]
            if not valid:
                continue

            d = int(rng.choice(valid))
            code_set.remove(c)
            code_set.add(d)
            improved = True
            break  # restart deletion pass with new set

        if not improved:
            break  # local optimum, let caller restart

    return code_set


def construct(q, n, radius):
    global _HARD
    _HARD = time.time() + 150.0
    if n < 0:
        return []
    if n == 0:
        return [[]]

    N = q ** n

    # Small / trivial cases: return a guaranteed-valid full-space cover.
    if N <= 4096 or radius >= n:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    powers = [q ** i for i in range(n)]
    neigh_size = 1 + n * (q - 1)
    neigh = np.zeros((N, neigh_size), dtype=np.int32)
    for w in range(N):
        neigh[w, 0] = w
        idx = 1
        for i in range(n):
            d = (w // powers[i]) % q
            for delta in range(1, q):
                new_d = (d + delta) % q
                neigh[w, idx] = w + (new_d - d) * powers[i]
                idx += 1

    # --- guaranteed-valid base code (tetracode for q=3,n>=4, else full space) ---
    base_list = _fallback_base(q, n, radius)
    base_set = set(_list_to_int(cw, q, n) for cw in base_list)

    best_code = base_set
    best_size = len(base_set)

    # --- randomized restarts: greedy init + local search ---
    deadline_search = _HARD - 20.0
    for i in range(30):
        if time.time() > deadline_search:
            break
        rng = np.random.default_rng(1000 + i)

        # occasionally seed from the structured base to bias the search
        if i % 5 == 0 and i > 0:
            seed = set(base_set)
        else:
            seed = set()

        code = _greedy_init(N, neigh, rng)
        if seed:
            code = code | seed
        code = _local_search(code, N, neigh, rng, deadline_search)

        if len(code) < best_size:
            best_size = len(code)
            best_code = code

        if best_size <= 486:
            break

    # --- guarantee validity: repair any uncovered words ---
    code = set(best_code)
    cov = np.zeros(N, dtype=np.int32)
    for c in code:
        cov[neigh[c, :]] += 1
    uncovered = np.where(cov == 0)[0]
    while len(uncovered) > 0 and _deadline(15.0):
        uc = int(uncovered[0])
        if uc not in code:
            code.add(uc)
            cov[neigh[uc, :]] += 1
        uncovered = np.where(cov == 0)[0]

    result = [_int_to_list(int(w), q, n) for w in code]
    return result
# EVOLVE-BLOCK-END
