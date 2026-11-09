import time
import random
from collections import defaultdict

# Precompute prime factorization vectors for all numbers 2..128
def _factorize(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

_NUM_FACTORS = {}
for _i in range(2, 129):
    _factors = _factorize(_i)
    _cnt = defaultdict(int)
    for _p in _factors:
        _cnt[_p] += 1
    _NUM_FACTORS[_i] = tuple(sorted(_cnt.items()))

def _add_vectors(v1, v2):
    if not v1:
        return v2
    if not v2:
        return v1
    d = defaultdict(int)
    for p, e in v1:
        d[p] += e
    for p, e in v2:
        d[p] += e
    return tuple(sorted(d.items()))

def _make_key(v):
    return frozenset(v) if v else frozenset()

def is_valid(seq):
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    best_seq = []

    def backtrack(current_seq, products_set, suffix_vecs):
        """
        current_seq: list of chosen values so far
        products_set: set of frozensets representing all block products
        suffix_vecs: list where suffix_vecs[k] is the prime-exponent vector of product of current_seq[k:]
                     suffix_vecs has len(current_seq)+1 entries, with suffix_vecs[len(current_seq)] = ()
        """
        nonlocal best_seq
        if time.time() > deadline:
            return
        if len(current_seq) > len(best_seq):
            best_seq = list(current_seq)

        # Try to add more elements
        for x in range(current_seq[-1] + 1, 129) if current_seq else range(2, 129):
            if time.time() > deadline:
                return

            # New block products: [x], [x, last], [x, last, second_last], ...
            # These correspond to x * suffix_vecs[k] for k = len(current_seq), len-1, ..., 0
            # i.e., x alone, x*last, x*last*second_last, ...
            x_vec = _NUM_FACTORS[x]
            new_keys = []
            # x alone
            new_keys.append(_make_key(x_vec))
            # x * suffix for each suffix
            for k in range(len(current_seq) - 1, -1, -1):
                # suffix_vecs[k] is product of current_seq[k:]
                combined = _add_vectors(x_vec, suffix_vecs[k])
                new_keys.append(_make_key(combined))

            # Check for conflicts
            conflict = False
            for key in new_keys:
                if key in products_set:
                    conflict = True
                    break
            if conflict:
                continue

            # No conflict, proceed
            new_products_set = products_set | set(new_keys)
            new_suffix_vecs = []
            # new_suffix_vecs[0] = x * product(current_seq) = x_vec + suffix_vecs[0]
            # new_suffix_vecs[1] = x * product(current_seq[1:]) = x_vec + suffix_vecs[1]
            # ...
            # new_suffix_vecs[len(current_seq)] = x_vec
            for k in range(len(current_seq) + 1):
                if k == len(current_seq):
                    new_suffix_vecs.append(x_vec)
                else:
                    new_suffix_vecs.append(_add_vectors(x_vec, suffix_vecs[k]))

            backtrack(current_seq + [x], new_products_set, new_suffix_vecs)

    # Initial backtracking
    backtrack([], set(), [()])

    # Local search: try to insert missing values
    improved = True
    max_iter = 0
    while improved and time.time() < deadline and max_iter < 200:
        improved = False
        max_iter += 1
        candidates = [x for x in range(2, 129) if x not in best_seq]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            # Find valid insertion positions
            lo = 0
            while lo < len(best_seq) and best_seq[lo] < x:
                lo += 1
            hi = len(best_seq)
            while hi > 0 and best_seq[hi - 1] > x:
                hi -= 1
            for i in range(hi, lo - 1, -1):
                new_seq = best_seq[:i] + [x] + best_seq[i:]
                if is_valid(new_seq):
                    best_seq = new_seq
                    improved = True
                    break
            if improved:
                break

    # Try removing and re-adding elements
    if time.time() < deadline:
        for _ in range(30):
            if time.time() > deadline:
                break
            for i in range(len(best_seq)):
                if time.time() > deadline:
                    break
                removed = best_seq[i]
                without = best_seq[:i] + best_seq[i+1:]
                missing = [x for x in range(2, 129) if x not in without]
                random.shuffle(missing)
                improved_here = False
                for x in missing:
                    if time.time() > deadline:
                        break
                    lo = 0
                    while lo < len(without) and without[lo] < x:
                        lo += 1
                    hi = len(without)
                    while hi > 0 and without[hi - 1] > x:
                        hi -= 1
                    for j in range(hi, lo - 1, -1):
                        new_seq = without[:j] + [x] + without[j:]
                        if is_valid(new_seq) and len(new_seq) > len(best_seq):
                            best_seq = new_seq
                            improved_here = True
                            break
                    if improved_here:
                        break
                if not improved_here and len(without) + 1 > len(best_seq):
                    # Put removed back if it was better
                    pass

    # Final safety check
    result = []
    for v in best_seq:
        if not result or v > result[-1]:
            result.append(v)
    if result and is_valid(result):
        return result
    return [2]
