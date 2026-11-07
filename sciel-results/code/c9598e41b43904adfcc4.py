import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 80.0


def _factorize(n, primes):
    """Return exponent vector of n with respect to primes list."""
    vec = []
    for p in primes:
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        vec.append(e)
    return tuple(vec)


def _can_add_vec(vecs, new_vec, prods):
    """Check if adding a vector keeps all subset products distinct.
    Uses the set of existing product vectors for efficiency."""
    for pv in prods:
        # new product = pv + new_vec
        new_pv = tuple(pv[i] + new_vec[i] for i in range(len(pv)))
        if new_pv in prods:
            return False
    return True


def construct():
    if time.time() > _DEADLINE:
        return [2]

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    nprimes = len(primes)

    # Precompute exponent vectors for all numbers 2..40
    all_vecs = {}
    for n in range(2, 41):
        all_vecs[n] = _factorize(n, primes)

    # All 12 primes are always mutually compatible (each has a unique prime factor)
    # Start with all primes
    prime_vecs = [all_vecs[p] for p in primes]

    # Build initial product set for primes
    prods = {tuple([0] * nprimes)}
    for v in prime_vecs:
        new_prods = set(prods)
        for pv in prods:
            new_pv = tuple(pv[i] + v[i] for i in range(nprimes))
            new_prods.add(new_pv)
        prods = new_prods

    # Candidate composites (non-prime numbers 2..40)
    candidates = [n for n in range(2, 41) if n not in primes]
    candidate_vecs = [all_vecs[n] for n in candidates]

    # Pre-check: which candidates can be added to the prime set?
    # A candidate v can be added iff for all p in prods, p+v not in prods
    # Since prods contains all square-free products (exponents 0 or 1),
    # and v has some exponent >= 2, p+v will have an exponent >= 2,
    # so p+v cannot be in prods. Thus ALL candidates pass this check.
    # But after adding some candidates, conflicts arise.

    # Strategy: branch-and-bound on candidates
    # Sort candidates to prioritize those that are "more independent"
    # (have larger max exponent or more non-zero components)
    cand_order = sorted(range(len(candidates)),
                        key=lambda i: (-max(candidate_vecs[i]), -sum(candidate_vecs[i])))

    best_count = nprimes
    best_selection = list(primes)

    def _search(start_idx, current_prods, current_selected, remaining_budget):
        nonlocal best_count, best_selection
        if time.time() > _DEADLINE - 5:
            return
        if len(current_selected) > best_count:
            best_count = len(current_selected)
            best_selection = list(current_selected)

        if remaining_budget == 0:
            return

        # Try adding each remaining candidate
        for i in range(start_idx, len(candidates)):
            if time.time() > _DEADLINE - 5:
                return
            v = candidate_vecs[i]
            n = candidates[i]

            # Check if v can be added
            conflict = False
            for pv in current_prods:
                new_pv = tuple(pv[j] + v[j] for j in range(nprimes))
                if new_pv in current_prods:
                    conflict = True
                    break
            if conflict:
                continue

            # Add it
            new_prods = set(current_prods)
            for pv in current_prods:
                new_pv = tuple(pv[j] + v[j] for j in range(nprimes))
                new_prods.add(new_pv)

            new_selected = current_selected + [n]
            _search(i + 1, new_prods, new_selected, remaining_budget - 1)

    # We want to find as many composites as possible to add to the 12 primes
    # Max possible: 18 total, so up to 6 composites
    # Try to find 6 first, then 5, etc.
    for target in range(6, 0, -1):
        if time.time() > _DEADLINE - 10:
            break
        # Reset best for this target level
        _search(0, prods, list(primes), target)
        if best_count >= nprimes + target:
            break

    result = best_selection[:best_count]
    # Ensure all are Python ints
    return [int(x) for x in result]
