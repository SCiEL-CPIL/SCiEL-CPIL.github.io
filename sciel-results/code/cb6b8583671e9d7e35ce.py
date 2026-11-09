"""Rewritten program: fast GF(2) linear-algebra search over an expanding window."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = P['n']                # 210
    max_val = P['maximum_window']  # 10210
    max_size = P['maximum_size']   # 100

    # ---------- prime sieve up to max_val ----------
    def sieve(limit):
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0:2] = b'\x00\x00'
        for i in range(2, int(limit ** 0.5) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes = sieve(max_val)
    num_primes = len(primes)
    prime_index = {p: i for i, p in enumerate(primes)}

    # ---------- parity bitmasks ----------
    # parity_mask[m] = bitmask of primes with ODD exponent in m
    def compute_parity_mask(m):
        mask = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                exp = 0
                while m % p == 0:
                    m //= p
                    exp += 1
                if exp & 1:
                    mask |= (1 << prime_index[p])
        if m > 1:
            # m is a prime factor (possibly > sqrt(original))
            mask |= (1 << prime_index[m])
        return mask

    parity = [0] * (max_val + 1)
    for m in range(2, max_val + 1):
        parity[m] = compute_parity_mask(m)

    target = compute_parity_mask(n)  # parity of 210 = primes {2,3,5,7}

    # ---------- exact square check ----------
    def is_square_big(x):
        r = math.isqrt(x)
        return r * r == x

    # ---------- verify ----------
    def verify(solution):
        if not isinstance(solution, (list, tuple)):
            return False
        if len(solution) < 1 or len(solution) > max_size:
            return False
        if len(set(solution)) != len(solution):
            return False
        for x in solution:
            if isinstance(x, bool) or not isinstance(x, int):
                return False
            if x <= n or x > max_val:
                return False
        prod = n
        for x in solution:
            prod *= x
        return is_square_big(prod)

    # Fallback known solution: [840]  (840*210 = 176400 = 420^2)
    best_solution = [840]
    best_score = 840 - n  # 630

    # ---------- GF(2) linear algebra helpers ----------
    def bit_length_mask(mask):
        return mask.bit_length() - 1 if mask else -1

    def gauss_min_subset(masks):
        """Given list of bitmasks, find minimum-cardinality subset whose XOR == target.
        Returns (size, indices) or (None, None) if unsolvable."""
        # Build rows: (mask, index)
        # We want a representation of target as XOR of a subset.
        # Standard approach: do Gaussian elimination keeping track of which original
        # vectors combine to form each basis vector, then express target.
        # To minimize cardinality, we use the standard trick: process so that each
        # pivot row is the combination with the fewest elements (use the row-reduction
        # that minimizes support via a greedy / minimum-weight representation).

        # Simpler robust method: since we only need existence with size<=max_size and
        # we want the *minimum* size, we do full elimination and then greedily reduce.
        #
        # Approach:
        # 1. Maintain a list of (mask, combo_set) where combo_set is the set of original
        #    indices that XOR to mask.
        # 2. For each new mask, try to reduce it by existing pivots; if it reduces to 0,
        #    we found a dependency (not useful for target).
        # 3. After building a basis, express target in terms of basis rows; the combo
        #    sets of those rows XOR together to give a solution.
        # 4. To minimize cardinality, use the "minimum weight" representation: after
        #    elimination, for each basis row, if it contains the pivot bit and we can
        #    reduce it using other rows to lower its weight, do so.

        # We'll implement a clean version:
        # basis: dict pivot_bit -> (mask, frozenset(indices))
        basis = {}
        # For each input, reduce
        for idx, msk in enumerate(masks):
            cur = msk
            combo = {idx}
            # reduce using existing pivots
            changed = True
            while cur != 0 and changed:
                changed = False
                pb = bit_length_mask(cur)
                if pb in basis:
                    bmask, bcombo = basis[pb]
                    cur ^= bmask
                    combo ^= bcombo
            if cur == 0:
                # dependency: combo XORs to 0; ignore for target solving
                continue
            pb = bit_length_mask(cur)
            basis[pb] = (cur, combo)

        # Now express target
        cur = target
        combo = set()
        if cur != 0:
            # reduce target by basis
            while cur != 0:
                pb = bit_length_mask(cur)
                if pb not in basis:
                    return None, None  # unsolvable
                bmask, bcombo = basis[pb]
                cur ^= bmask
                combo ^= bcombo
            # cur should be 0 now
        # combo is a solution; minimize its size by reducing with basis rows
        # (optional polish): try to reduce combo weight using each basis row
        # If basis row r (mask mr, combo cr) satisfies (combo XOR cr) has smaller
        # weight AND mr's pivot is in combo's span... Actually simplest: re-express.
        # We'll do a light greedy: for each basis pivot, if removing it from combo
        # (XOR with its combo) reduces total weight, do it.
        for pb in sorted(basis.keys()):
            bmask, bcombo = basis[pb]
            new_combo = combo ^ bcombo
            if len(new_combo) < len(combo):
                combo = new_combo
        return len(combo), sorted(combo)

    # ---------- main search: sweep M upward ----------
    # We process numbers in increasing order. At each M, the candidate set is
    # {211, 212, ..., M}. We maintain the parity list already computed.
    # For each M, run gauss_min_subset on masks of 211..M.
    # This is O(M * (cost of elimination)). To keep it fast, we stop as soon as
    # we find a solution with size <= max_size.

    # To avoid re-running elimination from scratch each M (which would be O(M^2) total),
    # we do incremental elimination: maintain basis as we add each new number.
    # However, expressing target in the *current* basis each step is cheap.
    # The catch: the minimum-cardinality solution may change as basis grows.
    # We'll re-express target against the current basis at each M and track the
    # combo; when its size first drops to <= max_size, that M is a candidate.
    # Because we sweep M upward, the first M where a valid (size<=max_size) solution
    # exists is the minimal M (bottleneck) we can achieve with this method.

    basis = {}
    # combo_for_target: we recompute by reducing target against basis each step.

    found_M = None
    found_combo = None

    # We'll iterate M from 211 to max_val
    # Maintain basis incrementally.
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        # Add parity[M] to basis
        msk = parity[M]
        idx = M  # use value as index for clarity
        cur = msk
        combo = {idx}
        while cur != 0:
            pb = bit_length_mask(cur)
            if pb in basis:
                bmask, bcombo = basis[pb]
                cur ^= bmask
                combo ^= bcombo
            else:
                break
        if cur != 0:
            pb = bit_length_mask(cur)
            basis[pb] = (cur, combo)

        # Express target against current basis
        tcur = target
        tcombo = set()
        solvable = True
        while tcur != 0:
            pb = bit_length_mask(tcur)
            if pb not in basis:
                solvable = False
                break
            bmask, bcombo = basis[pb]
            tcur ^= bmask
            tcombo ^= bcombo
        if not solvable or tcur != 0:
            continue
        # light weight reduction
        for pb in sorted(basis.keys()):
            bmask, bcombo = basis[pb]
            newc = tcombo ^ bcombo
            if len(newc) < len(tcombo):
                tcombo = newc
        size = len(tcombo)
        if size <= max_size:
            # candidate solution: the numbers in tcombo (all in [211, M])
            cand = sorted(tcombo)
            # verify
            if verify(cand):
                found_M = M
                found_combo = cand
                break

    if found_M is not None:
        cand_score = found_M - n
        if cand_score < best_score:
            best_score = cand_score
            best_solution = found_combo

    # Final verification of the returned solution
    if not verify(best_solution):
        best_solution = [840]
        best_score = 630

    # Ensure Python int types
    best_solution = [int(x) for x in best_solution]
    return list(best_solution)
# EVOLVE-BLOCK-END
