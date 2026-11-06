import time
import numpy as np
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

PRIMES = _sieve(1000)
CAP = 100000

def construct():
    start_time = time.time()
    deadline = start_time + 115

    best_L = 0
    best_classes = []

    # Pre-compute coverage bitsets for all (p, r) pairs
    # coverage[p_idx][r] = numpy array of length CAP+1, 1 where covered
    # This is too much memory for all primes*residues, so compute on the fly but cache key ones

    def compute_L_from_mask(mask):
        """Find the largest L such that mask[1..L] are all 1."""
        # Find first zero from position 1
        # mask is a numpy array of uint8
        # We want the first index i >= 1 where mask[i] == 0
        if len(mask) <= 1:
            return 0
        # Find first 0 in mask[1:]
        zeros = np.where(mask[1:] == 0)[0]
        if len(zeros) == 0:
            return CAP
        return zeros[0]  # This is the 0-based index in mask[1:], so the actual position is zeros[0]+1... wait
        # Actually mask[1:] has indices 0..CAP-1 corresponding to positions 1..CAP
        # So if zeros[0] = k, then position 1+k is uncovered, so L = k
        # Wait: mask[1:] index 0 = mask[1], index 1 = mask[2], etc.
        # If mask[1+k] == 0, then L = k (positions 1..k are covered)
        # Actually L should be the largest such that 1..L are covered.
        # If mask[1+k] == 0, then positions 1..k are covered, so L = k.
        # But we need L >= 1, so if mask[1] == 0, L = 0.
        # Let me re-check: zeros[0] is the index in mask[1:] where value is 0.
        # mask[1:][zeros[0]] = mask[1 + zeros[0]] = 0
        # So positions 1, 2, ..., 1+zeros[0]-1 = zeros[0] are covered.
        # So L = zeros[0].
        return zeros[0]

    def make_coverage_mask(prime_res):
        """Create a numpy uint8 array of length CAP+1 showing coverage."""
        mask = np.zeros(CAP + 1, dtype=np.uint8)
        for p, r in prime_res.items():
            s = r
            if s == 0:
                s = p
            mask[s::p] = 1
        return mask

    def build_classes(prime_res):
        return [[p, prime_res[p]] for p in sorted(prime_res.keys())]

    # Phase 1: Greedy initialization with multiple strategies
    def greedy_init(window, eff, deadline):
        nonlocal best_L, best_classes
        mask = np.zeros(CAP + 1, dtype=np.uint8)
        used = set()
        budget = Fraction(1)
        prime_res = {}

        n = 1
        while n <= CAP:
            if time.time() > deadline:
                break
            if mask[n]:
                n += 1
                continue

            best_p = None
            best_r = 0
            best_score = -1

            hi = min(n + window, CAP)

            for p in PRIMES:
                if p in used:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue

                r = n % p
                # Count how many uncovered positions in [n, hi] are covered by this class
                s = r
                if s == 0:
                    s = p
                # Positions: s, s+p, s+2p, ... <= hi
                # Filter to [n, hi]
                # First position >= n: k0 = ceil((n - s) / p) * p + s if n > s else s
                # Let's just iterate
                cnt = 0
                k = s
                while k <= hi:
                    if k >= n and not mask[k]:
                        cnt += 1
                    k += p

                if eff:
                    score = cnt * p
                else:
                    score = cnt

                if score > best_score or (score == best_score and (best_p is None or p < best_p)):
                    best_score = score
                    best_p = p
                    best_r = r

            if best_p is None:
                break

            used.add(best_p)
            budget -= Fraction(1, best_p)
            prime_res[best_p] = best_r

            s = best_r
            if s == 0:
                s = best_p
            mask[s::best_p] = 1

            n += 1

        L = compute_L_from_mask(mask)
        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)
        return L

    # Run several greedy strategies
    strategies = []
    for w in [500, 2000, 5000, 10000, 20000]:
        for eff in [True, False]:
            strategies.append((w, eff))

    for w, eff in strategies:
        if time.time() > deadline - 40:
            break
        greedy_init(w, eff, deadline)

    # Phase 2: Local search with bitset operations
    if time.time() < deadline - 10 and best_classes:
        prime_res = {p: r for p, r in best_classes}
        used = set(prime_res.keys())
        budget_used = sum(Fraction(1, p) for p in prime_res)
        remaining = Fraction(1) - budget_used
        mask = make_coverage_mask(prime_res)
        L = compute_L_from_mask(mask)

        no_improve = 0
        max_no_improve = 50

        while time.time() < deadline - 5 and no_improve < max_no_improve:
            gap = L + 1
            if gap > CAP:
                break
            improved = False

            # Move 1: Reassign residue of existing prime to cover gap
            candidates = []
            for p, r in prime_res.items():
                new_r = gap % p
                if new_r != r:
                    candidates.append((p, r, new_r))

            # Sort by potential impact: smaller primes first (more frequent)
            candidates.sort(key=lambda x: x[0])

            for p, old_r, new_r in candidates[:20]:
                if time.time() > deadline - 5:
                    return best_classes

                trial_res = dict(prime_res)
                trial_res[p] = new_r
                trial_mask = make_coverage_mask(trial_res)
                L2 = compute_L_from_mask(trial_mask)

                if L2 > L:
                    prime_res = trial_res
                    mask = trial_mask
                    L = L2
                    no_improve = 0
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_classes = build_classes(prime_res)
                    if L >= CAP:
                        return best_classes
                    break

            # Move 2: Add unused prime
            unused = [p for p in PRIMES if p not in used and remaining >= Fraction(1, p)]
            unused.sort()
            for p in unused[:20]:
                if time.time() > deadline - 5:
                    return best_classes

                r = gap % p
                new_mask = mask.copy()
                s = r
                if s == 0:
                    s = p
                new_mask[s::p] = 1
                L2 = compute_L_from_mask(new_mask)
                if L2 > L:
                    prime_res[p] = r
                    used.add(p)
                    remaining -= Fraction(1, p)
                    mask = new_mask
                    L = L2
                    no_improve = 0
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_classes = build_classes(prime_res)
                    if L >= CAP:
                        return best_classes
                    break

            # Move 3: Swap - remove one, add another
            if not improved:
                sorted_used = sorted(used, key=lambda p: -1.0 / p)  # largest cost first
                for p_remove in sorted_used[:10]:
                    if time.time() > deadline - 5:
                        return best_classes

                    trial_res = dict(prime_res)
                    del trial_res[p_remove]
                    trial_used = set(used)
                    trial_used.remove(p_remove)
                    trial_remaining = remaining + Fraction(1, p_remove)
                    trial_mask = make_coverage_mask(trial_res)
                    L_base = compute_L_from_mask(trial_mask)

                    for p_add in PRIMES:
                        if p_add in trial_used:
                            continue
                        if trial_remaining < Fraction(1, p_add):
                            continue

                        r_add = L_base % p_add if L_base < CAP else 0
                        trial_res2 = dict(trial_res)
                        trial_res2[p_add] = r_add
                        trial_mask2 = make_coverage_mask(trial_res2)
                        L2 = compute_L_from_mask(trial_mask2)

                        if L2 > L:
                            prime_res = trial_res2
                            used = set(prime_res.keys())
                            remaining = Fraction(1) - sum(Fraction(1, p) for p in prime_res)
                            mask = trial_mask2
                            L = L2
                            no_improve = 0
                            improved = True
                            if L > best_L:
                                best_L = L
                                best_classes = build_classes(prime_res)
                            if L >= CAP:
                                return best_classes
                            break

                    if improved:
                        break

            if not improved:
                no_improve += 1

    # Phase 3: Targeted DFS for gap-filling
    if time.time() < deadline - 5 and best_L < 999 and best_classes:
        prime_res = {p: r for p, r in best_classes}
        used = set(prime_res.keys())
        remaining = Fraction(1) - sum(Fraction(1, p) for p in prime_res)
        mask = make_coverage_mask(prime_res)
        L = compute_L_from_mask(mask)

        if L < CAP and remaining > 0:
            unused_primes = [p for p in PRIMES if p not in used and remaining >= Fraction(1, p)]
            unused_primes.sort()

            for p in unused_primes[:25]:
                if time.time() > deadline - 2:
                    break

                r = (L + 1) % p
                new_mask = mask.copy()
                s = r
                if s == 0:
                    s = p
                new_mask[s::p] = 1
                L2 = compute_L_from_mask(new_mask)

                if L2 > L:
                    prime_res[p] = r
                    used.add(p)
                    remaining -= Fraction(1, p)
                    mask = new_mask
                    L = L2
                    if L > best_L:
                        best_L = L
                        best_classes = build_classes(prime_res)
                    if L >= CAP:
                        break

    return best_classes
