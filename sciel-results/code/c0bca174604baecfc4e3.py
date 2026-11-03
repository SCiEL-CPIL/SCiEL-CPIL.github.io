"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from itertools import product as iter_product, combinations

# EVOLVE-BLOCK-START

def _generate_primes(n):
    """Generate first n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes

_PRIMES = _generate_primes(200)


def _score_fast(U):
    """Fast score computation using numpy."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / n_sums
    if ratio <= 1.0:
        return 1.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _symmetric_core_set(S, M):
    """
    Build U = {0} ∪ {x, M-x | x in S} where S ⊂ {1, ..., M-1}.
    This creates symmetry: if x is in U, so is M-x.
    The sumset concentrates around M and 2M, while difference set spreads across [-M, M].
    """
    U = {0}
    for x in S:
        if 0 < x < M:
            U.add(x)
            U.add(M - x)
    return sorted(U)


def _dual_grid_set(m1, m2, n1, n2, offset1=0, offset2=0):
    """
    Build a set from the union of two arithmetic progressions (dual-grid).
    AP1: {offset1 + k*m1 : k = 0, 1, ..., n1-1}
    AP2: {offset2 + k*m2 : k = 0, 1, ..., n2-1}
    U = AP1 ∪ AP2, always including 0.
    
    The idea: two grids with different moduli create a sparse set where
    the sumset has limited structure (concentrated) while the difference
    set spreads more widely.
    """
    U = {0}
    for k in range(n1):
        val = offset1 + k * m1
        if val >= 0:
            U.add(val)
    for k in range(n2):
        val = offset2 + k * m2
        if val >= 0:
            U.add(val)
    return sorted(U)


def _dual_grid_search(time_limit=25.0):
    """
    Search over dual-grid constructions: U = AP1 ∪ AP2 with different moduli.
    We vary m1, m2, lengths, and offsets to find good ratio |U-U|/|U+U|.
    """
    start = time.time()
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # Search over pairs of moduli and grid parameters
    # m1, m2 should be somewhat coprime for interesting structure
    for m1 in range(2, 80):
        if time.time() - start > time_limit:
            break
        for m2 in range(m1 + 1, min(m1 * 3, 120)):
            if time.time() - start > time_limit:
                break
            # Skip if gcd is too large (degenerate cases)
            if math.gcd(m1, m2) > 2:
                continue
            
            # Try various lengths and offsets
            max_len = max(5, 200 // max(m1, m2))
            for n1 in range(2, min(max_len, 15) + 1):
                if time.time() - start > time_limit:
                    break
                for n2 in range(2, min(max_len, 15) + 1):
                    if time.time() - start > time_limit:
                        break
                    # Try a few offset combinations
                    for off1 in [0, 1, m1 // 2]:
                        for off2 in [0, 1, m2 // 2]:
                            if time.time() - start > time_limit:
                                break
                            U = _dual_grid_set(m1, m2, n1, n2, off1, off2)
                            if len(U) < 3:
                                continue
                            s = _score_fast(U)
                            if s > best_score:
                                best_score = s
                                best_U = list(U)
    
    # Also try intersection-based grids: U = {x : x ≡ r1 mod m1 AND x ≡ r2 mod m2}
    # This is equivalent to a single AP with step lcm(m1,m2)
    for m1 in range(2, 50):
        if time.time() - start > time_limit:
            break
        for m2 in range(m1 + 1, min(m1 * 3, 80)):
            if time.time() - start > time_limit:
                break
            g = math.gcd(m1, m2)
            if g == 1:
                step = m1 * m2
            else:
                step = m1 * m2 // g
            if step > 5000:
                continue
            # U = {k*step : k = 0, ..., n}
            n_max = min(20, 5000 // step + 1)
            for n in range(3, n_max + 1):
                if time.time() - start > time_limit:
                    break
                U = list(range(0, (n + 1) * step, step))
                s = _score_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
    
    return best_U, best_score


def _exhaustive_small_search(max_n=6, max_val=50, time_limit=15.0):
    """
    Exhaustive search over small sets (up to max_n elements, max element max_val).
    We search over subsets of {1, ..., max_val} of size up to max_n-1 (since 0 is always included).
    """
    start = time.time()
    best_score = 0.0
    best_U = [0, 1, 3]

    # For n=2: {0, a}
    for a in range(1, max_val + 1):
        U = [0, a]
        s = _score_fast(U)
        if s > best_score:
            best_score = s
            best_U = U

    # For n=3: {0, a, b} with 1 <= a < b <= max_val
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            if time.time() - start > time_limit:
                return best_U, best_score
            U = [0, a, b]
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_U = U

    # For n=4: {0, a, b, c} with 1 <= a < b < c <= max_val
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            for c in range(b + 1, max_val + 1):
                if time.time() - start > time_limit:
                    return best_U, best_score
                U = [0, a, b, c]
                s = _score_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = U

    # For n=5: {0, a, b, c, d} with 1 <= a < b < c < d <= max_val (limit to smaller range)
    mv5 = min(max_val, 30)
    for a in range(1, mv5 + 1):
        for b in range(a + 1, mv5 + 1):
            for c in range(b + 1, mv5 + 1):
                for d in range(c + 1, mv5 + 1):
                    if time.time() - start > time_limit:
                        return best_U, best_score
                    U = [0, a, b, c, d]
                    s = _score_fast(U)
                    if s > best_score:
                        best_score = s
                        best_U = U

    # For n=6: {0, a, b, c, d, e} with 1 <= a < b < c < d < e <= max_val (limit to smaller range)
    mv6 = min(max_val, 20)
    for a in range(1, mv6 + 1):
        for b in range(a + 1, mv6 + 1):
            for c in range(b + 1, mv6 + 1):
                for d in range(c + 1, mv6 + 1):
                    for e in range(d + 1, mv6 + 1):
                        if time.time() - start > time_limit:
                            return best_U, best_score
                        U = [0, a, b, c, d, e]
                        s = _score_fast(U)
                        if s > best_score:
                            best_score = s
                            best_U = U

    return best_U, best_score


def _symmetric_core_search(time_limit=20.0):
    """
    Search over symmetric core constructions U = {0} ∪ {x, M-x | x in S}.
    For small |S|, we can try various M and S.
    """
    start = time.time()
    best_score = 0.0
    best_U = [0, 1, 3]

    # Try various M values and small subsets S
    for M in range(4, 200):
        if time.time() - start > time_limit:
            break

        # S of size 1: {x} -> U = {0, x, M-x}
        for x in range(1, M // 2 + 1):
            U = _symmetric_core_set([x], M)
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_U = list(U)

        # S of size 2: {x, y} -> U = {0, x, M-x, y, M-y}
        if M <= 100:
            for x in range(1, M // 2 + 1):
                for y in range(x + 1, M // 2 + 1):
                    if time.time() - start > time_limit:
                        break
                    U = _symmetric_core_set([x, y], M)
                    s = _score_fast(U)
                    if s > best_score:
                        best_score = s
                        best_U = list(U)

        # S of size 3: {x, y, z} -> U = {0, x, M-x, y, M-y, z, M-z}
        if M <= 60:
            for x in range(1, M // 3 + 1):
                for y in range(x + 1, (M - x) // 2 + 1):
                    for z in range(y + 1, (M - x - y) // 2 + 1):
                        if time.time() - start > time_limit:
                            break
                        U = _symmetric_core_set([x, y, z], M)
                        s = _score_fast(U)
                        if s > best_score:
                            best_score = s
                            best_U = list(U)

    # Also try non-symmetric extensions of good symmetric cores
    # For the best symmetric core found, try adding elements
    if best_score > 0 and best_U:
        U = list(best_U)
        max_u = max(U)
        # Try adding elements near M or 2M
        M = max_u
        candidates = set()
        for d in range(1, 30):
            candidates.add(M + d)
            candidates.add(2 * M - d)
            if d > 0:
                candidates.add(d)
        random.shuffle(list(candidates))
        for new_val in list(candidates)[:20]:
            if new_val <= 0 or new_val in U:
                continue
            new_U = sorted(U + [new_val])
            s = _score_fast(new_U)
            if s > best_score:
                best_score = s
                best_U = new_U

    return best_U, best_score


def _local_search(U, time_limit=20.0, max_steps=50000):
    """Local search with add, remove, and swap operations."""
    U = sorted(set(U))
    best_score = _score_fast(U)
    best_U = list(U)
    start = time.time()
    n = len(U)
    max_u = U[-1] if U else 0

    for step in range(max_steps):
        if time.time() - start > time_limit:
            break

        improved = False
        max_u = max(U) if U else 0

        # Try adding a new element
        candidates = []
        # Near max
        for d in range(1, 30):
            candidates.append(max_u + d)
        # Geometric extensions
        for d in range(1, 15):
            candidates.append(max_u * 2 + d)
            candidates.append(max_u * 3 + d)
        # Symmetric candidates
        for d in range(1, 20):
            candidates.append(max_u - d)
        # Interior random
        for _ in range(10):
            candidates.append(random.randint(1, max_u))

        random.shuffle(candidates)
        for new_val in candidates[:15]:
            if new_val <= 0 or new_val in U:
                continue
            new_U = sorted(U + [new_val])
            new_score = _score_fast(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                improved = True
                break

        if not improved:
            # Try removing an element (not 0)
            if len(U) > 2:
                idxs = list(range(1, len(U)))
                random.shuffle(idxs)
                for idx in idxs[:8]:
                    new_U = U[:idx] + U[idx+1:]
                    new_score = _score_fast(new_U)
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        improved = True
                        break

        if not improved:
            # Try swapping an element
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Try symmetric swap: replace with max_u - old_val or similar
                swap_candidates = [
                    max_u - old_val,
                    max_u + random.randint(1, 20),
                    max_u * 2 - old_val,
                    random.randint(1, max_u * 2 + 50),
                ]
                for new_val in swap_candidates:
                    if new_val <= 0 or new_val == old_val or new_val in U:
                        continue
                    new_U = list(U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    new_score = _score_fast(new_U)
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        improved = True
                        break

        if not improved and step > 200 and random.random() < 0.03:
            # Perturb: replace a random element with a random value
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                new_val = random.randint(1, max_u * 3)
                if new_val not in U:
                    U = list(U)
                    U[idx] = new_val
                    U = sorted(U)
                    best_score = _score_fast(U)
                    best_U = list(U)

    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_U = [0, 1, 3]
    best_score = _score_fast(best_U)

    # Phase 1: Exhaustive search over small sets
    if time.time() < deadline - 85:
        U, score = _exhaustive_small_search(max_n=6, max_val=50, time_limit=15.0)
        if score > best_score:
            best_score = score
            best_U = U

    # Phase 2: Symmetric core search
    if time.time() < deadline - 70:
        U, score = _symmetric_core_search(time_limit=15.0)
        if score > best_score:
            best_score = score
            best_U = U

    # Phase 3: Dual-grid search (NEW)
    if time.time() < deadline - 55:
        U, score = _dual_grid_search(time_limit=25.0)
        if score > best_score:
            best_score = score
            best_U = U

    # Phase 4: Try geometric sets {0, 1, b, b^2, ..., b^(n-1)}
    if time.time() < deadline - 45:
        for b in range(2, 40):
            if time.time() > deadline - 45:
                break
            for n in range(2, 12):
                if time.time() > deadline - 45:
                    break
                U = [0]
                val = 1
                for i in range(1, n):
                    val *= b
                    U.append(val)
                score = _score_fast(U)
                if score > best_score:
                    best_score = score
                    best_U = U

    # Phase 5: Local search from best seed
    if time.time() < deadline - 25:
        U, score = _local_search(best_U, time_limit=25.0, max_steps=50000)
        if score > best_score:
            best_score = score
            best_U = U

    # Phase 6: More local search rounds with perturbation
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb
            U = list(best_U)
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            max_u = max(U) if U else 0
            new_val = random.randint(1, max_u + max(10, max_u // 3))
            if new_val not in U:
                U.append(new_val)
            U = sorted(U)

            U, score = _local_search(U, time_limit=8.0, max_steps=15000)
            if score > best_score:
                best_score = score
                best_U = U

    # Phase 7: Final polish - try to add/remove single elements
    if time.time() < deadline - 5:
        U = list(best_U)
        max_u = max(U) if U else 0
        # Try adding each candidate
        for d in range(1, 50):
            new_val = max_u + d
            if new_val not in U:
                new_U = sorted(U + [new_val])
                s = _score_fast(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    max_u = max(U)
        # Try removing each element (not 0)
        for idx in range(len(U) - 1, 0, -1):
            if time.time() > deadline - 5:
                break
            new_U = U[:idx] + U[idx+1:]
            s = _score_fast(new_U)
            if s > best_score:
                best_score = s
                best_U = new_U
                U = new_U

    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)

    return sorted(best_U)

# EVOLVE-BLOCK-END
