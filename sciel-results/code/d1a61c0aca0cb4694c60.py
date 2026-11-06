"""Two Sidon sets with disjoint difference sets.
Uses number-theoretic construction via modular arithmetic plus optimized greedy
local improvement to maximize combined size under the disjoint-difference constraint."""

import time
import random
import math

def _check_sidon(els):
    """Check all pair sums a+b (a<=b) are unique."""
    n = len(els)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _nonzero_diffs(els):
    """Set of all nonzero differences (positive and negative)."""
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = els[j] - els[i]
                d.add(diff)
    return d

def _valid(A, B):
    """Check A,B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True

def _try_add_to_set(S, x, used_sums, other_diffs):
    """Try to add x to set S. Returns True if valid."""
    s2 = 2 * x
    if s2 in used_sums:
        return False
    for a in S:
        if (a + x) in used_sums:
            return False
    for a in S:
        d = x - a
        if d != 0 and d in other_diffs:
            return False
        d = a - x
        if d != 0 and d in other_diffs:
            return False
    return True

def _commit_add(S, x, used_sums):
    """Actually add x to S and update used_sums."""
    S.append(x)
    used_sums.add(2 * x)
    for a in S:
        if a != x:
            used_sums.add(a + x)

def _modular_sidon_construction(p, k, offset=0):
    """
    Construct a Sidon set using modular arithmetic.
    Uses the idea: if p is prime and we take elements {i*p mod M : i in S} where
    S is a Sidon set mod p, we can map to [1, n] via collision-avoiding injection.
    
    Simpler approach: use quadratic residues or structured sets.
    """
    # For now, this is a placeholder for future number-theoretic constructions
    return []

def _number_theoretic_seed(rng, n=256):
    """
    Generate seeds using number-theoretic principles.
    Strategy: Pick a prime p, use quadratic residues or structured subsets.
    """
    # Find primes near n
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    # Try different prime-based constructions
    primes = [p for p in range(10, 100) if is_prime(p)]
    
    for p in primes:
        # Quadratic residues mod p
        qr = sorted(set((i * i) % p for i in range(p)))
        # Map to [1, n] by scaling
        scale = n // p
        A = []
        for r in qr:
            val = r * scale + 1
            if val <= n and val not in A:
                A.append(val)
        
        # Complementary structure for B: non-residues
        nqr = sorted(set(range(p)) - set(qr))
        B = []
        for r in nqr:
            val = r * scale + 1
            if val <= n and val not in B:
                B.append(val)
        
        if A and B and _valid(A, B):
            return A, B
    
    return None, None

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    n = 256
    max_size = 64

    best_A = [1]
    best_B = [2]
    best_score = 2

    rng = random.Random(42)

    def _attempt_build(seed_A=None, seed_B=None, strat="greedy"):
        """Build A and B from optional seeds using optimized greedy strategy."""
        A = list(seed_A) if seed_A else []
        B = list(seed_B) if seed_B else []

        if not A and not B:
            if strat == "A_first":
                A.append(rng.randint(1, n))
            else:
                B.append(rng.randint(1, n))

        if not A:
            A.append(rng.randint(1, n))
        if not B:
            B.append(rng.randint(1, n))

        A = list(set(A))
        B = list(set(B))

        if not _valid(A, B):
            return None, None, 0

        used_sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                used_sums_A.add(A[i] + A[j])

        used_sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                used_sums_B.add(B[i] + B[j])

        diffs_A = _nonzero_diffs(A)
        diffs_B = _nonzero_diffs(B)

        # Precompute candidate lists for faster lookup
        candidates_A = [x for x in range(1, n + 1) if x not in A]
        candidates_B = [x for x in range(1, n + 1) if x not in B]
        rng.shuffle(candidates_A)
        rng.shuffle(candidates_B)

        for _round in range(300):
            if time.time() > deadline - 5:
                break
            progress = False

            # Try to add to A
            if len(A) < max_size and candidates_A:
                added = False
                for idx, x in enumerate(candidates_A):
                    if _try_add_to_set(A, x, used_sums_A, diffs_B):
                        _commit_add(A, x, used_sums_A)
                        for a in A:
                            if a != x:
                                diffs_A.add(x - a)
                                diffs_A.add(a - x)
                        candidates_A.pop(idx)
                        progress = True
                        added = True
                        break
                if added:
                    continue

            # Try to add to B
            if len(B) < max_size and candidates_B:
                added = False
                for idx, x in enumerate(candidates_B):
                    if _try_add_to_set(B, x, used_sums_B, diffs_A):
                        _commit_add(B, x, used_sums_B)
                        for b in B:
                            if b != x:
                                diffs_B.add(x - b)
                                diffs_B.add(b - x)
                        candidates_B.pop(idx)
                        progress = True
                        added = True
                        break

            if not progress:
                break

        if _valid(A, B):
            return A, B, len(A) + len(B)
        return None, None, 0

    def _polish(A, B, time_limit=3.0):
        """Intensive local improvement using exhaustive candidate checking."""
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            
            # Try all possible additions to A
            if len(A) < max_size:
                for x in range(1, n + 1):
                    if x not in A:
                        test_A = A + [x]
                        if _valid(test_A, B):
                            A = test_A
                            improved = True
                            break
            
            # Try all possible additions to B
            if len(B) < max_size:
                for x in range(1, n + 1):
                    if x not in B:
                        test_B = B + [x]
                        if _valid(A, test_B):
                            B = test_B
                            improved = True
                            break
            
            # Try swaps: remove one from A, add two new ones
            if len(A) < max_size - 1:
                for i in range(len(A)):
                    for x in range(1, n + 1):
                        if x not in A:
                            test_A = A[:i] + A[i+1:] + [x]
                            if _valid(test_A, B):
                                A = test_A
                                improved = True
                                break
                    if improved:
                        break
            
            if len(B) < max_size - 1:
                for i in range(len(B)):
                    for x in range(1, n + 1):
                        if x not in B:
                            test_B = B[:i] + B[i+1:] + [x]
                            if _valid(A, test_B):
                                B = test_B
                                improved = True
                                break
                    if improved:
                        break
        
        return A, B

    def _deep_polish(A, B, time_limit=8.0):
        """More aggressive local search with 2-opt style moves."""
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            
            # Try all possible additions to A
            if len(A) < max_size:
                for x in range(1, n + 1):
                    if x not in A:
                        test_A = A + [x]
                        if _valid(test_A, B):
                            A = test_A
                            improved = True
                            break
            
            # Try all possible additions to B
            if len(B) < max_size:
                for x in range(1, n + 1):
                    if x not in B:
                        test_B = B + [x]
                        if _valid(A, test_B):
                            B = test_B
                            improved = True
                            break
            
            # Try swaps: remove one from A, add one new one (net 0)
            if len(A) > 1:
                for i in range(len(A)):
                    removed = A[i]
                    base_A = A[:i] + A[i+1:]
                    for x in range(1, n + 1):
                        if x not in base_A:
                            test_A = base_A + [x]
                            if _valid(test_A, B):
                                A = test_A
                                improved = True
                                break
                    if improved:
                        break
            
            # Try swaps: remove one from B, add one new one (net 0)
            if len(B) > 1:
                for i in range(len(B)):
                    removed = B[i]
                    base_B = B[:i] + B[i+1:]
                    for x in range(1, n + 1):
                        if x not in base_B:
                            test_B = base_B + [x]
                            if _valid(A, test_B):
                                B = test_B
                                improved = True
                                break
                    if improved:
                        break
            
            # Try 2-opt: remove two from A, add two new ones
            if len(A) < max_size - 1 and len(A) >= 2:
                for i in range(len(A)):
                    for j in range(i+1, len(A)):
                        base_A = A[:i] + A[i+1:j] + A[j+1:]
                        for x in range(1, n + 1):
                            if x not in base_A:
                                for y in range(x+1, n + 1):
                                    if y not in base_A and y != x:
                                        test_A = base_A + [x, y]
                                        if _valid(test_A, B):
                                            A = test_A
                                            improved = True
                                            break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try 2-opt: remove two from B, add two new ones
            if len(B) < max_size - 1 and len(B) >= 2:
                for i in range(len(B)):
                    for j in range(i+1, len(B)):
                        base_B = B[:i] + B[i+1:j] + B[j+1:]
                        for x in range(1, n + 1):
                            if x not in base_B:
                                for y in range(x+1, n + 1):
                                    if y not in base_B and y != x:
                                        test_B = base_B + [x, y]
                                        if _valid(A, test_B):
                                            B = test_B
                                            improved = True
                                            break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
        
        return A, B

    # Main search loop
    for it in range(50000):
        if time.time() > deadline - 3:
            break

        strat = it % 6

        if strat == 0:
            # Random single seeds
            A0 = [rng.randint(1, n)]
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")
        elif strat == 1:
            # A first
            A0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, None, "A_first")
        elif strat == 2:
            # B first
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(None, B0, "B_first")
        elif strat == 3:
            # Small random starts
            sa = rng.randint(1, 3)
            sb = rng.randint(1, 3)
            A0 = sorted(rng.sample(range(1, n + 1), sa))
            B0 = sorted(rng.sample(range(1, n + 1), sb))
            A, B, score = _attempt_build(A0, B0, "greedy")
        elif strat == 4:
            # Wider spacing seeds
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")
        else:
            # Number-theoretic seeds
            A0, B0 = _number_theoretic_seed(rng, n)
            if A0 is None:
                A0 = [rng.randint(1, n)]
                B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")

        if A is not None and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)

            # Polish with exhaustive local search
            if time.time() < deadline - 15:
                best_A, best_B = _deep_polish(best_A, best_B, time_limit=8.0)
                best_score = len(best_A) + len(best_B)

    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2

    return [best_A, best_B]
