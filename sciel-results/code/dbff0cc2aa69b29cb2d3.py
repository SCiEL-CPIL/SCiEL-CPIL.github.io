import time
from fractions import Fraction
import random

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    # Reduced deadline to 110s to ensure we return before the 150s kill switch
    deadline = start_time + 110

    primes = _sieve(1000)
    CAP = 100000
    random.seed(42)

    best_L = 0
    best_classes = []

    def compute_L(covered):
        # Optimized: stop early if we hit 0
        L = 0
        for i in range(1, CAP + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def build_classes(prime_res):
        return [[p, prime_res[p]] for p in sorted(prime_res.keys())]

    def make_coverage(prime_res):
        covered = bytearray(CAP + 1)
        for p, r in prime_res.items():
            s = r
            if s == 0:
                s = p
            for k in range(s, CAP + 1, p):
                covered[k] = 1
        return covered

    def greedy_init(deadline):
        nonlocal best_L, best_classes
        covered = bytearray(CAP + 1)
        used = set()
        budget = Fraction(1)
        prime_res = {}

        n = 1
        while n <= CAP:
            if time.time() > deadline:
                break
            if covered[n]:
                n += 1
                continue

            best_p = None
            best_r = 0
            best_score = -1

            window = 3000
            hi = min(n + window, CAP)

            # Limit candidates to improve speed
            candidate_primes = [p for p in primes if p not in used and budget >= Fraction(1, p)]
            # Sort by p to prioritize smaller primes (higher density) if scores are close, or just randomize
            random.shuffle(candidate_primes)
            
            for p in candidate_primes:
                r = n % p
                cnt = 0
                k = n
                while k <= hi:
                    if not covered[k]:
                        cnt += 1
                    k += p

                score = cnt * p
                if score > best_score:
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
            for k in range(s, CAP + 1, best_p):
                covered[k] = 1

            n += 1

        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)
        return L

    def local_search(classes, deadline):
        nonlocal best_L, best_classes
        if not classes:
            return

        prime_res = {p: r for p, r in classes}
        used = set(prime_res.keys())
        budget_used = sum(Fraction(1, p) for p in prime_res)
        remaining = Fraction(1) - budget_used

        covered = make_coverage(prime_res)
        L = compute_L(covered)

        if L >= CAP:
            if L > best_L:
                best_L = L
                best_classes = build_classes(prime_res)
            return

        no_improve = 0
        max_no_improve = 50 # Reduced max no-improve count to save time

        while time.time() < deadline and no_improve < max_no_improve:
            gap = L + 1
            improved = False

            # Move 1: Reassign residue of an existing prime
            # Limit number of candidates to test to avoid timeout
            candidates = list(prime_res.items())
            random.shuffle(candidates)
            candidates = candidates[:20] # Only test 20 random reassignments
            
            for p, r in candidates:
                if time.time() > deadline:
                    return
                new_r = gap % p
                if new_r == r:
                    continue
                
                # Efficiently update coverage
                # Unmark old
                s_old = r
                if s_old == 0: s_old = p
                for k in range(s_old, CAP + 1, p):
                    if covered[k]:
                        # We can't just unmark because other primes might cover it.
                        # So we must recalculate or use a count array.
                        # Recalculating is too slow. 
                        # Let's stick to the previous method but limit iterations.
                        pass
                
                # The previous method of creating a new_cov and computing L is O(CAP) per move.
                # With CAP=100k and 20 moves, that's 2M ops per iteration.
                # 50 iterations = 100M ops. Might be too slow in Python.
                
                # Alternative: Just try a few moves and hope for the best.
                # If this is too slow, we might just skip local search or make it very light.
                
                # Let's try a lighter local search: only try adding new primes if budget allows.
                # Reassignment is expensive to evaluate correctly without a full re-scan or complex bookkeeping.
                # Given the time constraints, we will prioritize "Add Prime" moves which are easier to evaluate incrementally?
                # No, "Add Prime" also requires checking if it helps L.
                
                # Let's just do a very limited local search:
                # Try to add one unused prime with residue gap % p.
                # If it increases L, keep it.
                # If not, try reassigning one existing prime (only 5 candidates).
                
                pass # Skip the complex reassignment loop for now to ensure speed

            # Move 2: Add an unused prime
            if not improved:
                unused = [p for p in primes if p not in used and remaining >= Fraction(1, p)]
                random.shuffle(unused)
                unused = unused[:10] # Limit candidates

                for p in unused:
                    if time.time() > deadline:
                        return
                    r = gap % p
                    new_cov = covered[:]
                    s = r
                    if s == 0:
                        s = p
                    for k in range(s, CAP + 1, p):
                        new_cov[k] = 1
                    L2 = compute_L(new_cov)
                    if L2 > L:
                        prime_res[p] = r
                        used.add(p)
                        remaining -= Fraction(1, p)
                        covered = new_cov
                        L = L2
                        no_improve = 0
                        improved = True
                        if L > best_L:
                            best_L = L
                            best_classes = build_classes(prime_res)
                        if L >= CAP:
                            return
                        break

            if not improved:
                no_improve += 1

        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)

    # Phase 1: Greedy initialization
    greedy_init(deadline - 10)

    # Phase 2: Local search refinement
    if time.time() < deadline - 2 and best_classes:
        local_search(best_classes, deadline)

    if not best_classes:
        # Fallback: Use small primes to ensure valid return
        small = [p for p in primes if p <= 97]
        # Ensure budget constraint
        total_cost = sum(Fraction(1, p) for p in small)
        while total_cost > Fraction(1) and small:
            small.pop()
            total_cost = sum(Fraction(1, p) for p in small)
        best_classes = [[p, 0] for p in small]

    return best_classes
