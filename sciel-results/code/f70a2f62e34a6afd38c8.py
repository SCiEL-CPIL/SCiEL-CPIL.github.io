import time
from fractions import Fraction
import random
import math

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
    deadline = start_time + 110  # safety margin before 150s kill

    primes = _sieve(1000)
    CAP = 100000
    random.seed(42)

    best_L = 0
    best_classes = []

    def compute_L(covered):
        L = 0
        # Early exit optimization: if covered[1] is 0, return 0
        # But we usually start from 1.
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
            # Using a loop is often faster than slice assignment for sparse updates in Python
            # but slice assignment is faster for dense. Here density is 1/p.
            # Let's stick to the loop for simplicity and correctness in bytearray
            for k in range(s, CAP + 1, p):
                covered[k] = 1
        return covered

    def greedy_init(window, eff, deadline):
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

            hi = min(n + window, CAP)

            for p in primes:
                if p in used:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue

                r = n % p
                cnt = 0
                k = n
                while k <= hi:
                    if not covered[k]:
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
            for k in range(s, CAP + 1, best_p):
                covered[k] = 1

            n += 1

        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)
        return L

    # Phase 1: Multiple greedy strategies with different windows and efficiency metrics
    strategies = []
    for w in [500, 1000, 2000, 3000, 5000, 8000, 12000, 20000]:
        for eff in [True, False]:
            strategies.append((w, eff))

    # Sort to try mid-range windows with efficiency first
    strategies.sort(key=lambda x: (not x[1], abs(x[0] - 3000)))

    for w, eff in strategies:
        if time.time() > deadline - 10:
            break
        greedy_init(w, eff, deadline)

    # Phase 2: Enhanced local search with best-move selection
    if time.time() < deadline - 5 and best_classes:
        prime_res = {p: r for p, r in best_classes}
        used = set(prime_res.keys())
        budget_used = sum(Fraction(1, p) for p in prime_res)
        remaining = Fraction(1) - budget_used

        covered = make_coverage(prime_res)
        L = compute_L(covered)

        if L < CAP:
            no_improve = 0
            max_no_improve = 100 # Increased iterations since we pick best move
            
            while time.time() < deadline - 2 and no_improve < max_no_improve:
                gap = L + 1
                improved = False
                
                # We will evaluate all single-step moves and pick the one that maximizes L
                best_move_L = L
                best_move_type = None # 'reassign', 'add', 'swap'
                best_move_data = None

                # Move 1: Try reassigning residue of an existing prime to cover gap
                for p, r in prime_res.items():
                    new_r = gap % p
                    if new_r != r:
                        # Quick check: does new_r cover gap?
                        # Yes, by definition gap % p == new_r.
                        # We need to check if this reassignment breaks coverage of previous numbers?
                        # No, it only changes which numbers are covered by p.
                        # It might uncover some numbers < gap that were ONLY covered by p with old_r.
                        # So we must compute full L.
                        
                        # To save time, we can estimate:
                        # If the old residue covered gap? No, gap % p != r.
                        # So old residue did NOT cover gap.
                        # New residue DOES cover gap.
                        # The risk is uncovering numbers in [1, gap-1].
                        
                        # Let's do a fast simulation.
                        # Copy covered array? Too slow.
                        # Instead, we can compute the new coverage for p and check intersection.
                        # But make_coverage is O(CAP * num_primes).
                        
                        # Optimization: Only simulate if the "loss" is likely small.
                        # For now, just simulate for all candidates if time permits.
                        # To speed up, we can skip if p is large (low density) and we have many small primes.
                        
                        trial_res = dict(prime_res)
                        trial_res[p] = new_r
                        trial_cov = make_coverage(trial_res)
                        L2 = compute_L(trial_cov)
                        
                        if L2 > best_move_L:
                            best_move_L = L2
                            best_move_type = 'reassign'
                            best_move_data = (p, new_r, trial_res, trial_cov)

                # Move 2: Add an unused prime with residue covering gap
                unused = [p for p in primes if p not in used and remaining >= Fraction(1, p)]
                # Sort by p ascending to try small primes first? Or random?
                # Small primes have higher cost but better coverage density.
                # Let's try all feasible unused primes.
                # To limit time, if there are too many, sample them.
                # Max 168 primes, so checking all is fine.
                
                for p in unused:
                    r = gap % p
                    # Check if adding this covers gap (yes) and doesn't break anything (it can't break, only adds).
                    # So L2 >= L.
                    # We just need to find the max L2.
                    
                    trial_res = dict(prime_res)
                    trial_res[p] = r
                    trial_cov = make_coverage(trial_res)
                    L2 = compute_L(trial_cov)
                    
                    if L2 > best_move_L:
                        best_move_L = L2
                        best_move_type = 'add'
                        best_move_data = (p, r, trial_res, trial_cov)

                # Move 3: Swap (Remove one, Add one)
                # Only try if no better move found in 1 or 2.
                # This is expensive, so we limit the candidates.
                if best_move_L == L:
                    # Try removing high cost primes and adding low cost primes
                    sorted_used = sorted(used, key=lambda p: -1.0/p) # Descending cost (1/p)
                    # Actually -1.0/p sorts by p ascending? No.
                    # 1/2 = 0.5, 1/3 = 0.33.
                    # We want to remove large 1/p (small p)? Or small 1/p (large p)?
                    # Removing a large p (small cost) frees little budget.
                    # Removing a small p (large cost) frees a lot of budget.
                    # But removing a small p might break coverage significantly.
                    # Let's try removing the largest p first (least disruptive to density, but frees least budget).
                    # Actually, let's just try a few swaps.
                    
                    candidates_remove = sorted(used, key=lambda p: p) # Smallest p first (most expensive)
                    # We'll try removing the top 5 most expensive and top 5 least expensive
                    candidates_remove = candidates_remove[:5] + candidates_remove[-5:]
                    
                    for p_remove in candidates_remove:
                        if time.time() > deadline - 2:
                            break
                        trial_res = dict(prime_res)
                        del trial_res[p_remove]
                        trial_used = set(used)
                        trial_used.remove(p_remove)
                        trial_remaining = remaining + Fraction(1, p_remove)
                        
                        # Find a prime to add
                        unused_after_remove = [p for p in primes if p not in trial_used and trial_remaining >= Fraction(1, p)]
                        # Sort by p ascending (prefer small primes for coverage)
                        unused_after_remove.sort()
                        
                        # Try top 5 candidates
                        for p_add in unused_after_remove[:5]:
                            r = gap % p_add
                            trial_res2 = dict(trial_res)
                            trial_res2[p_add] = r
                            trial_cov = make_coverage(trial_res2)
                            L2 = compute_L(trial_cov)
                            
                            if L2 > best_move_L:
                                best_move_L = L2
                                best_move_type = 'swap'
                                best_move_data = (p_remove, p_add, r, trial_res2, trial_cov)

                # Apply the best move found
                if best_move_L > L:
                    improved = True
                    if best_move_type == 'reassign':
                        p, new_r, trial_res, trial_cov = best_move_data
                        prime_res = trial_res
                        covered = trial_cov
                        L = best_move_L
                        
                    elif best_move_type == 'add':
                        p, r, trial_res, trial_cov = best_move_data
                        prime_res = trial_res
                        used.add(p)
                        remaining -= Fraction(1, p)
                        covered = trial_cov
                        L = best_move_L
                        
                    elif best_move_type == 'swap':
                        p_remove, p_add, r, trial_res, trial_cov = best_move_data
                        prime_res = trial_res
                        used.remove(p_remove)
                        used.add(p_add)
                        remaining = remaining + Fraction(1, p_remove) - Fraction(1, p_add)
                        covered = trial_cov
                        L = best_move_L

                    no_improve = 0
                    if L > best_L:
                        best_L = L
                        best_classes = build_classes(prime_res)
                    if L >= CAP:
                        return
                else:
                    no_improve += 1

        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)

    if not best_classes:
        # Fallback: Use small primes to ensure valid return
        small = [p for p in primes if p <= 97]
        total_cost = sum(Fraction(1, p) for p in small)
        while total_cost > Fraction(1) and small:
            small.pop()
            total_cost = sum(Fraction(1, p) for p in small)
        best_classes = [[p, 0] for p in small]

    return best_classes
