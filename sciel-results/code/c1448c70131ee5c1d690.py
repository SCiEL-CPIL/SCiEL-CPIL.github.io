"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import isqrt

def _primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}

    # Fast coverage check: given a list of (p, r) classes, return the largest L
    # such that every integer in 1..L is covered.
    def compute_L(classes, max_check=100000):
        # We'll build a covered array up to max_check
        covered = bytearray(max_check + 1)
        for p, r in classes:
            start = r if r > 0 else p
            for n in range(start, max_check + 1, p):
                covered[n] = 1
        L = 0
        for i in range(1, max_check + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    # Greedy construction: build a set of classes that covers as much as possible
    # Uses a bitset for uncovered numbers for efficiency
    def greedy_build(L_target, primes_list, budget, time_limit):
        """
        Greedily select prime-residue classes to cover 1..L_target.
        Returns (classes, budget_used)
        """
        covered = bytearray(L_target + 1)
        used_primes = set()
        classes = []
        remaining_budget = budget
        t0 = time.time()

        # Precompute for each prime the list of numbers it covers for each residue
        # But that's too much memory. Instead, compute on the fly.

        while time.time() - t0 < time_limit:
            best_p = None
            best_r = None
            best_gain = 0

            for p in primes_list:
                if p in used_primes:
                    continue
                if remaining_budget < recip[p]:
                    continue

                # Find the residue r that maximizes new coverage
                best_r_local = 0
                best_gain_local = 0
                for r in range(p):
                    start = r if r > 0 else p
                    if start > L_target:
                        continue
                    gain = 0
                    for n in range(start, L_target + 1, p):
                        if not covered[n]:
                            gain += 1
                    if gain > best_gain_local:
                        best_gain_local = gain
                        best_r_local = r

                if best_gain_local > best_gain:
                    best_gain = best_gain_local
                    best_p = p
                    best_r = best_r_local

            if best_p is None or best_gain == 0:
                break

            used_primes.add(best_p)
            classes.append([best_p, best_r])
            remaining_budget -= recip[best_p]

            # Update covered
            p = best_p
            r = best_r
            start = r if r > 0 else p
            for n in range(start, L_target + 1, p):
                covered[n] = 1

        return classes, remaining_budget

    # Local search: try to improve by swapping residues or adding/removing classes
    def local_search(classes, L_target, primes_list, budget, time_limit):
        """
        Local search to improve the solution.
        Returns improved classes list.
        """
        t0 = time.time()
        best_classes = list(classes)
        best_L = compute_L(best_classes, min(L_target + 100, 100000))
        used_primes = set(p for p, r in best_classes)
        remaining_budget = budget - sum(recip[p] for p, r in best_classes)

        while time.time() - t0 < time_limit:
            improved = False

            # Try swapping residues of existing classes
            for i, (p, r) in enumerate(best_classes):
                for r_new in range(p):
                    if r_new == r:
                        continue
                    trial_classes = list(best_classes)
                    trial_classes[i] = [p, r_new]
                    trial_L = compute_L(trial_classes, min(L_target + 100, 100000))
                    if trial_L > best_L:
                        best_classes = trial_classes
                        best_L = trial_L
                        improved = True

            # Try adding a new class
            for p in primes_list:
                if p in used_primes:
                    continue
                if remaining_budget < recip[p]:
                    continue
                for r in range(p):
                    trial_classes = list(best_classes) + [[p, r]]
                    trial_L = compute_L(trial_classes, min(L_target + 100, 100000))
                    if trial_L > best_L:
                        best_classes = trial_classes
                        best_L = trial_L
                        used_primes.add(p)
                        remaining_budget -= recip[p]
                        improved = True
                        break

            # Try removing a class and re-optimizing
            for i in range(len(best_classes)):
                p, r = best_classes[i]
                trial_classes = [c for j, c in enumerate(best_classes) if j != i]
                trial_L = compute_L(trial_classes, min(L_target + 100, 100000))
                if trial_L >= best_L and len(trial_classes) < len(best_classes):
                    # Removing a class doesn't hurt coverage and frees budget
                    best_classes = trial_classes
                    best_L = trial_L
                    used_primes.discard(p)
                    remaining_budget += recip[p]
                    improved = True

            if not improved:
                break

        return best_classes

    # Main algorithm:
    # 1. Start with a small set of primes (e.g., primes >= 13) and use greedy to cover 1..L
    # 2. Use local search to improve
    # 3. Try adding smaller primes (11, 7, 5, 3, 2) to extend coverage

    # Phase 1: Greedy construction with primes >= 13
    initial_primes = [p for p in primes if p >= 13]
    L_target = 2000
    classes, remaining_budget = greedy_build(L_target, initial_primes, Fraction(1, 1), 30)

    if not classes:
        # Fallback: return a simple solution
        return [[p, 0] for p in initial_primes[:10]]

    current_L = compute_L(classes, 100000)

    # Phase 2: Local search
    classes = local_search(classes, current_L + 100, primes, Fraction(1, 1), 30)
    current_L = compute_L(classes, 100000)

    # Phase 3: Try to extend by adding smaller primes
    small_primes = [p for p in primes if p < 13]
    small_primes.sort(reverse=True)  # 11, 7, 5, 3, 2

    for p in small_primes:
        if time.time() > deadline - 10:
            break
        if remaining_budget < recip[p]:
            continue

        # Try all residues for this prime
        best_r = None
        best_L_new = current_L
        for r in range(p):
            trial_classes = classes + [[p, r]]
            trial_L = compute_L(trial_classes, min(current_L + 1000, 100000))
            if trial_L > best_L_new:
                best_L_new = trial_L
                best_r = r

        if best_r is not None:
            classes.append([p, best_r])
            remaining_budget -= recip[p]
            current_L = best_L_new

            # Run local search again
            classes = local_search(classes, current_L + 100, primes, Fraction(1, 1), 20)
            current_L = compute_L(classes, 100000)

    # Final verification
    final_L = compute_L(classes, 100000)

    # Verify budget constraint
    total_budget = sum(recip[p] for p, r in classes)
    if total_budget > Fraction(1, 1):
        # This shouldn't happen, but just in case
        # Remove classes until budget is satisfied
        while total_budget > Fraction(1, 1) and classes:
            classes.pop()
            total_budget = sum(recip[p] for p, r in classes)
        final_L = compute_L(classes, 100000)

    return classes

# EVOLVE-BLOCK-END
