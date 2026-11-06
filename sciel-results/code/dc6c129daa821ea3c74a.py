"""Rewritten covering-class search with faster local search and smarter prime selection."""
import time
import numpy as np
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    primes = _sieve(1000)
    max_check = 100000

    def compute_L(covered):
        """Compute the largest L such that 1..L are all covered."""
        sub = covered[1:]
        false_indices = np.where(~sub)[0]
        if len(false_indices) == 0:
            return max_check
        return int(false_indices[0])

    def apply_residue(covered, p, r):
        """Apply a residue class to the covered array (in-place)."""
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        covered[start::p] = True

    def remove_residue(covered, p, r):
        """Remove a residue class from the covered array (in-place)."""
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        covered[start::p] = False

    def greedy_residues_numpy(prime_list, max_check=100000):
        """Assign residues greedily using numpy for speed."""
        if not prime_list:
            return []

        covered = np.zeros(max_check + 1, dtype=bool)
        classes = []

        for p in prime_list:
            if time.time() > deadline:
                break

            # Find the first uncovered number
            sub = covered[1:]
            false_indices = np.where(~sub)[0]
            if len(false_indices) == 0:
                classes.append([p, 0])
                continue

            first_uncovered = int(false_indices[0])
            r_target = first_uncovered % p
            best_r = r_target
            best_L = first_uncovered - 1

            # For small primes, try all residues; for large, try target + neighbors
            if p <= 30:
                residues_to_try = list(range(p))
            else:
                residues_to_try = [r_target]
                for d in range(-5, 6):
                    r_alt = (r_target + d) % p
                    if r_alt != r_target:
                        residues_to_try.append(r_alt)

            for r in residues_to_try:
                if r == 0:
                    start = p
                else:
                    start = r
                if start < 1:
                    start += p
                mask = np.zeros(max_check + 1, dtype=bool)
                mask[start::p] = True
                new_covered = covered | mask
                new_L = compute_L(new_covered)
                if new_L > best_L:
                    best_L = new_L
                    best_r = r

            classes.append([p, int(best_r)])

            # Update covered array
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            covered[start::p] = True

        return classes

    def optimize_residues_fast(classes, max_check=100000, time_limit=None):
        """
        Fast local search: cyclic coordinate descent with smart candidate selection.
        """
        if not classes:
            return classes, 0

        if time_limit is None:
            time_limit = deadline - 5

        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            apply_residue(covered, p, r)

        current_L = compute_L(covered)

        while time.time() < time_limit:
            any_improvement = False
            for idx in range(len(classes)):
                if time.time() > time_limit:
                    break

                p, r = classes[idx]

                # Remove current residue
                remove_residue(covered, p, r)
                L_without = compute_L(covered)

                # Determine candidates
                if p <= 20:
                    candidates = list(range(p))
                else:
                    # Focus around the gap
                    candidates = [r]
                    for d in range(-10, 11):
                        alt = (r + d) % p
                        if alt != r:
                            candidates.append(alt)
                    # Add random samples
                    for _ in range(min(15, p // 10)):
                        rand_r = random.randint(0, p - 1)
                        if rand_r not in candidates:
                            candidates.append(rand_r)

                best_r = r
                best_L = current_L

                for new_r in candidates:
                    if new_r == r:
                        continue

                    apply_residue(covered, p, new_r)
                    new_L = compute_L(covered)

                    if new_L > best_L:
                        best_L = new_L
                        best_r = new_r
                    else:
                        remove_residue(covered, p, new_r)

                # Apply the best residue found
                if best_r != r:
                    if best_r != r:
                        # Check if best_r is already applied
                        test_start = best_r if best_r != 0 else p
                        if test_start >= 1 and test_start <= max_check:
                            if not covered[test_start]:
                                apply_residue(covered, p, best_r)
                        else:
                            apply_residue(covered, p, best_r)

                    current_L = best_L
                    classes[idx][1] = best_r
                    any_improvement = True
                else:
                    # No improvement, ensure original is applied
                    test_start = r if r != 0 else p
                    if test_start >= 1 and test_start <= max_check:
                        if not covered[test_start]:
                            apply_residue(covered, p, r)

            if not any_improvement:
                break

        return classes, current_L

    def swap_optimization(classes, max_check=100000, time_limit=None):
        """
        Try swapping residues between pairs of primes to escape local optima.
        """
        if not classes or len(classes) < 2:
            return classes, 0

        if time_limit is None:
            time_limit = deadline - 5

        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            apply_residue(covered, p, r)

        current_L = compute_L(covered)

        n = len(classes)
        while time.time() < time_limit:
            any_improvement = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        break

                    p_i, r_i = classes[i]
                    p_j, r_j = classes[j]

                    # Try swapping residues
                    remove_residue(covered, p_i, r_i)
                    remove_residue(covered, p_j, r_j)
                    apply_residue(covered, p_i, r_j)
                    apply_residue(covered, p_j, r_i)

                    new_L = compute_L(covered)

                    if new_L > current_L:
                        current_L = new_L
                        classes[i][1] = r_j
                        classes[j][1] = r_i
                        any_improvement = True
                    else:
                        # Revert
                        remove_residue(covered, p_i, r_j)
                        remove_residue(covered, p_j, r_i)
                        apply_residue(covered, p_i, r_i)
                        apply_residue(covered, p_j, r_j)

            if not any_improvement:
                break

        return classes, current_L

    best_classes = []
    best_L = 0

    def evaluate_strategy(selected_primes, optimize_time=10):
        nonlocal best_classes, best_L
        if time.time() > deadline - 5:
            return
        if not selected_primes:
            return

        # 1. Greedy Init
        classes = greedy_residues_numpy(selected_primes, max_check)

        # 2. Optimize
        remaining = deadline - time.time()
        opt_time = min(remaining - 5, optimize_time)
        if opt_time < 1:
            return

        classes, L = optimize_residues_fast(classes, max_check, time_limit=time.time() + opt_time)

        # 3. Swap optimization if time permits
        remaining = deadline - time.time()
        if remaining > 10:
            swap_time = min(remaining - 5, 8)
            if swap_time > 2:
                classes, L2 = swap_optimization(classes, max_check, time_limit=time.time() + swap_time)
                if L2 > L:
                    L = L2

        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]

    # Generate Candidate Strategies

    # Strategy 1: Smallest primes (Dense)
    primes_asc = sorted(primes)
    budget = Fraction(1)
    selected = []
    for p in primes_asc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    evaluate_strategy(selected, optimize_time=25)

    # Strategy 2: Largest primes (Many)
    primes_desc = sorted(primes, reverse=True)
    budget = Fraction(1)
    selected = []
    for p in primes_desc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    evaluate_strategy(selected, optimize_time=25)

    # Strategy 2b: Medium primes (balanced)
    if time.time() < deadline - 40:
        primes_mid = sorted(primes, key=lambda x: abs(x - 100))
        budget = Fraction(1)
        selected = []
        for p in primes_mid:
            cost = Fraction(1, p)
            if budget >= cost:
                budget -= cost
                selected.append(p)
            else:
                break
        evaluate_strategy(selected, optimize_time=20)

    # Strategy 3: Mixed (Small core, large fill)
    if time.time() < deadline - 30:
        for num_small in [2, 3, 4, 5]:
            if time.time() > deadline - 30:
                break
            small_primes = primes_asc[:num_small]
            budget = Fraction(1)
            for p in small_primes:
                budget -= Fraction(1, p)
            if budget < 0:
                continue

            selected = list(small_primes)
            for p in primes_desc:
                if p in small_primes:
                    continue
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)

            evaluate_strategy(selected, optimize_time=15)

    # Strategy 4: Randomized
    if time.time() < deadline - 25:
        random.seed(42)
        for restart in range(8):
            if time.time() > deadline - 25:
                break

            random_primes = primes.copy()
            random.shuffle(random_primes)
            budget = Fraction(1)
            selected = []
            for p in random_primes:
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)
                else:
                    continue

            if selected:
                evaluate_strategy(selected, optimize_time=12)

    # Strategy 5: Hybrid with weighted selection
    if time.time() < deadline - 30:
        weights = [1.0 / p for p in primes]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        random.seed(123)
        for restart in range(6):
            if time.time() > deadline - 30:
                break

            selected = []
            budget = Fraction(1)
            used = set()

            for p in primes:
                if p in used:
                    continue
                if random.random() < 0.3:
                    cost = Fraction(1, p)
                    if budget >= cost:
                        budget -= cost
                        selected.append(p)
                        used.add(p)

            if selected:
                evaluate_strategy(selected, optimize_time=15)

    # Strategy 6: Focused on small primes with targeted optimization
    if time.time() < deadline - 20:
        # Use only primes up to 100, which are cheap and dense
        small_primes = [p for p in primes if p <= 100]
        budget = Fraction(1)
        selected = []
        for p in small_primes:
            cost = Fraction(1, p)
            if budget >= cost:
                budget -= cost
                selected.append(p)
        evaluate_strategy(selected, optimize_time=20)

    # Strategy 7: Primes up to 200
    if time.time() < deadline - 15:
        mid_primes = [p for p in primes if p <= 200]
        budget = Fraction(1)
        selected = []
        for p in mid_primes:
            cost = Fraction(1, p)
            if budget >= cost:
                budget -= cost
                selected.append(p)
        evaluate_strategy(selected, optimize_time=15)

    return best_classes if best_classes else [[2, 1]]

# EVOLVE-BLOCK-END
