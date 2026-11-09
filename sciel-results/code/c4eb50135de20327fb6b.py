"""Program to construct a set of 12 distinct positive integers with distinct subset sums, minimizing the largest element."""
import time
import random

# EVOLVE-BLOCK-START

def _check_valid(elements):
    """Check if all 4096 subset sums are distinct. Returns True if valid."""
    subset_sums = {0}
    for elem in elements:
        new_sums = set()
        for s in subset_sums:
            ns = s + elem
            if ns in subset_sums:
                return False
            new_sums.add(ns)
        subset_sums = subset_sums | new_sums
    return len(subset_sums) == 4096

def _greedy_construct():
    """Greedy minimal extension: pick smallest valid next element."""
    elements = [1]
    subset_sums = {0, 1}
    for _ in range(11):
        candidate = 2
        while True:
            has_collision = False
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    has_collision = True
                    break
            if not has_collision:
                break
            candidate += 1
        elements.append(candidate)
        new_sums = set()
        for s in subset_sums:
            new_sums.add(s + candidate)
        subset_sums = subset_sums | new_sums
    return elements

def construct():
    """
    Randomized local search: start from greedy solution, then use simulated
    annealing / random restarts to find a set with smaller maximum element.
    """
    deadline = time.time() + 80  # safety margin before 90s kill
    rng = random.Random(42)

    # Phase 1: Get greedy solution as initial best
    best = _greedy_construct()
    best_max = max(best)
    best_sorted = sorted(best)

    # Phase 2: Local search - try to reduce the largest element
    # Strategy: repeatedly pick a random element to replace with a smaller value,
    # check validity, and keep improvements.
    
    current = best_sorted[:]
    current_max = best_max

    # Precompute for speed: track subset sums incrementally is complex,
    # so we'll use the full check but optimize the search space.

    # Key insight: for a "complete sequence" (distinct subset sums),
    # the elements must satisfy: if sorted a1 < a2 < ... < an,
    # then ai <= 1 + sum(a1..a_{i-1}) for the sequence to be "complete"
    # (all sums from 0 to total are achievable). But we don't need completeness,
    # just distinctness. However, a necessary condition for distinctness is:
    # ai <= 1 + sum of all other elements (otherwise ai can't be formed by others,
    # but that's not quite right either).
    #
    # Better approach: just do randomized search.

    # Try multiple strategies:
    # 1. Replace the largest element with a smaller one
    # 2. Replace any element with a smaller one
    # 3. Random perturbation

    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break

        # Pick a random element to try replacing
        idx = rng.randint(0, len(current) - 1)
        old_val = current[idx]

        # Try to replace with a smaller value
        # Bias toward replacing the largest elements
        if idx == len(current) - 1 or rng.random() < 0.7:
            # Try smaller values
            # Sample from a range below old_val
            if old_val <= 1:
                no_improve_count += 1
                continue
            # Try a random smaller value, biased toward close to old_val
            # or try the smallest possible
            if rng.random() < 0.3:
                # Try smallest unused value
                used = set(current)
                new_val = 1
                while new_val in used:
                    new_val += 1
            else:
                # Random value less than old_val
                new_val = rng.randint(1, old_val - 1)
                if new_val in set(current):
                    no_improve_count += 1
                    continue
        else:
            # Random element, try smaller or larger
            if rng.random() < 0.5:
                if old_val <= 1:
                    no_improve_count += 1
                    continue
                new_val = rng.randint(1, old_val - 1)
                if new_val in set(current):
                    no_improve_count += 1
                    continue
            else:
                # Try slightly larger
                new_val = old_val + rng.randint(1, 5)
                if new_val in set(current):
                    no_improve_count += 1
                    continue

        # Create candidate
        candidate = current[:]
        candidate[idx] = new_val
        candidate = sorted(candidate)

        # Quick check: are all elements distinct and positive?
        if len(set(candidate)) != len(candidate) or any(x <= 0 for x in candidate):
            no_improve_count += 1
            continue

        # Check validity
        if _check_valid(candidate):
            cand_max = max(candidate)
            if cand_max < current_max:
                current = candidate
                current_max = cand_max
                no_improve_count = 0
                if current_max < best_max:
                    best_max = current_max
                    best_sorted = current[:]
            else:
                # Accept with some probability (simulated annealing style)
                # to escape local optima
                if rng.random() < 0.1:
                    current = candidate
                    current_max = cand_max
                    no_improve_count += 1
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1

    return best_sorted

# EVOLVE-BLOCK-END
