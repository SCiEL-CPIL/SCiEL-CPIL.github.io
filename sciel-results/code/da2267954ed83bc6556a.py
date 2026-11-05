import time
import random
import itertools

def _is_sidon(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _has_sidon_subset(A, k):
    """Check if A has ANY Sidon subset of size k."""
    for combo in itertools.combinations(A, k):
        if _is_sidon(combo):
            return True
    return False

def _count_sidon_subsets(A, k):
    """Count the number of Sidon subsets of size k in A."""
    count = 0
    for combo in itertools.combinations(A, k):
        if _is_sidon(combo):
            count += 1
    return count

def _max_sidon_size(A):
    """Compute the maximum size of a Sidon subset in A."""
    n = len(A)
    for k in range(n, 0, -1):
        if _has_sidon_subset(A, k):
            return k
    return 0

def _soft_cost(A, k_target):
    """
    Soft cost: sum of exp(-lambda * count) for k_target and k_target-1.
    Lower is better. This is smooth-ish.
    Actually, let's just use a weighted count:
    cost = 10 * count(k_target) + 1 * count(k_target - 1)
    """
    c_t = _count_sidon_subsets(A, k_target)
    c_t1 = _count_sidon_subsets(A, k_target - 1)
    return 10 * c_t + c_t1

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, keep explicit deadline with margin
    deadline = start_time + 80.0

    # Initial candidate: [0, 1, ..., 11]
    current_A = list(range(12))
    best_A = list(current_A)
    best_val = _max_sidon_size(best_A)  # 4

    # Target: drive count of 4-Sidon subsets to 0
    target_k = best_val  # 4
    current_cost = _soft_cost(current_A, target_k)
    best_cost = current_cost

    # Track best "hard" improvement separately
    best_hard_val = best_val
    best_hard_A = list(best_A)

    iterations = 0
    no_improve = 0
    rng = random.Random(42)

    while time.time() < deadline:
        iterations += 1

        # Perturb: pick 1-2 random positions and change them
        A_new = list(current_A)
        num_changes = rng.randint(1, 2)
        changed = set()
        for _ in range(num_changes):
            idx = rng.randint(0, 11)
            if idx in changed:
                continue
            changed.add(idx)

            # Choose new value
            if rng.random() < 0.5:
                # Small step
                step = rng.randint(-3, 3)
                new_val = A_new[idx] + step
            else:
                # Random in range, biased toward small values (denser = more collisions)
                new_val = rng.randint(0, 40)

            # Ensure validity
            if new_val < 0:
                new_val = 0
            if new_val > 128:
                new_val = 128

            # Ensure distinctness
            if new_val in A_new:
                # Try a few times to find a distinct value
                found = False
                for attempt in range(5):
                    new_val = rng.randint(0, 40)
                    if new_val not in A_new:
                        found = True
                        break
                if not found:
                    new_val = rng.randint(0, 128)
                    if new_val in A_new:
                        continue

            A_new[idx] = new_val

        # Evaluate
        new_val_score = _max_sidon_size(A_new)

        if new_val_score < best_hard_val:
            # Hard improvement!
            best_hard_val = new_val_score
            best_hard_A = list(A_new)
            best_A = list(A_new)
            best_val = new_val_score
            target_k = max(2, new_val_score)
            current_cost = _soft_cost(current_A, target_k)
            best_cost = current_cost
            no_improve = 0

            if best_hard_val <= 3:
                # Great! Return immediately
                return best_hard_A
        else:
            # No hard improvement; use soft cost
            new_cost = _soft_cost(A_new, target_k)

            if new_cost < best_cost:
                current_A = A_new
                current_cost = new_cost
                best_cost = new_cost
                no_improve = 0
            elif new_cost == best_cost and rng.random() < 0.3:
                # Accept equal-cost move to explore
                current_A = A_new
            else:
                # Simulated annealing: accept worse with some probability
                if rng.random() < 0.1:
                    current_A = A_new
                    current_cost = new_cost
                no_improve += 1

        # Periodic reset if stuck for too long
        if no_improve > 200:
            # Reset to best known and try a different strategy
            current_A = list(best_A)
            current_cost = _soft_cost(current_A, target_k)
            no_improve = 0

            # Try a different initial structure: dense cluster
            if rng.random() < 0.3:
                # Dense: all in [0, 20]
                current_A = sorted(rng.sample(range(0, 21), 12))
                current_cost = _soft_cost(current_A, target_k)

    return best_A
