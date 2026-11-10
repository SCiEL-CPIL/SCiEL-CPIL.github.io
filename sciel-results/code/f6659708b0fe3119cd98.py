import random
import itertools
import time
from collections import defaultdict


def _is_sidon(subset):
    """Check if a list of distinct ints is a Sidon set (pair sums unique, incl. diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _count_collisions(A):
    """Count the number of distinct sum values that are produced by 2+ distinct pairs.
    Lower is better: 0 collisions means every pair-sum is unique (the whole set is Sidon).
    We use this as a smooth-ish surrogate for the max-Sidon-subset objective."""
    sum_count = defaultdict(int)
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            sum_count[s] += 1
    return sum(1 for c in sum_count.values() if c >= 2)


def _max_sidon_subset_size(A):
    """Exhaustively find the largest Sidon subset of A (A has 12 elements)."""
    n = len(A)
    # Try from largest down
    for k in range(n, 1, -1):
        for subset in itertools.combinations(A, k):
            if _is_sidon(list(subset)):
                return k
    # At minimum, any single element is a Sidon set of size 1;
    # any two distinct elements always have unique pair sums? No: {a,b} has sums a+a, a+b, b+b which are distinct if a!=b.
    return 2


def _local_search(A, deadline, rng):
    """Local search minimizing collision count via single-element swaps."""
    n = len(A)
    A = list(A)
    current_collisions = _count_collisions(A)
    best_A = list(A)
    best_collisions = current_collisions
    used = set(A)
    universe = list(range(129))

    # Simulated-annealing-ish: accept worse moves early
    T = 5.0
    T_min = 0.1
    cooling = 0.995

    step = 0
    while time.time() < deadline:
        step += 1
        # Pick a random index to replace
        i = rng.randrange(n)
        old_val = A[i]
        # Candidate new values: random sample from available
        available = [x for x in universe if x not in used]
        if not available:
            break
        new_val = rng.choice(available)

        # Apply
        A[i] = new_val
        used.remove(old_val)
        used.add(new_val)
        new_collisions = _count_collisions(A)

        # Accept?
        delta = new_collisions - current_collisions
        if delta <= 0 or (T > T_min and rng.random() < pow(2.718281828, -delta / max(T, 1e-9))):
            current_collisions = new_collisions
            if new_collisions < best_collisions:
                best_collisions = new_collisions
                best_A = list(A)
                if best_collisions == 0:
                    break
        else:
            # Revert
            A[i] = old_val
            used.remove(new_val)
            used.add(old_val)

        T *= cooling

    return best_A, best_collisions


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (evaluator kills at 90s)
    rng = random.Random(12345)

    # Initial candidate: seed set
    seed_set = list(range(12))
    best_A = list(seed_set)
    best_collisions = _count_collisions(seed_set)

    # Track the best by actual max-Sidon-subset-size when feasible
    best_score = _max_sidon_subset_size(seed_set)
    best_by_score = list(seed_set)

    # Multi-restart local search
    max_restarts = 40
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        if restart == 0:
            A = list(seed_set)
        else:
            A = rng.sample(range(129), 12)

        # Local search on this starting point
        candidate, cand_collisions = _local_search(A, deadline - 5, rng)

        # Evaluate candidate by actual max Sidon subset size (exact, but 12 choose k is small)
        cand_score = _max_sidon_subset_size(candidate)

        if cand_score < best_score:
            best_score = cand_score
            best_by_score = list(candidate)
            if best_score <= 3:
                # Can't do better than 3 realistically (any 3 distinct ints form a Sidon set? 
                # {a,b,c}: sums 2a, a+b, a+c, 2b, b+c, 2c. Need all 6 distinct.
                # If a<b<c and c-a > b-a and 2c - (a+b) > 0 etc. Not always, but often.
                # Score 3 is a great result.)
                break
        elif cand_score == best_score and cand_collisions < best_collisions:
            best_collisions = cand_collisions
            best_A = list(candidate)
            # Also update best_by_score if collisions tie-break
            if cand_collisions < _count_collisions(best_by_score):
                best_by_score = list(candidate)

    # Final: return the best by score; if tied, prefer lower collisions
    # Re-evaluate best_by_score to be safe
    final_score = _max_sidon_subset_size(best_by_score)
    if final_score <= best_score:
        return best_by_score
    else:
        return best_A


if __name__ == "__main__":
    result = construct()
    print("Candidate:", result)
    print("Max Sidon subset size:", _max_sidon_subset_size(result))
    print("Collisions:", _count_collisions(result))
