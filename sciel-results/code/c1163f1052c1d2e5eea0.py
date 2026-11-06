"""Sorting networks on 18 wires: correct baseline + fast randomized pruning."""
import random
import time

N = 18
DEADLINE_MARGIN = 8.0  # seconds of safety before the 90s kill


def _bitset_apply(net, state):
    """Apply compare-exchange network to a 18-bit state; bit i = wire i.

    compare-exchange (i, j): wire i gets min, wire j gets max.
    """
    for i, j in net:
        a = (state >> i) & 1
        b = (state >> j) & 1
        if a and not b:
            state = (state & ~(1 << i)) | (1 << j)
        elif b and not a:
            state = (state & ~(1 << j)) | (1 << i)
    return state


def _sorted_form(state):
    """Canonical sorted (ascending) form of a 0/1 input: 0s in low bits, 1s in high bits."""
    k = bin(state).count('1')
    return ((1 << N) - 1) ^ ((1 << (N - k)) - 1)


def _is_valid(net):
    """True iff net sorts every 0/1 input to its canonical sorted form."""
    for state in range(1 << N):
        if _bitset_apply(net, state) != _sorted_form(state):
            return False
    return True


def _odd_even_transposition():
    """Correct odd-even transposition network: N phases, N*(N-1)/2 = 153 comparators."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net


def construct():
    """Return a list of comparator pairs (i, j) that sorts all 18 inputs."""
    start = time.time()
    deadline = start + (90.0 - DEADLINE_MARGIN)

    # Correct baseline (153 comparators).
    best = _odd_even_transposition()
    best_valid = _is_valid(best)
    assert best_valid, "Baseline odd-even transposition must be valid."

    rng = random.Random(20240517)

    # Greedy redundant-comparator removal. Each removal is checked for validity.
    # A full validity check is 2^18 bitset applications; keep the count bounded.
    max_checks = 120
    checks_done = 0

    changed = True
    while changed and checks_done < max_checks:
        if time.time() > deadline:
            break
        changed = False
        for idx in range(len(best)):
            if time.time() > deadline or checks_done >= max_checks:
                break
            candidate = best[:idx] + best[idx + 1:]
            checks_done += 1
            if _is_valid(candidate):
                best = candidate
                changed = True
                break  # restart scan (indices shifted)

    # Escape local minima: add a random comparator, then re-prune.
    if checks_done < max_checks and time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline or checks_done >= max_checks:
                break
            i = rng.randrange(N)
            j = rng.randrange(i + 1, N)
            pos = rng.randrange(len(best) + 1)
            candidate = best[:pos] + [(i, j)] + best[pos:]
            checks_done += 1
            if _is_valid(candidate):
                # Try to prune any comparator from the augmented network.
                improved = False
                for idx in range(len(candidate)):
                    if time.time() > deadline or checks_done >= max_checks:
                        break
                    cand2 = candidate[:idx] + candidate[idx + 1:]
                    checks_done += 1
                    if _is_valid(cand2):
                        best = cand2
                        improved = True
                        break
                # If we did not improve, discard the augmentation.

    # Final safety check; fall back to the guaranteed-valid baseline if needed.
    if not _is_valid(best):
        best = _odd_even_transposition()

    return best
