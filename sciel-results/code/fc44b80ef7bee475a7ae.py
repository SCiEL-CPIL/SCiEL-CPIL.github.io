"""Rewritten program: simulated-annealing over integer Sidon sets with 0,
covering differences 1..30, minimizing the maximum element."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

# ---------------------------------------------------------------------------
# Fast helpers
# ---------------------------------------------------------------------------

def _diffs_set(s):
    """Return set of all positive differences of sorted list s."""
    n = len(s)
    d = set()
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            d.add(s[j] - si)
    return d

def _is_sidon(s):
    """True iff all positive differences are distinct."""
    n = len(s)
    seen = set()
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            d = s[j] - si
            if d in seen:
                return False
            seen.add(d)
    return True

def _covers_1_to_30(s):
    d = _diffs_set(s)
    for x in range(1, 31):
        if x not in d:
            return False
    return True

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

# ---------------------------------------------------------------------------
# Greedy baseline (fast, deterministic)
# ---------------------------------------------------------------------------

def _greedy_construct():
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs_list(sorted(a)):
            continue
        found = False
        candidates = []
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    candidates.append(x)
        candidates = sorted(set(candidates))
        for x in candidates:
            trial = sorted(a + [x])
            dl = diffs_list(trial)
            if len(set(dl)) == len(dl):
                a = trial
                found = True
                break
        if not found:
            y = max(a) + 1
            while True:
                trial = sorted(a + [y, y + d])
                dl = diffs_list(trial)
                if len(set(dl)) == len(dl):
                    a = trial
                    break
                y += 1
    return a

# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------

def _energy(s):
    """
    Composite energy:
      - large penalty for each duplicate difference (Sidon violation)
      - penalty for each uncovered target in 1..30
      - the maximum element (diameter)
    Lower is better.
    """
    n = len(s)
    if n < 1:
        return 10**9
    seen = set()
    dup_count = 0
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            d = s[j] - si
            if d in seen:
                dup_count += 1
            else:
                seen.add(d)
    uncovered = sum(1 for x in range(1, 31) if x not in seen)
    diameter = s[-1]
    # Weight: duplicates are severe, uncovered moderate, diameter primary
    return 1000 * dup_count + 50 * uncovered + diameter

def _simulated_annealing():
    rng = random.Random(42)
    best = None
    best_val = 10**9
    best_energy = 10**9

    # Multiple restarts
    for restart in range(30):
        if time.time() > _DEADLINE:
            break

        # Initialize: start from greedy result or random small Sidon set
        if restart == 0:
            cur = _greedy_construct()
        else:
            # Random initialization: small Sidon set containing 0
            size = rng.randint(10, 14)
            upper = rng.randint(80, 250)
            cur = [0]
            attempts = 0
            while len(cur) < size and attempts < 500:
                x = rng.randint(1, upper)
                trial = sorted(cur + [x])
                if _is_sidon(trial):
                    cur = trial
                attempts += 1
            if len(cur) < 8:
                continue

        cur = sorted(cur)
        cur_e = _energy(cur)
        best_in_run = cur[:]
        best_e_in_run = cur_e

        T = 200.0
        T_min = 0.5
        cooling = 0.995
        n_steps = 50000
        max_steps = 200000

        step = 0
        while step < max_steps and T > T_min:
            if time.time() > _DEADLINE:
                break

            # Choose a mutation
            op = rng.random()
            if op < 0.35 and len(cur) < 32:
                # Add a random element
                hi = max(cur) + rng.randint(1, 40)
                x = rng.randint(1, hi)
                if x not in cur:
                    trial = sorted(cur + [x])
                    trial_e = _energy(trial)
                    if trial_e <= cur_e or rng.random() < _accept_prob(trial_e - cur_e, T):
                        cur = trial
                        cur_e = trial_e
            elif op < 0.60 and len(cur) > 8:
                # Remove a random non-zero element
                idx = rng.randint(1, len(cur) - 1)
                trial = cur[:idx] + cur[idx + 1:]
                trial_e = _energy(trial)
                if trial_e <= cur_e or rng.random() < _accept_prob(trial_e - cur_e, T):
                    cur = trial
                    cur_e = trial_e
            elif op < 0.85:
                # Shift a random non-zero element by a small delta
                idx = rng.randint(1, len(cur) - 1)
                delta = rng.randint(-15, 15)
                if delta == 0:
                    continue
                new_val = cur[idx] + delta
                if new_val <= 0:
                    continue
                if new_val in cur:
                    continue
                trial = cur[:idx] + [new_val] + cur[idx + 1:]
                trial = sorted(trial)
                if len(trial) != len(set(trial)):
                    continue
                trial_e = _energy(trial)
                if trial_e <= cur_e or rng.random() < _accept_prob(trial_e - cur_e, T):
                    cur = trial
                    cur_e = trial_e
            else:
                # Swap two non-zero elements
                if len(cur) < 3:
                    continue
                i, j = rng.sample(range(1, len(cur)), 2)
                trial = cur[:]
                trial[i], trial[j] = trial[j], trial[i]
                trial = sorted(trial)
                if len(trial) != len(set(trial)):
                    continue
                trial_e = _energy(trial)
                if trial_e <= cur_e or rng.random() < _accept_prob(trial_e - cur_e, T):
                    cur = trial
                    cur_e = trial_e

            step += 1
            T *= cooling

            # Track best
            if cur_e < best_e_in_run:
                best_e_in_run = cur_e
                best_in_run = cur[:]

            # If we have a valid solution, check diameter
            if cur_e < 100:  # likely valid or close
                if _valid(cur):
                    val = max(cur)
                    if val < best_val:
                        best_val = val
                        best = cur[:]
                        best_energy = cur_e

        # After this run, check best_in_run
        if _valid(best_in_run):
            val = max(best_in_run)
            if val < best_val:
                best_val = val
                best = best_in_run[:]
                best_energy = _energy(best_in_run)

    return best

def _accept_prob(delta_e, T):
    if delta_e <= 0:
        return 1.0
    if T <= 0:
        return 0.0
    return min(1.0, 2.718281828459045 ** (-delta_e / T))

# ---------------------------------------------------------------------------
# Local search refinement on a valid set
# ---------------------------------------------------------------------------

def _local_search_refine(start):
    """Try to reduce the maximum element of a valid Sidon set."""
    if start is None:
        return None
    rng = random.Random(999)
    cur = sorted(start)
    cur_val = max(cur)
    best = cur[:]
    best_val = cur_val

    for step in range(20000):
        if time.time() > _DEADLINE:
            break
        if not _valid(cur):
            break

        # Try to reduce the largest element
        idx = len(cur) - 1
        # Try shifting the largest element down
        improved = False
        for delta in range(1, 20):
            new_val = cur[idx] - delta
            if new_val <= 0:
                break
            if new_val in cur:
                continue
            trial = cur[:idx] + [new_val] + cur[idx + 1:]
            trial = sorted(trial)
            if _is_sidon(trial) and _covers_1_to_30(trial):
                cur = trial
                cur_val = max(cur)
                if cur_val < best_val:
                    best_val = cur_val
                    best = cur[:]
                improved = True
                break

        if not improved:
            # Try shifting other elements to free up space
            idx2 = rng.randint(1, len(cur) - 2)
            delta = rng.randint(-10, 10)
            if delta == 0:
                continue
            new_val = cur[idx2] + delta
            if new_val <= 0:
                continue
            if new_val in cur:
                continue
            trial = cur[:idx2] + [new_val] + cur[idx2 + 1:]
            trial = sorted(trial)
            if _is_sidon(trial) and _covers_1_to_30(trial):
                cur = trial
                cur_val = max(cur)
                if cur_val < best_val:
                    best_val = cur_val
                    best = cur[:]

    return best

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()

    # Start with greedy baseline
    g = _greedy_construct()
    best = g
    best_val = max(g)

    # Simulated annealing
    sa = _simulated_annealing()
    if sa is not None:
        if _valid(sa) and max(sa) < best_val:
            best = sa
            best_val = max(sa)

    # Local search refinement
    if best is not None:
        refined = _local_search_refine(best)
        if refined is not None and max(refined) < best_val:
            best = refined
            best_val = max(refined)

    return best

# EVOLVE-BLOCK-END
