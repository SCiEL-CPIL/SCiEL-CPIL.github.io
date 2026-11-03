"""Construct a finite set U of nonnegative integers (containing 0) to maximize
   1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
"""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

_DEADLINE = 110.0  # safety margin before 120s kill


def _sumset_size(U):
    """Return number of distinct sums a+b for a,b in U."""
    sums = set()
    n = len(U)
    for i in range(n):
        ai = U[i]
        for j in range(i, n):
            sums.add(ai + U[j])
    return len(sums)


def _diffset_size(U):
    """Return number of distinct differences a-b for a,b in U."""
    diffs = set()
    n = len(U)
    for i in range(n):
        ai = U[i]
        for j in range(n):
            diffs.add(ai - U[j])
    return len(diffs)


def _score(U):
    """Compute the task score for a set U (list of nonnegative ints, 0 in U)."""
    if not U or 0 not in U:
        return -1.0
    U = sorted(set(U))
    mx = U[-1]
    if mx == 0:
        return 1.0
    ss = _sumset_size(U)
    ds = _diffset_size(U)
    if ss == 0:
        return 1.0
    ratio = ds / ss
    if ratio <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)


class _State:
    """Incrementally maintained sum/diff sets for a set U (containing 0)."""
    __slots__ = ("U", "sums", "diffs")

    def __init__(self, U):
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        self.U = U
        self.sums = set()
        self.diffs = set()
        n = len(U)
        for i in range(n):
            ai = U[i]
            for j in range(i, n):
                self.sums.add(ai + U[j])
            for j in range(n):
                self.diffs.add(ai - U[j])

    def add(self, x):
        """Add element x to U, updating sums/diffs incrementally."""
        U = self.U
        n = len(U)
        for i in range(n):
            ai = U[i]
            self.sums.add(ai + x)
            self.diffs.add(ai - x)
            self.diffs.add(x - ai)
        self.U = U + [x]


def _score_from_state(st):
    U = st.U
    mx = U[-1]
    if mx == 0:
        return 1.0
    ss = len(st.sums)
    ds = len(st.diffs)
    if ss == 0:
        return 1.0
    ratio = ds / ss
    if ratio <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)


def _greedy_maximal_independent(st, deadline, max_rounds=2000):
    """
    Graph-coloring-flavored greedy: at each step, pick the candidate integer x
    that maximizes net score improvement (dispersion gain in |U-U| vs collision
    penalty in |U+U|) while keeping max(U) bounded. Stop when no candidate
    improves the score or when the deadline is approached.
    """
    best_score = _score_from_state(st)
    best_U = list(st.U)
    t0 = time.time()

    # Candidate universe: small integers up to some bound, plus geometric-ish
    # values to encourage spread.
    max_val = max(200, 4 * st.U[-1]) if st.U[-1] > 0 else 200
    candidates = list(range(1, max_val + 1))
    # Add some geometric candidates beyond max_val
    v = max_val
    for _ in range(30):
        v *= 2
        candidates.append(v)
        if v > 10**7:
            break
    random.shuffle(candidates)

    rounds = 0
    while time.time() < deadline and rounds < max_rounds:
        rounds += 1
        current_score = _score_from_state(st)
        best_gain = 1e-9
        best_x = None

        # Evaluate a subset of candidates each round to keep cost down
        # We scan up to 200 candidates per round
        n_cand = min(len(candidates), 200)
        for x in candidates[:n_cand]:
            if x in st.U:
                continue
            # Fast pre-check: if x is very large, the denominator grows a lot
            new_mx = max(st.U[-1], x)
            # Compute new sums/diffs sizes incrementally (without modifying st)
            # New sums: old sums + {a + x for a in U}
            # New diffs: old diffs + {a - x for a in U} + {x - a for a in U}
            U = st.U
            n = len(U)
            new_sum_count = len(st.sums)
            new_diff_count = len(st.diffs)
            # Count new sums
            new_sums = 0
            for a in U:
                if (a + x) not in st.sums:
                    new_sums += 1
            # Count new diffs
            new_diffs = 0
            for a in U:
                if (a - x) not in st.diffs:
                    new_diffs += 1
                if (x - a) not in st.diffs:
                    new_diffs += 1
            new_ss = new_sum_count + new_sums
            new_ds = new_diff_count + new_diffs
            if new_ss == 0:
                continue
            ratio = new_ds / new_ss
            if ratio <= 0:
                continue
            new_score = 1.0 + math.log(ratio) / math.log(2 * new_mx + 1)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_x = x

        if best_x is None:
            # No improving candidate in this scan; try a fresh random subset
            random.shuffle(candidates)
            # If still no improvement after a few reshuffles, stop
            if rounds % 10 == 0 and rounds > 50:
                break
            continue

        # Commit the best candidate
        st.add(best_x)
        new_score = _score_from_state(st)
        if new_score > best_score:
            best_score = new_score
            best_U = list(st.U)
        else:
            # Shouldn't happen due to gain check, but revert if it does
            # (We can't easily revert, so just stop)
            break

    return best_U, best_score


def _local_search(U, time_limit):
    """Perform local search to improve score of U using greedy + swaps."""
    best_U = list(U)
    best_score = _score(best_U)
    t_start = time.time()

    # First, run the greedy maximal independent set
    st = _State(best_U)
    U_opt, s_opt = _greedy_maximal_independent(st, t_start + time_limit)
    if s_opt > best_score:
        best_score = s_opt
        best_U = U_opt

    # Then try some removals and replacements
    improved = True
    while improved and time.time() < t_start + time_limit:
        improved = False
        mx = best_U[-1]

        # Try removing an element
        for i, val in enumerate(best_U):
            if val == 0:
                continue
            if time.time() > t_start + time_limit:
                return best_U, best_score
            new_U = best_U[:i] + best_U[i + 1:]
            s = _score(new_U)
            if s > best_score:
                best_score = s
                best_U = new_U
                improved = True
                break

        # Try replacing an element with a random value
        if time.time() < t_start + time_limit - 2:
            for i, val in enumerate(best_U):
                if val == 0:
                    continue
                for _ in range(3):
                    new_val = random.randint(1, max(10, 2 * mx))
                    if new_val in best_U:
                        continue
                    new_U = best_U.copy()
                    new_U[i] = new_val
                    new_U.sort()
                    s = _score(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break

    return best_U, best_score


def construct():
    t0 = time.time()
    best_U = [0, 1, 3]
    best_score = _score(best_U)

    def consider(U):
        nonlocal best_U, best_score
        if time.time() > _DEADLINE - 3:
            return
        try:
            s = _score(U)
        except Exception:
            return
        if s > best_score:
            best_score = s
            best_U = list(U)

    # 1. Generate diverse initial candidates
    candidates = []

    # Geometric
    for base in range(2, 10):
        for k in range(2, 20):
            U = [0]
            v = 1
            for _ in range(k):
                v *= base
                U.append(v)
            candidates.append(U)

    # Quadratic
    for k in range(2, 20):
        U = [i * i for i in range(k)]
        candidates.append(U)

    # Triangular
    for k in range(2, 20):
        U = [i * (i + 1) // 2 for i in range(k)]
        candidates.append(U)

    # Random geometric-ish
    for seed in range(50):
        rng = random.Random(seed)
        U = [0]
        v = rng.randint(1, 5)
        U.append(v)
        for _ in range(rng.randint(3, 15)):
            v = int(v * rng.uniform(1.1, 2.0))
            U.append(v)
        candidates.append(sorted(set(U)))

    # 2. Local search on each candidate
    time_per_candidate = 0.5  # seconds
    for U in candidates:
        if time.time() > _DEADLINE - 10:
            break
        s_init = _score(U)
        if s_init < best_score - 0.1:
            continue

        U_opt, s_opt = _local_search(U, time_per_candidate)
        consider(U_opt)

    # 3. If time remains, do a global local search on the best found
    if time.time() < _DEADLINE - 5:
        U_opt, s_opt = _local_search(best_U, _DEADLINE - time.time() - 5)
        consider(U_opt)

    return best_U
# EVOLVE-BLOCK-END
