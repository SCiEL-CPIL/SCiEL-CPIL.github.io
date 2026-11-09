"""Simulated-annealing search for two Sidon sets with disjoint difference sets."""

import time
import random
import math

N = 256
MAX_EACH = 64
DEADLINE = 85  # seconds budget (5s safety margin before 90s kill)


def _valid_pair(A, B):
    """Full verification: both Sidon, nonzero diffs disjoint, nonempty."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    # Sidon check for A
    sums = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            v = A[i] + A[j]
            if v in sums:
                return False
            sums.add(v)
    # Sidon check for B
    sums = set()
    n = len(B)
    for i in range(n):
        for j in range(i, n):
            v = B[i] + B[j]
            if v in sums:
                return False
            sums.add(v)
    # Diff disjointness
    dA = set()
    n = len(A)
    for i in range(n):
        for j in range(i):
            dA.add(A[i] - A[j])
    dB = set()
    n = len(B)
    for i in range(n):
        for j in range(i):
            dB.add(B[i] - B[j])
    if dA & dB:
        return False
    return True


def _score(A, B):
    return len(A) + len(B)


class _State:
    """Maintains A, B, and incremental sum/diff conflict counters."""

    def __init__(self, A, B):
        self.A = list(A)
        self.B = list(B)
        self.A_sums = set()
        self.B_sums = set()
        self.A_diffs = set()
        self.B_diffs = set()
        self._rebuild()

    def _rebuild(self):
        self.A_sums = set()
        self.B_sums = set()
        self.A_diffs = set()
        self.B_diffs = set()
        n = len(self.A)
        for i in range(n):
            for j in range(i, n):
                self.A_sums.add(self.A[i] + self.A[j])
        n = len(self.B)
        for i in range(n):
            for j in range(i, n):
                self.B_sums.add(self.B[i] + self.B[j])
        n = len(self.A)
        for i in range(n):
            for j in range(i):
                self.A_diffs.add(self.A[i] - self.A[j])
        n = len(self.B)
        for i in range(n):
            for j in range(i):
                self.B_diffs.add(self.B[i] - self.B[j])

    def _can_add_to_A(self, x):
        """Check if x can be added to A (Sidon + diff disjoint with B)."""
        if len(self.A) >= MAX_EACH:
            return False
        # Sidon: x+x and x+a for all a in A must not be in A_sums
        if x + x in self.A_sums:
            return False
        for a in self.A:
            if x + a in self.A_sums:
                return False
        # Diffs: |x - a| for all a in A must not be in A_diffs or B_diffs
        for a in self.A:
            d = abs(x - a)
            if d in self.A_diffs or d in self.B_diffs:
                return False
        return True

    def _can_add_to_B(self, x):
        if len(self.B) >= MAX_EACH:
            return False
        if x + x in self.B_sums:
            return False
        for b in self.B:
            if x + b in self.B_sums:
                return False
        for b in self.B:
            d = abs(x - b)
            if d in self.B_diffs or d in self.A_diffs:
                return False
        return True

    def _add_to_A(self, x):
        self.A.append(x)
        self.A.sort()
        self.A_sums.add(x + x)
        self.A_diffs.clear()
        n = len(self.A)
        for i in range(n):
            for j in range(i):
                self.A_diffs.add(self.A[i] - self.A[j])
        for a in self.A:
            if a != x:
                self.A_sums.add(x + a)

    def _add_to_B(self, x):
        self.B.append(x)
        self.B.sort()
        self.B_sums.add(x + x)
        self.B_diffs.clear()
        n = len(self.B)
        for i in range(n):
            for j in range(i):
                self.B_diffs.add(self.B[i] - self.B[j])
        for b in self.B:
            if b != x:
                self.B_sums.add(x + b)

    def _remove_from_A(self, x):
        idx = self.A.index(x)
        self.A.pop(idx)
        self._rebuild()

    def _remove_from_B(self, x):
        idx = self.B.index(x)
        self.B.pop(idx)
        self._rebuild()

    def soft_objective(self, beta=0.5):
        """
        Smooth surrogate: maximize |A|+|B| while penalizing blocked candidates.
        Uses log-sum-exp for soft count of elements that could be added.
        Higher is better.
        """
        sa = len(self.A)
        sb = len(self.B)
        # Count how many elements could be added to A and B
        blocked_A = 0
        blocked_B = 0
        for x in range(1, N + 1):
            if x not in self.A and not self._can_add_to_A(x):
                blocked_A += 1
            if x not in self.B and not self._can_add_to_B(x):
                blocked_B += 1
        # Soft penalty: fewer blocked = better
        # Use -beta * log(1 + blocked) as a smooth surrogate
        penalty = beta * (math.log(1 + blocked_A) + math.log(1 + blocked_B))
        return sa + sb - penalty


def _greedy_init(rng, time_budget):
    """Build initial A and B via greedy, then return State."""
    A = []
    B = []
    A_sums = set()
    B_sums = set()
    A_diffs = set()
    B_diffs = set()

    pool = list(range(1, N + 1))
    rng.shuffle(pool)

    for x in pool:
        if len(A) >= MAX_EACH:
            break
        ok = True
        if x + x in A_sums:
            ok = False
        else:
            for a in A:
                if x + a in A_sums:
                    ok = False
                    break
        if ok:
            for a in A:
                d = abs(x - a)
                if d in A_diffs:
                    ok = False
                    break
        if ok:
            A.append(x)
            A_sums.add(x + x)
            for a in A:
                if a != x:
                    A_sums.add(x + a)
            A_diffs.clear()
            n = len(A)
            for i in range(n):
                for j in range(i):
                    A_diffs.add(A[i] - A[j])
            A.sort()

    for x in pool:
        if len(B) >= MAX_EACH:
            break
        ok = True
        if x + x in B_sums:
            ok = False
        else:
            for b in B:
                if x + b in B_sums:
                    ok = False
                    break
        if ok:
            for b in B:
                d = abs(x - b)
                if d in B_diffs or d in A_diffs:
                    ok = False
                    break
        if ok:
            B.append(x)
            B_sums.add(x + x)
            for b in B:
                if b != x:
                    B_sums.add(x + b)
            B_diffs.clear()
            n = len(B)
            for i in range(n):
                for j in range(i):
                    B_diffs.add(B[i] - B[j])
            B.sort()

    return _State(A, B)


def _simulated_annealing(state, rng, time_limit):
    """
    Simulated annealing with swap/insert/remove moves.
    Uses smooth surrogate objective for acceptance.
    """
    best_A = list(state.A)
    best_B = list(state.B)
    best_score = _score(best_A, best_B)

    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    beta = 0.3

    while time.time() < time_limit and T > T_min:
        move_type = rng.random()
        improved = False

        if move_type < 0.3:
            # Insert: add random element to A or B
            side = rng.random() < 0.5
            if side == 0 and len(state.A) < MAX_EACH:
                candidates = [x for x in range(1, N + 1) if x not in state.A]
                if candidates:
                    x = rng.choice(candidates)
                    if state._can_add_to_A(x):
                        old_obj = state.soft_objective(beta)
                        state._add_to_A(x)
                        new_obj = state.soft_objective(beta)
                        delta = new_obj - old_obj
                        if delta > 0 or rng.random() < math.exp(delta / T):
                            improved = True
                        else:
                            state._remove_from_A(x)
            elif side == 1 and len(state.B) < MAX_EACH:
                candidates = [x for x in range(1, N + 1) if x not in state.B]
                if candidates:
                    x = rng.choice(candidates)
                    if state._can_add_to_B(x):
                        old_obj = state.soft_objective(beta)
                        state._add_to_B(x)
                        new_obj = state.soft_objective(beta)
                        delta = new_obj - old_obj
                        if delta > 0 or rng.random() < math.exp(delta / T):
                            improved = True
                        else:
                            state._remove_from_B(x)

        elif move_type < 0.6:
            # Swap: remove from one, add to other
            side = rng.random() < 0.5
            if side == 0 and state.A:
                # Remove random from A, try adding random to B
                old_obj = state.soft_objective(beta)
                x_remove = rng.choice(state.A)
                state._remove_from_A(x_remove)
                # Try to add to B
                candidates = [x for x in range(1, N + 1) if x not in state.B]
                if candidates:
                    x_add = rng.choice(candidates)
                    if state._can_add_to_B(x_add):
                        state._add_to_B(x_add)
                        new_obj = state.soft_objective(beta)
                        delta = new_obj - old_obj
                        if delta > 0 or rng.random() < math.exp(delta / T):
                            improved = True
                        else:
                            state._remove_from_B(x_add)
                            state._add_to_A(x_remove)
                    else:
                        state._add_to_A(x_remove)
            elif side == 1 and state.B:
                old_obj = state.soft_objective(beta)
                x_remove = rng.choice(state.B)
                state._remove_from_B(x_remove)
                candidates = [x for x in range(1, N + 1) if x not in state.A]
                if candidates:
                    x_add = rng.choice(candidates)
                    if state._can_add_to_A(x_add):
                        state._add_to_A(x_add)
                        new_obj = state.soft_objective(beta)
                        delta = new_obj - old_obj
                        if delta > 0 or rng.random() < math.exp(delta / T):
                            improved = True
                        else:
                            state._remove_from_A(x_add)
                            state._add_to_B(x_remove)
                    else:
                        state._add_to_B(x_remove)

        else:
            # Remove: drop random element (to unblock future additions)
            side = rng.random() < 0.5
            if side == 0 and len(state.A) > 1:
                old_obj = state.soft_objective(beta)
                x = rng.choice(state.A)
                state._remove_from_A(x)
                new_obj = state.soft_objective(beta)
                delta = new_obj - old_obj
                if delta > 0 or rng.random() < math.exp(delta / T):
                    improved = True
                else:
                    state._add_to_A(x)
            elif side == 1 and len(state.B) > 1:
                old_obj = state.soft_objective(beta)
                x = rng.choice(state.B)
                state._remove_from_B(x)
                new_obj = state.soft_objective(beta)
                delta = new_obj - old_obj
                if delta > 0 or rng.random() < math.exp(delta / T):
                    improved = True
                else:
                    state._add_to_B(x)

        T *= cooling

        # Track best
        cur_score = _score(state.A, state.B)
        if cur_score > best_score:
            best_score = cur_score
            best_A = list(state.A)
            best_B = list(state.B)

    return best_A, best_B


def _local_search_add(state, rng, time_limit):
    """Greedy add-only local search (fast, no annealing)."""
    best_A = list(state.A)
    best_B = list(state.B)
    best_score = _score(best_A, best_B)

    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try adding to A
        if len(state.A) < MAX_EACH:
            candidates = [x for x in range(1, N + 1) if x not in state.A]
            rng.shuffle(candidates)
            for x in candidates:
                if state._can_add_to_A(x):
                    state._add_to_A(x)
                    improved = True
                    break
        # Try adding to B
        if len(state.B) < MAX_EACH:
            candidates = [x for x in range(1, N + 1) if x not in state.B]
            rng.shuffle(candidates)
            for x in candidates:
                if state._can_add_to_B(x):
                    state._add_to_B(x)
                    improved = True
                    break

        cur_score = _score(state.A, state.B)
        if cur_score > best_score:
            best_score = cur_score
            best_A = list(state.A)
            best_B = list(state.B)

    return best_A, best_B


def construct():
    start = time.time()
    deadline = start + DEADLINE

    best_A = []
    best_B = []
    best_score = 0

    rng = random.Random(12345)

    # Multiple random restarts
    n_restarts = 15
    per_restart_time = (DEADLINE - 5) / n_restarts

    for r in range(n_restarts):
        if time.time() > deadline - 2:
            break

        restart_deadline = start + (r + 1) * per_restart_time
        if restart_deadline > deadline - 2:
            restart_deadline = deadline - 2

        # Greedy init
        state = _greedy_init(rng, 1.0)
        cur_score = _score(state.A, state.B)

        if cur_score > best_score:
            best_score = cur_score
            best_A = list(state.A)
            best_B = list(state.B)

        # Local search (add-only) first - fast
        ls_deadline = min(restart_deadline - 1.0, start + 3.0)
        if time.time() < ls_deadline:
            A2, B2 = _local_search_add(state, rng, ls_deadline)
            s2 = _score(A2, B2)
            if s2 > best_score:
                best_score = s2
                best_A = list(A2)
                best_B = list(B2)
                state = _State(A2, B2)

        # Simulated annealing - slower but can escape plateaus
        sa_deadline = restart_deadline
        if time.time() < sa_deadline:
            A3, B3 = _simulated_annealing(state, rng, sa_deadline)
            s3 = _score(A3, B3)
            if s3 > best_score:
                best_score = s3
                best_A = list(A3)
                best_B = list(B3)

    # Final verification
    if best_A and best_B and _valid_pair(best_A, best_B):
        return [best_A, best_B]

    # Fallback
    A_fb = [1, 2, 4, 8, 16, 32, 64, 128]
    B_fb = [3, 6, 12, 24, 48, 96, 192, 255]
    if _valid_pair(A_fb, B_fb):
        return [A_fb, B_fb]
    return [[1], [3]]


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    A, B = construct()
    t1 = time.time()
    print(f"A={A}")
    print(f"B={B}")
    print(f"Score={len(A)+len(B)}, time={t1-t0:.2f}s")
    print(f"Valid={_valid_pair(A, B)}")
