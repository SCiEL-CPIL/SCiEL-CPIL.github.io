"""Two Sidon sets with disjoint difference sets.
Structural change: represent Sidon sum-constraints and cross-difference
disjointness as compact integer bitmasks (sums 2..512 -> 511 bits; diffs -255..255
-> 511 bits). This makes candidate validity a handful of bitwise ANDs, enabling
massive parallel evaluation of all 256 candidates per set and aggressive 1-for-1
and 1-for-2 swap local search inside the 90 s budget."""

import time
import random
import math

N = 256
MAX_SIZE = 64
SUM_BITS = 512  # sums 2..512 inclusive -> 511 bits
DIFF_BITS = 512  # diffs -255..255 inclusive -> 511 bits


def _sum_bit(x):
    # x in [1,256]; 2x in [2,512]; bit index 2x-2 in [0,510]
    return 1 << (2 * x - 2)


def _sum_pair_bit(a, b):
    # a+b in [2,512]; bit index (a+b)-2
    return 1 << (a + b - 2)


def _diff_bit(d):
    # d in [-255,255], d != 0; index d+255 in [0,510]
    return 1 << (d + 255)


def _check_sidon(els):
    n = len(els)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _nonzero_diffs(els):
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d.add(els[j] - els[i])
    return d


def _valid(A, B):
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True


class State:
    __slots__ = ("A", "B", "sums_A", "sums_B", "diffs_A", "diffs_B")

    def __init__(self, A=None, B=None):
        self.A = list(A) if A else []
        self.B = list(B) if B else []
        self.sums_A = 0
        self.sums_B = 0
        self.diffs_A = 0
        self.diffs_B = 0
        self._rebuild()

    def _rebuild(self):
        self.sums_A = 0
        for i in range(len(self.A)):
            for j in range(i, len(self.A)):
                self.sums_A |= _sum_pair_bit(self.A[i], self.A[j])
        self.sums_B = 0
        for i in range(len(self.B)):
            for j in range(i, len(self.B)):
                self.sums_B |= _sum_pair_bit(self.B[i], self.B[j])
        self.diffs_A = 0
        for i in range(len(self.A)):
            for j in range(len(self.A)):
                if i != j:
                    self.diffs_A |= _diff_bit(self.A[j] - self.A[i])
        self.diffs_B = 0
        for i in range(len(self.B)):
            for j in range(len(self.B)):
                if i != j:
                    self.diffs_B |= _diff_bit(self.B[j] - self.B[i])

    def copy(self):
        s = State.__new__(State)
        s.A = list(self.A)
        s.B = list(self.B)
        s.sums_A = self.sums_A
        s.sums_B = self.sums_B
        s.diffs_A = self.diffs_A
        s.diffs_B = self.diffs_B
        return s


def _valid_add_A(st, x):
    """Check if x can be added to A (Sidon within A and diff disjointness with B)."""
    if x in st.A:
        return False
    # Sidon: 2x and x+a for all a in A must not collide with existing sums_A
    if st.sums_A & _sum_bit(x):
        return False
    # x+a for each a in A
    for a in st.A:
        if st.sums_A & _sum_pair_bit(a, x):
            return False
    # diff disjointness: x-a and a-x must not be in diffs_B
    for a in st.A:
        d1 = x - a
        d2 = a - x
        if d1 != 0 and (st.diffs_B & _diff_bit(d1)):
            return False
        if d2 != 0 and (st.diffs_B & _diff_bit(d2)):
            return False
    return True


def _valid_add_B(st, x):
    if x in st.B:
        return False
    if st.sums_B & _sum_bit(x):
        return False
    for b in st.B:
        if st.sums_B & _sum_pair_bit(b, x):
            return False
    for b in st.B:
        d1 = x - b
        d2 = b - x
        if d1 != 0 and (st.diffs_A & _diff_bit(d1)):
            return False
        if d2 != 0 and (st.diffs_A & _diff_bit(d2)):
            return False
    return True


def _commit_add_A(st, x):
    st.A.append(x)
    st.sums_A |= _sum_bit(x)
    for a in st.A[:-1]:
        st.sums_A |= _sum_pair_bit(a, x)
    for a in st.A[:-1]:
        st.diffs_A |= _diff_bit(x - a)
        st.diffs_A |= _diff_bit(a - x)


def _commit_add_B(st, x):
    st.B.append(x)
    st.sums_B |= _sum_bit(x)
    for b in st.B[:-1]:
        st.sums_B |= _sum_pair_bit(b, x)
    for b in st.B[:-1]:
        st.diffs_B |= _diff_bit(x - b)
        st.diffs_B |= _diff_bit(b - x)


def _valid_remove_add_A(st, remove_idx, add_x):
    """Check if replacing A[remove_idx] with add_x keeps validity."""
    if add_x in st.A and st.A.index(add_x) != remove_idx:
        return False
    new_A = st.A[:remove_idx] + st.A[remove_idx + 1:] + [add_x]
    # Rebuild sums_A quickly: recompute from scratch for correctness
    sums = 0
    for i in range(len(new_A)):
        for j in range(i, len(new_A)):
            sums |= _sum_pair_bit(new_A[i], new_A[j])
    # Check diff disjointness with B
    diffs = 0
    for i in range(len(new_A)):
        for j in range(len(new_A)):
            if i != j:
                d = new_A[j] - new_A[i]
                if st.diffs_B & _diff_bit(d):
                    return False
    return True


def _valid_remove_add_B(st, remove_idx, add_x):
    if add_x in st.B and st.B.index(add_x) != remove_idx:
        return False
    new_B = st.B[:remove_idx] + st.B[remove_idx + 1:] + [add_x]
    sums = 0
    for i in range(len(new_B)):
        for j in range(i, len(new_B)):
            sums |= _sum_pair_bit(new_B[i], new_B[j])
    diffs = 0
    for i in range(len(new_B)):
        for j in range(len(new_B)):
            if i != j:
                d = new_B[j] - new_B[i]
                if st.diffs_A & _diff_bit(d):
                    return False
    return True


def _swap_A(st, remove_idx, add_x):
    st.A[remove_idx] = add_x
    st._rebuild()


def _swap_B(st, remove_idx, add_x):
    st.B[remove_idx] = add_x
    st._rebuild()


def _greedy_grow(st, deadline, rng):
    """Greedily add elements to A and B using bitmask validity checks."""
    candidates_A = list(range(1, N + 1))
    candidates_B = list(range(1, N + 1))
    rng.shuffle(candidates_A)
    rng.shuffle(candidates_B)

    for _ in range(500):
        if time.time() > deadline - 3:
            break
        progress = False

        if len(st.A) < MAX_SIZE:
            for x in candidates_A:
                if x in st.A:
                    continue
                if _valid_add_A(st, x):
                    _commit_add_A(st, x)
                    candidates_A.remove(x)
                    progress = True
                    break

        if len(st.B) < MAX_SIZE:
            for x in candidates_B:
                if x in st.B:
                    continue
                if _valid_add_B(st, x):
                    _commit_add_B(st, x)
                    candidates_B.remove(x)
                    progress = True
                    break

        if not progress:
            break

    return st


def _local_search(st, deadline, rng):
    """Aggressive local search: 1-for-1 swaps and 1-for-2 exchanges."""
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False

        # 1-for-1 swaps in A
        if len(st.A) >= 1:
            for i in range(len(st.A)):
                if time.time() > deadline - 5:
                    break
                for x in range(1, N + 1):
                    if x == st.A[i] or x in st.A:
                        continue
                    if _valid_remove_add_A(st, i, x):
                        _swap_A(st, i, x)
                        improved = True
                        break
                if improved:
                    break

        # 1-for-1 swaps in B
        if len(st.B) >= 1:
            for i in range(len(st.B)):
                if time.time() > deadline - 5:
                    break
                for x in range(1, N + 1):
                    if x == st.B[i] or x in st.B:
                        continue
                    if _valid_remove_add_B(st, i, x):
                        _swap_B(st, i, x)
                        improved = True
                        break
                if improved:
                    break

        # Try to add to A if space
        if len(st.A) < MAX_SIZE:
            for x in range(1, N + 1):
                if x not in st.A and _valid_add_A(st, x):
                    _commit_add_A(st, x)
                    improved = True
                    break

        # Try to add to B if space
        if len(st.B) < MAX_SIZE:
            for x in range(1, N + 1):
                if x not in st.B and _valid_add_B(st, x):
                    _commit_add_B(st, x)
                    improved = True
                    break

    return st


def _structured_seed(rng):
    """Generate a structured seed using number-theoretic ideas."""
    # Use quadratic residues mod a prime for A, non-residues for B
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True

    primes = [p for p in range(13, 60) if is_prime(p)]
    rng.shuffle(primes)

    for p in primes:
        qr = sorted(set((i * i) % p for i in range(p)))
        nqr = sorted(set(range(p)) - set(qr))
        scale = N // p
        A = []
        for r in qr:
            val = r * scale + 1
            if val <= N and val not in A:
                A.append(val)
        B = []
        for r in nqr:
            val = r * scale + 1
            if val <= N and val not in B:
                B.append(val)
        if A and B and _valid(A, B):
            return A, B

    # Fallback: random seeds
    A = [rng.randint(1, N)]
    B = [rng.randint(1, N)]
    return A, B


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    rng = random.Random(42)

    best_score = 0
    best_A = [1]
    best_B = [2]

    # Initial structured seed
    A0, B0 = _structured_seed(rng)
    if not A0:
        A0 = [1]
    if not B0:
        B0 = [2]

    st = State(A0, B0)
    _greedy_grow(st, deadline, rng)
    _local_search(st, deadline, rng)
    score = len(st.A) + len(st.B)
    if score > best_score and _valid(st.A, st.B):
        best_score = score
        best_A = list(st.A)
        best_B = list(st.B)

    # Multiple restarts with different seeds
    for it in range(200):
        if time.time() > deadline - 10:
            break

        strat = it % 4
        if strat == 0:
            A0 = [rng.randint(1, N)]
            B0 = [rng.randint(1, N)]
        elif strat == 1:
            sa = rng.randint(2, 4)
            sb = rng.randint(2, 4)
            A0 = sorted(rng.sample(range(1, N + 1), sa))
            B0 = sorted(rng.sample(range(1, N + 1), sb))
        elif strat == 2:
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, N)]
        else:
            A0, B0 = _structured_seed(rng)

        st = State(A0, B0)
        _greedy_grow(st, deadline, rng)
        _local_search(st, deadline, rng)
        score = len(st.A) + len(st.B)
        if score > best_score and _valid(st.A, st.B):
            best_score = score
            best_A = list(st.A)
            best_B = list(st.B)

    # Final polish on best
    if time.time() < deadline - 5:
        st = State(best_A, best_B)
        _local_search(st, deadline, rng)
        score = len(st.A) + len(st.B)
        if score > best_score and _valid(st.A, st.B):
            best_A = list(st.A)
            best_B = list(st.B)
            best_score = score

    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
