import time
import random

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    n = 128
    range_mask = (1 << 129) - 2  # bits 1..128

    # Safe base: all numbers in (64,128]; any two sum >128 so never in S.
    H = 0
    for i in range(65, 129):
        H |= 1 << i

    def repair(S):
        """Greedy add-to-completion: repeatedly add the addable element with
        the smallest sum-degree until none remain. Returns maximal (valid) S."""
        S_list = [i for i in range(1, 129) if (S >> i) & 1]
        while True:
            S_list.sort()
            m = len(S_list)
            blocked1 = 0
            blocked2 = 0
            for i in range(m):
                u = S_list[i]
                lim = 128 - u
                for j in range(i + 1, m):
                    v = S_list[j]
                    if v > lim:
                        break
                    s = u + v
                    if (S >> s) & 1:
                        # pair active (u+v in S): blocks x with x+u in S and x+v in S
                        blocked1 |= (S >> u) & (S >> v)
                    # pair with u+v == x=s: blocks candidate x=s if s+u in S and s+v in S
                    if (S >> (s + u)) & 1 and (S >> (s + v)) & 1:
                        blocked2 |= (1 << s)
            addable = range_mask & ~S & ~blocked1 & ~blocked2
            if addable == 0:
                break
            # choose addable element with minimal sum-degree
            best_x = -1
            best_deg = 10 ** 9
            mm = addable
            while mm:
                lsb = mm & (-mm)
                x = lsb.bit_length() - 1
                deg = (S & (S >> x)).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_x = x
                mm ^= lsb
            S |= 1 << best_x
            S_list.append(best_x)
        return S

    # Initial solution: repair the safe base.
    current = repair(H)
    best = current
    best_size = current.bit_count()

    # Iterated Local Search: random kick + repair, accept all, track best.
    while time.time() < deadline:
        k = random.choice([1, 1, 2, 2, 3, 4])
        trial = current
        bits = [i for i in range(1, 129) if (trial >> i) & 1]
        for _ in range(k):
            if bits:
                r = random.randrange(len(bits))
                idx = bits[r]
                bits[r] = bits[-1]
                bits.pop()
                trial &= ~(1 << idx)
        trial = repair(trial)
        current = trial
        sz = trial.bit_count()
        if sz > best_size:
            best = trial
            best_size = sz
        # occasionally re-seed from best to avoid long drift
        if random.random() < 0.03:
            current = best

    # Polish: systematic 1-removal scan on best (removing a "bad" element can
    # unblock many additions).
    improved = True
    while improved and time.time() < deadline:
        improved = False
        b = best
        bits = [i for i in range(1, 129) if (b >> i) & 1]
        for idx in bits:
            if time.time() > deadline:
                break
            trial = b & ~(1 << idx)
            trial = repair(trial)
            if trial.bit_count() > b.bit_count():
                b = trial
                best = b
                best_size = b.bit_count()
                improved = True
                bits = [i for i in range(1, 129) if (b >> i) & 1]

    # Final add pass.
    best = repair(best)

    S_set = set(i for i in range(1, 129) if (best >> i) & 1)
    if not _is_valid_set(S_set):
        S_set = set(range(65, 129))
    return sorted(S_set)


def _is_valid_set(S):
    """Ground-truth check: no three distinct a<b<c with all pair sums in S."""
    L = sorted(S)
    for i in range(len(L)):
        a = L[i]
        for j in range(i + 1, len(L)):
            b = L[j]
            if (a + b) not in S:
                continue
            for k in range(j + 1, len(L)):
                c = L[k]
                if (a + c) in S and (b + c) in S:
                    return False
    return True
