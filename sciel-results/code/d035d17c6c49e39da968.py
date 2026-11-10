import time
import random

N = 180

# ---------- Precomputation (once, module level) ----------
# For each x, the set of partners that would complete a forbidden triple with x.
# A triple (a,b,c) distinct satisfies 1/a = 1/b + 1/c  <=>  (a-b)(a-c) = a^2.
# For a fixed x, partners y are those where y != x, 1<=y<=N, and
# (x-y) divides x^2 with the co-partner z = x + x^2/(x-y) in [1,N] and distinct.
_partners = [set() for _ in range(N + 1)]
for a in range(1, N + 1):
    a2 = a * a
    # iterate over all divisors d of a^2
    for d in range(1, int(a2 ** 0.5) + 1):
        if a2 % d == 0:
            for dd in (d, a2 // d):
                for sign in (1, -1):
                    diff = sign * dd          # a - b = diff  => b = a - diff
                    b = a - diff
                    z = a + a2 // dd          # a - c = a^2 / (a-b) = a^2/diff
                    # b and z are the two partners; both must be valid & distinct
                    if 1 <= b <= N and 1 <= z <= N and b != a and z != a and b != z:
                        _partners[a].add(b)
                        _partners[a].add(z)

# Bitset representation of partner sets for fast intersection
_partner_bits = [0] * (N + 1)
for x in range(1, N + 1):
    m = 0
    for p in _partners[x]:
        m |= 1 << p
    _partner_bits[x] = m

# ---------- Fast bitset helpers ----------
def _is_safe(S_bits, x):
    """True if x can be added to S (bitmask) without forming a triple."""
    return (_partner_bits[x] & S_bits) == 0

def _greedy_fill(S_bits, order):
    """Add safe elements in given order, return (bits, count_added)."""
    added = 0
    for x in order:
        if not (S_bits >> x) & 1 and _is_safe(S_bits, x):
            S_bits |= 1 << x
            added += 1
    return S_bits, added

def _count(S_bits):
    return bin(S_bits).count('1')

def _to_list(S_bits):
    return [x for x in range(1, N + 1) if (S_bits >> x) & 1]

def construct():
    start = time.time()
    deadline = start + 85.0   # safety margin under 90s
    rng = random.Random(12345)

    # ---------- Phase 0: good initialisation ----------
    # Start from the large safe core: the top half is reciprocal-free, then
    # greedily pull in lower elements in an order that keeps conflicts low.
    S_bits = 0
    # top-down order tends to keep the set dense and safe
    order0 = list(range(N, 0, -1))
    S_bits, _ = _greedy_fill(S_bits, order0)
    best_bits = S_bits
    best_size = _count(S_bits)

    # ---------- Local-search driver (Max-SAT style) ----------
    # Objective: maximise |S| subject to no forbidden triple.
    # Moves: (a) add a safe element, (b) 1-swap (remove u, add v),
    #        (c) 2-swap (remove {u,v}, add {w,z}) -- coordinated moves
    #        that escape single-add local optima.

    def _local_search(S_bits, max_rounds=400):
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 1.0:
            improved = False
            rounds += 1

            # (a) add any safe element
            for x in range(1, N + 1):
                if time.time() > deadline - 1.0:
                    break
                if not (S_bits >> x) & 1 and _is_safe(S_bits, x):
                    S_bits |= 1 << x
                    improved = True
                    break

            if not improved:
                # (b) 1-swap: remove u, then greedily add everything safe
                for u in range(1, N + 1):
                    if time.time() > deadline - 1.0:
                        break
                    if not ((S_bits >> u) & 1):
                        continue
                    S_temp = S_bits & ~(1 << u)
                    before = _count(S_temp)
                    S_temp, added = _greedy_fill(S_temp, range(1, N + 1))
                    if _count(S_temp) > before + 0:  # net gain >= 1
                        # net gain = added - 1 (we removed u)
                        if added - 1 > 0:
                            S_bits = S_temp
                            improved = True
                            break

            if not improved:
                # (c) 2-swap: remove two, greedily refill
                elements = [x for x in range(1, N + 1) if (S_bits >> x) & 1]
                rng.shuffle(elements)
                for i in range(len(elements)):
                    if time.time() > deadline - 1.0:
                        break
                    u = elements[i]
                    for j in range(i + 1, len(elements)):
                        if time.time() > deadline - 1.0:
                            break
                        v = elements[j]
                        S_temp = S_bits & ~(1 << u) & ~(1 << v)
                        before = _count(S_temp)
                        S_temp, added = _greedy_fill(S_temp, range(1, N + 1))
                        if added - 2 > 0:
                            S_bits = S_temp
                            improved = True
                            break
                    if improved:
                        break
        return S_bits

    best_bits = _local_search(best_bits)
    best_size = _count(best_bits)

    # ---------- Phase 2: random restarts + perturbations ----------
    # Occasionally perturb the best solution (drop a few elements) and re-optimise
    # to escape local optima. Also try fresh random-order fills.
    for it in range(2000):
        if time.time() > deadline - 2.0:
            break

        if it % 3 == 0:
            # fresh random-order greedy fill
            order = list(range(1, N + 1))
            rng.shuffle(order)
            cand, _ = _greedy_fill(0, order)
            cand = _local_search(cand, max_rounds=120)
        elif it % 3 == 1:
            # perturb best: drop k random elements, refill
            cand = best_bits
            elems = _to_list(cand)
            for _ in range(rng.randint(1, 4)):
                if elems:
                    cand &= ~(1 << rng.choice(elems))
                    elems = _to_list(cand)
            cand, _ = _greedy_fill(cand, range(1, N + 1))
            cand = _local_search(cand, max_rounds=150)
        else:
            # targeted: drop an element involved in many potential conflicts, refill
            cand = best_bits
            elems = _to_list(cand)
            if not elems:
                continue
            # pick the element whose removal frees the most partners
            best_u = None
            best_gain = -1
            for u in rng.sample(elems, min(len(elems), 25)):
                S_temp = cand & ~(1 << u)
                _, added = _greedy_fill(S_temp, range(1, N + 1))
                gain = added - 1
                if gain > best_gain:
                    best_gain = gain
                    best_u = u
            if best_u is not None:
                cand = cand & ~(1 << best_u)
                cand, _ = _greedy_fill(cand, range(1, N + 1))
                cand = _local_search(cand, max_rounds=150)

        if _count(cand) > best_size:
            best_size = _count(cand)
            best_bits = cand

    # ---------- Phase 3: final high-precision polish ----------
    # Tighter pass: repeatedly add safe + best 1-swap + best 2-swap until stable.
    for _ in range(200):
        if time.time() > deadline - 0.5:
            break
        changed = False
        # add safe
        for x in range(1, N + 1):
            if not (best_bits >> x) & 1 and _is_safe(best_bits, x):
                best_bits |= 1 << x
                changed = True
                break
        if not changed:
            # best 1-swap
            for u in range(1, N + 1):
                if time.time() > deadline - 0.5:
                    break
                if not ((best_bits >> u) & 1):
                    continue
                S_temp = best_bits & ~(1 << u)
                S_temp, added = _greedy_fill(S_temp, range(1, N + 1))
                if added - 1 > 0:
                    best_bits = S_temp
                    changed = True
                    break
        if not changed:
            # best 2-swap
            elements = _to_list(best_bits)
            for i in range(len(elements)):
                if time.time() > deadline - 0.5:
                    break
                u = elements[i]
                for j in range(i + 1, len(elements)):
                    if time.time() > deadline - 0.5:
                        break
                    v = elements[j]
                    S_temp = best_bits & ~(1 << u) & ~(1 << v)
                    S_temp, added = _greedy_fill(S_temp, range(1, N + 1))
                    if added - 2 > 0:
                        best_bits = S_temp
                        changed = True
                        break
                if changed:
                    break
        if not changed:
            break

    # ---------- Final verification (defensive) ----------
    result = _to_list(best_bits)
    Sset = set(result)
    for a in result:
        for b in _partners[a]:
            if b in Sset:
                # find the co-partner
                diff = a - b
                if diff != 0 and a * a % diff == 0:
                    z = a + a * a // diff
                    if z in Sset and z != a and z != b:
                        # violation: drop a (shouldn't happen)
                        Sset.discard(a)
    result = sorted(Sset)
    if not result:
        result = [180]
    return result
