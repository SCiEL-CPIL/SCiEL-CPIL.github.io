# EVOLVE-BLOCK-START
import time
import random
import math

_N = 256
_DEADLINE = 80.0

# Precompute all APs as (start, step, length) and for each index,
# the list of AP indices it belongs to.
_all_aps = []  # list of (start, step, length)
_ap_index = []  # for each AP index, list of member indices
_idx_to_aps = [[] for _ in range(_N)]  # for each index, list of AP indices

def _build():
    global _all_aps, _ap_index
    for s in range(1, _N + 1):
        for a in range(_N):
            max_len = (_N - 1 - a) // s + 1
            for L in range(1, max_len + 1):
                ap_idx = len(_all_aps)
                members = [a + k * s for k in range(L)]
                _all_aps.append((a, s, L))
                _ap_index.append(members)
                for m in members:
                    _idx_to_aps[m].append(ap_idx)

_build()
_NUM_APS = len(_all_aps)

def _init_sums(signs):
    """Compute sum for each AP."""
    sums = [0] * _NUM_APS
    for ap_idx, (a, s, L) in enumerate(_all_aps):
        total = 0
        for k in range(L):
            total += signs[a + k * s]
        sums[ap_idx] = total
    return sums

def _max_abs(sums):
    best = 0
    for v in sums:
        av = v if v >= 0 else -v
        if av > best:
            best = av
    return best

def _flip_update(signs, i, sums):
    """Flip index i and update all affected AP sums. Returns new max_abs."""
    signs[i] = -signs[i]
    delta = -2 * signs[i]  # change in value at index i (signs[i] is new value)
    # Actually: old value was -signs[i] (since we just flipped), new is signs[i].
    # Change = signs[i] - (-signs[i]) = 2*signs[i]. But signs[i] is already new.
    # Let's recompute: before flip, signs[i] was old. We set signs[i] = -signs[i].
    # So new = -old. Change = new - old = -2*old = 2*new. Since new = signs[i],
    # change = 2 * signs[i].
    change = 2 * signs[i]
    for ap_idx in _idx_to_aps[i]:
        sums[ap_idx] += change
    best = 0
    for v in sums:
        av = v if v >= 0 else -v
        if av > best:
            best = av
    return best

def _greedy_polish(signs, sums, max_iters=3000, deadline=None):
    """Greedy: flip the index that most reduces max_abs."""
    current = _max_abs(sums)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.monotonic() > deadline:
            break
        best_gain = 0
        best_i = -1
        for i in range(_N):
            old_val = signs[i]
            signs[i] = -signs[i]
            change = 2 * signs[i]
            for ap_idx in _idx_to_aps[i]:
                sums[ap_idx] += change
            new_max = _max_abs(sums)
            # Revert
            for ap_idx in _idx_to_aps[i]:
                sums[ap_idx] -= change
            signs[i] = old_val
            gain = current - new_max
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i >= 0 and best_gain > 0:
            signs[best_i] = -signs[best_i]
            change = 2 * signs[best_i]
            for ap_idx in _idx_to_aps[best_i]:
                sums[ap_idx] += change
            current -= best_gain
            improved = True
    return current

def _anneal(signs, sums, start_temp=4.0, end_temp=0.01, steps=30000, deadline=None):
    """Simulated annealing with incremental evaluation."""
    current = _max_abs(sums)
    best = current
    best_signs = signs[:]
    best_sums = sums[:]
    temp = start_temp
    cooling = (start_temp - end_temp) / max(1, steps)
    rng = random.Random()
    for step in range(steps):
        if deadline is not None and time.monotonic() > deadline:
            break
        i = rng.randrange(_N)
        old_val = signs[i]
        signs[i] = -signs[i]
        change = 2 * signs[i]
        for ap_idx in _idx_to_aps[i]:
            sums[ap_idx] += change
        new_val = _max_abs(sums)
        delta = new_val - current
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            current = new_val
            if current < best:
                best = current
                best_signs = signs[:]
                best_sums = sums[:]
        else:
            # Revert
            for ap_idx in _idx_to_aps[i]:
                sums[ap_idx] -= change
            signs[i] = old_val
        temp -= cooling
        if temp < end_temp:
            temp = end_temp
    return best, best_signs, best_sums

def _thue_morse():
    return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(_N)]

def _alternating():
    return [1 if i % 2 == 0 else -1 for i in range(_N)]

def construct():
    t0 = time.monotonic()
    deadline = t0 + _DEADLINE

    rng = random.Random(42)
    candidates = []
    candidates.append(_thue_morse())
    candidates.append(_alternating())
    for _ in range(6):
        candidates.append([rng.choice((-1, 1)) for _ in range(_N)])
    for blk in [2, 4, 8, 16, 32]:
        sig = []
        for i in range(_N):
            pos = i % blk
            block = i // blk
            ss = 1 if block % 2 == 0 else -1
            sig.append(ss if pos % 2 == 0 else -ss)
        candidates.append(sig)

    scored = []
    for c in candidates:
        s = _init_sums(c)
        v = _max_abs(s)
        scored.append((v, c, s))
    scored.sort(key=lambda x: x[0])

    best_val, best_signs, best_sums = scored[0]

    # Polish the best
    dl1 = min(deadline, t0 + _DEADLINE * 0.25)
    cv = _greedy_polish(best_signs, best_sums, max_iters=2000, deadline=dl1)
    if cv < best_val:
        best_val = cv

    # Anneal if time remains
    if time.monotonic() < deadline - 3:
        sa_dl = deadline - 1
        val, sig, sm = _anneal(best_signs, best_sums, start_temp=3.0,
                               end_temp=0.01, steps=50000, deadline=sa_dl)
        if val < best_val:
            best_val = val
            best_signs = sig
            best_sums = sm

    # Final polish
    if time.monotonic() < deadline - 1:
        _greedy_polish(best_signs, best_sums, max_iters=1500, deadline=deadline - 0.5)
        best_val = _max_abs(best_sums)

    return list(best_signs)

# EVOLVE-BLOCK-END
