"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 17
LIMIT = 1 << N  # 131072

# EVOLVE-BLOCK-START
import time
import random
import math

N = 17
LIMIT = 1 << N  # 131072


def _get_initial_bitmasks(n):
    """S[i]: 2^n-bit int, bit k set iff input k has a 1 on wire i."""
    masks = [0] * n
    for i in range(n):
        m = 0
        for k in range(LIMIT):
            if (k >> i) & 1:
                m |= (1 << k)
        masks[i] = m
    return masks


_INITIAL_MASKS = _get_initial_bitmasks(N)


def _simulate(net, masks=None):
    """Apply compare-exchange network to bitmask state; return new state."""
    if masks is None:
        masks = _INITIAL_MASKS[:]
    else:
        masks = masks[:]
    for i, j in net:
        a = masks[i]
        b = masks[j]
        masks[i] = a & b
        masks[j] = a | b
    return masks


def _is_sorted(state):
    for k in range(N - 1):
        if state[k] & ~state[k + 1]:
            return False
    return True


def _is_valid(net, state=None):
    return _is_sorted(_simulate(net, state))


def _odd_even_transposition():
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def construct():
    start = time.time()
    deadline = start + 83.0  # 90s wall-clock limit; keep 7s safety margin

    # ---- Starting network (valid): odd-even transposition, 136 comparators ----
    net = _odd_even_transposition()
    state = _simulate(net)
    if not _is_sorted(state):
        # Fallback: return the (known-valid) transposition network.
        return net

    best_net = net[:]
    best_len = len(net)

    # ---- Phase 1: greedy deletion (fast, cheap) to reach a local minimum ----
    while time.time() < deadline:
        removed = False
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            if idx >= len(net):
                continue
            removed_pair = net.pop(idx)
            st = _simulate(net, state)
            if _is_sorted(st):
                state = st
                removed = True
            else:
                net.insert(idx, removed_pair)
            if time.time() > deadline:
                break
        if not removed:
            break

    best_net = net[:]
    best_len = len(net)

    # ---- Phase 2: simulated annealing to escape local minima ----
    # Moves: delete (improving), insert redundant pair (worsening, enables escape),
    # swap order, and move a comparator. Accept worsening moves with probability
    # exp(-(delta)/T) where delta = increase in comparator count.
    T = 3.0
    T_min = 0.02
    # Geometric cooling; pick rate so we roughly hit T_min before the deadline.
    # We do not fix a step count (budget-driven), we just cool each step.
    cool = 0.9990

    # Precompute all candidate redundant pairs (i<j), for insertion moves.
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    while time.time() < deadline and T > T_min:
        # Choose a move
        r = random.random()
        if r < 0.45 and len(net) > 0:
            # delete a random comparator
            idx = random.randrange(len(net))
            removed_pair = net.pop(idx)
            st = _simulate(net, state)
            delta = -1  # one fewer comparator
            if _is_sorted(st):
                # always accept improving
                state = st
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = net[:]
            else:
                # accept worsening? (deletion can't be worse in count, but can break)
                # here it broke sorting -> reject (delta in count is -1 but invalid)
                net.insert(idx, removed_pair)
                # keep state unchanged
        elif r < 0.70:
            # insert a redundant (duplicate) comparator
            pair = random.choice(all_pairs)
            idx = random.randrange(len(net) + 1)
            net.insert(idx, pair)
            st = _simulate(net, state)
            delta = +1  # one more comparator
            if _is_sorted(st):
                # valid; this is a worsening in count -> accept with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    state = st
                else:
                    net.pop(idx)
            else:
                net.pop(idx)
        elif r < 0.88 and len(net) > 1:
            # swap two adjacent comparators
            idx = random.randrange(len(net) - 1)
            net[idx], net[idx + 1] = net[idx + 1], net[idx]
            st = _simulate(net, state)
            delta = 0
            if _is_sorted(st):
                state = st
            else:
                net[idx], net[idx + 1] = net[idx + 1], net[idx]
        else:
            # move a comparator to a random new position
            if len(net) > 1:
                idx = random.randrange(len(net))
                pair = net.pop(idx)
                new_idx = random.randrange(len(net) + 1)
                net.insert(new_idx, pair)
                st = _simulate(net, state)
                delta = 0
                if _is_sorted(st):
                    state = st
                else:
                    net.remove(pair)
                    net.insert(idx, pair)

        T *= cool
        if time.time() > deadline:
            break

    # ---- Phase 3: final greedy deletion polish on the best found network ----
    if time.time() < deadline:
        polish = best_net[:]
        pstate = _simulate(polish)
        if _is_sorted(pstate):
            improved = True
            while improved and time.time() < deadline:
                improved = False
                indices = list(range(len(polish)))
                random.shuffle(indices)
                for idx in indices:
                    if idx >= len(polish):
                        continue
                    removed_pair = polish.pop(idx)
                    st = _simulate(polish, pstate)
                    if _is_sorted(st):
                        pstate = st
                        improved = True
                    else:
                        polish.insert(idx, removed_pair)
                    if time.time() > deadline:
                        break
            if _is_sorted(_simulate(polish)) and len(polish) < best_len:
                best_net = polish[:]
                best_len = len(polish)

    # ---- Final verification ----
    if _is_valid(best_net):
        return best_net
    # Fallback to the original valid transposition network.
    return _odd_even_transposition()

# EVOLVE-BLOCK-END
