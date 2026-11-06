import time
import random
import numpy as np

N = 16
NUM_INPUTS = 1 << N

# Precompute all 65536 boolean inputs: shape (65536, 16), int8
def _precompute():
    idx = np.arange(NUM_INPUTS, dtype=np.uint16)
    bits = 1 << np.arange(N, dtype=np.uint16)
    x = (idx[:, None] & bits).astype(np.int8)  # (65536, 16)
    # Sorted version: for each input with k ones, output is [0]*k + [1]*(16-k)
    pc = np.zeros(NUM_INPUTS, dtype=np.int8)
    for b in range(N):
        pc += (idx >> b) & 1
    sorted_x = np.zeros((NUM_INPUTS, N), dtype=np.int8)
    for c in range(N):
        sorted_x[:, c] = (c >= pc).astype(np.int8)
    return x, sorted_x

_INPUTS, _SORTED = _precompute()

def _apply(net, x):
    x = x.copy()
    for i, j in net:
        a = x[:, i]
        b = x[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        x[:, i] = lo
        x[:, j] = hi
    return x

def _is_valid(net):
    if not net:
        return False
    out = _apply(net, _INPUTS)
    return bool(np.array_equal(out, _SORTED))

def _batcher(n):
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _batcher(half)
    right = _batcher(half)
    # Even-odd merge
    eo = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            eo.append((i, i + 1))
    return left + right + eo

def construct():
    start = time.time()
    deadline = start + 70.0
    rng = random.Random(42)

    best_net = None
    best_len = 10**9

    # Seed with known-good networks
    seeds = [
        [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)],  # 120
        _batcher(N),  # Batcher
    ]
    for s in seeds:
        if _is_valid(s) and len(s) < best_len:
            best_len = len(s)
            best_net = list(s)

    # Local search: hill-climbing with random restarts
    # Mutation: change one comparator, add one, or remove one
    def mutate(net):
        net = list(net)
        op = rng.random()
        if op < 0.33:
            # change a comparator
            if net:
                pos = rng.randrange(len(net))
                i = rng.randrange(N)
                j = rng.randrange(N)
                if i == j:
                    j = (j + 1) % N
                if i > j:
                    i, j = j, i
                net[pos] = (i, j)
        elif op < 0.66:
            # insert a comparator
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                j = (j + 1) % N
            if i > j:
                i, j = j, i
            pos = rng.randrange(len(net) + 1)
            net.insert(pos, (i, j))
        else:
            # remove a comparator
            if net:
                pos = rng.randrange(len(net))
                net.pop(pos)
        return net

    # Hill-climb from best
    current = list(best_net) if best_net else seeds[0][:]
    cur_len = len(current)
    no_improve = 0
    max_no_improve = 30

    while time.time() < deadline:
        if no_improve >= max_no_improve:
            # Restart from a seed or random
            if rng.random() < 0.5:
                current = list(rng.choice(seeds))
            else:
                ln = rng.randint(40, 70)
                current = []
                for _ in range(ln):
                    i = rng.randrange(N)
                    j = rng.randrange(N)
                    if i == j:
                        j = (j + 1) % N
                    if i > j:
                        i, j = j, i
                    current.append((i, j))
            no_improve = 0
            cur_len = len(current)
            if not _is_valid(current):
                # If restart is invalid, skip to next iteration
                continue

        # Try a mutation
        cand = mutate(current)
        if not cand:
            continue
        if _is_valid(cand):
            if len(cand) < cur_len:
                current = cand
                cur_len = len(cand)
                no_improve = 0
                if cur_len < best_len:
                    best_len = cur_len
                    best_net = list(current)
            else:
                no_improve += 1
        else:
            no_improve += 1

    if best_net is None:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    return best_net
