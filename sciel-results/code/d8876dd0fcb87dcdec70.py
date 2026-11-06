"""Search for a 12-state binary synchronizing automaton with a long reset threshold."""
import time
import random
from collections import deque

N = 12
DEADLINE = time.time() + 70.0  # safety margin vs 90s wall-clock


def _reset_length(trans_a, trans_b):
    """Exact shortest reset-word length via BFS on the subset automaton.
    Returns None if the automaton is not synchronizing."""
    full = (1 << N) - 1
    seen = {full}
    queue = deque([(full, 0)])
    while queue:
        s, d = queue.popleft()
        if s & (s - 1) == 0:  # single bit -> reset
            return d
        for t in (trans_a, trans_b):
            ns = 0
            for i in range(N):
                if s & (1 << i):
                    ns |= 1 << t[i]
            if ns not in seen:
                seen.add(ns)
                queue.append((ns, d + 1))
    return None


def _baseline():
    """Halving map (a) + cycle (b). Multi-step collapse -> longer reset word."""
    a = [i // 2 for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    return a, b


def construct():
    start = time.time()
    best_a, best_b = _baseline()
    best_len = _reset_length(best_a, best_b) or 0

    rng = random.Random(12345)

    def rand_table():
        return [rng.randrange(N) for _ in range(N)]

    def mutate(table):
        t = list(table)
        for _ in range(rng.randrange(1, 4)):
            t[rng.randrange(N)] = rng.randrange(N)
        return t

    cur_a, cur_b = _baseline()
    cur_len = best_len

    while time.time() < DEADLINE:
        # Occasionally jump to a fresh random table to escape local optima.
        if rng.random() < 0.05:
            cand_a, cand_b = rand_table(), rand_table()
        else:
            cand_a, cand_b = mutate(cur_a), mutate(cur_b)

        L = _reset_length(cand_a, cand_b)
        if L is None:
            continue
        if L > cur_len:
            cur_a, cur_b, cur_len = cand_a, cand_b, L
        if cur_len > best_len:
            best_a, best_b, best_len = cur_a, cur_b, cur_len

    return list(best_a), list(best_b)
