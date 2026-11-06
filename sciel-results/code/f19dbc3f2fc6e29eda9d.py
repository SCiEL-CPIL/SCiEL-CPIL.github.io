"""Černý problem, N=12: maximize the exact shortest reset-word length.

Strategy:
1. Start with the Vershik–Kondakov "pinwheel" automaton (known threshold 2N-3 = 21).
2. Use a time-bounded local search to try to find an automaton with a longer
   shortest reset word. We perform random mutations on the transition tables and
   verify each candidate with an optimized BFS on the subset automaton.
3. Keep the best valid synchronizing automaton found within the time budget.
4. Return the best automaton found (or the pinwheel if no improvement is found).

The BFS is optimized using precomputed transition tables and bitmask operations.
"""
import time
import random
from collections import deque

N = 12
_DEADLINE = 80.0  # seconds; evaluator kills at 90s, leave a safety margin


def _time_left():
    return _DEADLINE - time.time()


def _pinwheel():
    """Vershik–Kondakov pinwheel automaton on N states.

    letter 'a': successor on the cycle
    letter 'b': predecessor on the cycle, except state 0 maps to 1 (the pin)
    """
    a = [(i + 1) % N for i in range(N)]
    b = [(i - 1) % N for i in range(N)]
    b[0] = 1  # the pin
    return a, b


def _seed():
    """Fallback: simple synchronizing automaton (cycle + map to 0)."""
    return [(i + 1) % N for i in range(N)], [0] * N


def _bfs_reset_length(a, b):
    """Exact shortest reset-word length via BFS on the subset automaton.

    Returns (length, reset_word) where reset_word is a list of 0/1 letters.
    Returns (None, None) if the automaton does not synchronize.
    """
    full = (1 << N) - 1
    # Precompute image of every subset under each letter.
    img_a = [0] * (1 << N)
    img_b = [0] * (1 << N)
    for s in range(1 << N):
        sa = 0
        sb = 0
        for i in range(N):
            if s >> i & 1:
                sa |= 1 << a[i]
                sb |= 1 << b[i]
        img_a[s] = sa
        img_b[s] = sb

    dist = [-1] * (1 << N)
    prev = [(-1, -1)] * (1 << N)
    dist[full] = 0
    q = deque([full])
    while q:
        s = q.popleft()
        if s & (s - 1) == 0 and s != 0:  # a non-empty singleton
            # reconstruct word
            word = []
            cur = s
            while cur != full:
                p, letter = prev[cur]
                word.append(letter)
                cur = p
            word.reverse()
            return dist[s], word
        for letter, img in ((0, img_a), (1, img_b)):
            ns = img[s]
            if dist[ns] == -1:
                dist[ns] = dist[s] + 1
                prev[ns] = (s, letter)
                q.append(ns)
    return None, None


def _local_search(a0, b0, time_limit=60.0):
    """Local search to find an automaton with a longer reset word.
    
    Mutates the transition tables randomly and checks synchronizing property.
    Returns the best (a, b) found with its reset length, or (a0, b0, 0) if no improvement.
    """
    best_a, best_b = list(a0), list(b0)
    best_len, _ = _bfs_reset_length(best_a, best_b)
    if best_len is None:
        best_len = 0
    
    start_time = time.time()
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        if _time_left() < 5.0:
            break
        
        # Mutate: pick a random state and letter, change its destination
        a = list(best_a)
        b = list(best_b)
        
        # Try a few mutations in one step
        for _ in range(2):
            letter = rng.randint(0, 1)
            state = rng.randint(0, N - 1)
            new_dest = rng.randint(0, N - 1)
            if letter == 0:
                a[state] = new_dest
            else:
                b[state] = new_dest
        
        # Check if synchronizing and get reset length
        length, _ = _bfs_reset_length(a, b)
        if length is not None and length > best_len:
            best_a, best_b = a, b
            best_len = length
    
    return best_a, best_b, best_len


# EVOLVE-BLOCK-START
def construct():
    # Start with the pinwheel automaton
    a, b = _pinwheel()
    
    # Verify the pinwheel's exact threshold
    length, _ = _bfs_reset_length(a, b)
    if length is None:
        # Fallback to seed if pinwheel doesn't synchronize (shouldn't happen)
        return _seed()
    
    best_len = length
    
    # Try local search to improve
    if _time_left() > 10.0:
        a, b, best_len = _local_search(a, b, time_limit=_time_left() - 5.0)
    
    return a, b
# EVOLVE-BLOCK-END
