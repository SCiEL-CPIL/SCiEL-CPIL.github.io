"""Rewritten program for the Černý problem (N=12).

Strategy: structural inversion of the Černý automaton's reset algebra.
Instead of a constant second letter (immediate collapse), we use a
"controlled fold" letter that preserves the cycle except for a targeted
pair, forcing the reset word to shift the fold around the cycle many
times before full synchronization. A deadline-guarded local search then
refines the fold parameters to maximize the exact reset threshold.
"""
import time
import random
from collections import deque

N = 12

def _reset_threshold(trans_a, trans_b):
    """BFS on subset automaton to find exact shortest reset word length.
    Returns (length, word) or (None, None) if not synchronizing."""
    full = frozenset(range(N))
    # Map subsets to index for fast lookup
    visited = {full: 0}
    parent = {full: None}  # (parent_subset, letter)
    queue = deque([full])
    while queue:
        s = queue.popleft()
        for letter, trans in ((0, trans_a), (1, trans_b)):
            ns = frozenset(trans[x] for x in s)
            if ns not in visited:
                visited[ns] = visited[s] + 1
                parent[ns] = (s, letter)
                if len(ns) == 1:
                    # Reconstruct word length
                    return visited[ns], ns
                queue.append(ns)
    return None, None

def _construct_base(fold_source, fold_target):
    """Build the base automaton.
    Letter 0: cycle 0->1->2->...->11->0
    Letter 1: identity except fold_source -> fold_target (and to keep it
    a function, we let fold_target also map to itself or another state).
    Actually, to create a "fold", we map fold_source to fold_target,
    and leave everything else as identity. This creates a single
    non-invertible step. The reset word must shift fold_source into
    fold_target's position repeatedly.
    """
    trans_a = [(i + 1) % N for i in range(N)]
    trans_b = list(range(N))
    trans_b[fold_source] = fold_target
    return trans_a, trans_b

def _try_parameters(fold_source, fold_target):
    trans_a, trans_b = _construct_base(fold_source, fold_target)
    length, _ = _reset_threshold(trans_a, trans_b)
    return length if length is not None else 0

def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90s kill

    best_length = 0
    best_params = (0, 1)

    # Phase 1: evaluate all (fold_source, fold_target) pairs
    # There are 12*12 = 144 possibilities, each BFS is fast.
    for fs in range(N):
        for ft in range(N):
            if fs == ft:
                continue
            length = _try_parameters(fs, ft)
            if length > best_length:
                best_length = length
                best_params = (fs, ft)
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    trans_a, trans_b = _construct_base(*best_params)

    # Phase 2: local search — perturb the fold map to try to improve.
    # We allow trans_b to be any function, but start from the base and
    # flip individual state mappings.
    deadline2 = start + 80
    random.seed(42)
    current_a, current_b = trans_a, list(trans_b)
    current_len = best_length

    while time.time() < deadline2:
        # Randomly pick a state and change its image under letter 1
        i = random.randrange(N)
        old = current_b[i]
        new = random.randrange(N)
        current_b[i] = new
        length, _ = _reset_threshold(current_a, current_b)
        if length is not None and length > current_len:
            current_len = length
            trans_a, trans_b = list(current_a), list(current_b)
        else:
            current_b[i] = old  # revert

    return trans_a, trans_b
