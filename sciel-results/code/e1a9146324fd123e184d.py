"""Rewritten: shortest superpermutation via greedy nearest-unvisited walk over the
(n-1)-window graph with randomized restarts and a deadline guard."""
import time
import random
from itertools import permutations
from math import factorial

# Deadline guard: evaluator kills at 150s wall-clock; leave a safety margin.
_DEADLINE = 140.0


def _verify(word, n):
    """Return True if every permutation of 1..n appears as n consecutive symbols."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        w = word[i:i + n]
        if len(set(w)) == n:
            seen.add(w)
    return len(seen) == factorial(n)


def _build_nodes_edges(n):
    """Nodes: all permutations of (n-1) distinct symbols from 1..n (windows of length n-1).
    Edge from node a to node b exists iff b is a valid n-1 window reachable by a single
    transition: b == a[1:] + (x,) where x in 1..n and x not in a[1:].
    We represent nodes as tuples; adjacency is computed on the fly (cheap: n choices)."""
    nodes = list(permutations(range(1, n + 1), n - 1))
    index = {t: i for i, t in enumerate(nodes)}
    return nodes, index


def _neighbors(node, n, index):
    """Yield (neighbor_node_tuple, neighbor_index) for a window node.
    Transition: drop first symbol, append a new symbol x not in node[1:]."""
    tail = node[1:]
    tail_set = set(tail)
    for x in range(1, n + 1):
        if x not in tail_set:
            nb = tail + (x,)
            j = index.get(nb)
            if j is not None:
                yield nb, j


def _greedy_walk(nodes, index, n, rng, start):
    """Greedy nearest-unvisited walk. Returns (word_list, visited_count).
    Distance metric: prefer neighbors that are unvisited (weight 0) over visited (weight 1),
    with a small stochastic tie-break among candidates to diversify restarts.
    The walk continues until all nodes are visited (or no unvisited neighbor reachable and
    we allow revisits to bridge back to unvisited nodes)."""
    num_nodes = len(nodes)
    visited = [False] * num_nodes
    cur = start
    visited[cur] = True
    visited_count = 1
    # The word starts as the first window.
    word = list(nodes[cur])
    # We need to track the full sequence of nodes to reconstruct, but word is enough:
    # each step appends one symbol (the new symbol added to the window).
    while visited_count < num_nodes:
        # Collect candidate neighbors.
        cands = list(_neighbors(nodes[cur], n, index))
        if not cands:
            break
        # Score: 0 if unvisited, 1 if visited; add tiny noise for tie-break.
        best = None
        best_key = None
        for nb, j in cands:
            key = (0 if not visited[j] else 1, rng.random())
            if best_key is None or key < best_key:
                best_key = key
                best = (nb, j)
        if best is None:
            break
        nb, j = best
        # Append the new symbol (last element of nb).
        word.append(nb[-1])
        if not visited[j]:
            visited[j] = True
            visited_count += 1
        cur = j
    return word, visited_count


def construct(n):
    start_time = time.time()
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # Build the graph.
    nodes, index = _build_nodes_edges(n)
    num_nodes = len(nodes)

    # Baseline: the classic recursive construction length = sum_{k=1}^{n} k!
    baseline_len = sum(factorial(k) for k in range(1, n + 1))

    rng = random.Random(12345)
    best_word = None
    best_len = baseline_len + 1  # we'll always have a valid fallback

    # Generate a few random starting nodes to diversify restarts.
    # Budget: run restarts until deadline. Each restart is O(num_nodes * n) roughly.
    # For n=7, num_nodes = 7!/2 = 2520, cheap.
    max_restarts = 20000
    restart = 0
    while restart < max_restarts:
        if time.time() - start_time > _DEADLINE - 5.0:
            break
        start = rng.randrange(num_nodes)
        word, vc = _greedy_walk(nodes, index, n, rng, start)
        if vc == num_nodes:
            # Verify it's a valid superpermutation.
            if _verify(word, n):
                if len(word) < best_len:
                    best_len = len(word)
                    best_word = word
                    # Early exit if we hit a strong bound.
                    if best_len <= 5905:
                        # Keep searching a bit more but don't over-invest.
                        pass
        restart += 1

    # Fallback: classic recursive construction (guaranteed valid).
    if best_word is None:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                               if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        best_word = word
        best_len = len(word)

    return "".join(map(str, best_word))
