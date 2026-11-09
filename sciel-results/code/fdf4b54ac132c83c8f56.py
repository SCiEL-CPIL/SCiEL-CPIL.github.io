"""Combined program for large pairwise non-coprime sets containing 255."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations
import numpy as np

_DEADLINE = time.time() + 85  # 90s wall-clock budget with 5s safety margin

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _validate(S):
    """Check that S is a valid non-coprime pairwise set containing 255."""
    S = set(S)
    if 255 not in S:
        return False
    if len(S) < 1:
        return False
    for a, b in combinations(sorted(S), 2):
        if math.gcd(a, b) == 1:
            return False
    return True

def _build_union_set(primes):
    """Build the set of numbers in [1,255] divisible by at least one prime in `primes`."""
    S = set()
    for p in primes:
        for k in range(p, 256, p):
            S.add(k)
    return S

def _try_add_element(S, candidate, n=255):
    """Check if adding candidate to S keeps all pairwise gcd > 1."""
    for a in S:
        if math.gcd(a, candidate) == 1:
            return False
    return True

def _greedy_complete(S, n=255):
    """Greedily add numbers from [1,n] that preserve the non-coprime property."""
    S = set(S)
    added = True
    while added:
        added = False
        for cand in range(1, n + 1):
            if cand not in S and _try_add_element(S, cand, n):
                S.add(cand)
                added = True
    return S

def _local_search(S, n=255, max_rounds=50):
    """Try to improve S by swapping out one element for another."""
    S = set(S)
    for _ in range(max_rounds):
        if time.time() > _DEADLINE:
            break
        improved = False
        S_list = sorted(S)
        for i, remove in enumerate(S_list):
            S_minus = S - {remove}
            for cand in range(1, n + 1):
                if cand == remove or cand in S_minus:
                    continue
                if _try_add_element(S_minus, cand, n):
                    S = S_minus | {cand}
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return S

def construct():
    start = time.time()
    N = 255
    best = None
    best_size = 0

    # Precompute candidates: numbers in [1,255] that share a factor with 255
    # Primes of 255: 3, 5, 17
    candidates = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    n_cand = len(candidates)
    cand_to_idx = {val: idx for idx, val in enumerate(candidates)}
    idx_255 = cand_to_idx[255]

    # Build adjacency list for the graph induced by candidates
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)

    def greedy_clique(start_node=None, random_order=False, order=None):
        """
        Greedy clique construction.
        If order is provided, use that order of nodes.
        """
        if order is None:
            if random_order:
                nodes = list(range(n_cand))
                random.shuffle(nodes)
            else:
                # Sort by degree descending
                nodes = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
        else:
            nodes = order

        clique = []
        if start_node is not None:
            # Ensure start_node is first
            if start_node in nodes:
                nodes.remove(start_node)
            nodes.insert(0, start_node)
            clique.append(start_node)
            candidates_for_next = set(adj[start_node])
        else:
            candidates_for_next = set(nodes)

        for node in nodes:
            if node in candidates_for_next:
                clique.append(node)
                # Update candidates_for_next to intersection with adj[node]
                candidates_for_next &= adj[node]
                
        return clique

    # Strategy 1: Greedy clique with various orderings
    # 1a: Degree-sorted
    clique_indices = greedy_clique(start_node=idx_255, random_order=False)
    clique_vals = [candidates[i] for i in clique_indices]
    if 255 not in clique_vals:
        clique_vals.append(255)
    if len(clique_vals) > best_size:
        best = set(clique_vals)
        best_size = len(best)

    # 1b: Random restarts
    random.seed(42)
    for _ in range(1000):
        if time.time() > _DEADLINE - 10:
            break
        clique_indices = greedy_clique(start_node=idx_255, random_order=True)
        clique_vals = [candidates[i] for i in clique_indices]
        if 255 not in clique_vals:
            clique_vals.append(255)
        if len(clique_vals) > best_size:
            best = set(clique_vals)
            best_size = len(best)

    # Strategy 2: Union of multiples of common prime subsets
    # This can sometimes find larger sets than pure clique heuristics
    # because it starts with a large base that is already a clique (multiples of a single prime)
    # or a union of cliques that can be merged.
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    prime_sets_to_try = [
        [3], # Multiples of 3
        [5], # Multiples of 5
        [17], # Multiples of 17
        [2, 3, 5],
        [2, 3],
        [2, 5],
        [3, 5],
        [2, 3, 5, 7],
        [2, 3, 7],
        [2, 5, 7],
        [3, 5, 7],
        [2, 3, 5, 11],
        [2, 3, 5, 7, 11],
        [2, 3, 11],
        [2, 5, 11],
        [3, 5, 11],
        [2, 3, 5, 13],
        [2, 3, 5, 7, 13],
        [2, 3, 17],
        [3, 5, 17],
        [2, 5, 17],
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > _DEADLINE - 5:
            break
        S = _build_union_set(primes)
        S.add(255)
        S = _greedy_complete(S, 255)
        S = _local_search(S, 255, max_rounds=10)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Strategy 3: Local search on the best result so far
    if best is None:
        # Fallback: multiples of 3
        best = set(x for x in range(1, 256) if x % 3 == 0)
    best = _local_search(best, 255, max_rounds=30)
    if len(best) > best_size:
        best_size = len(best)

    # Final validation
    if not _validate(best):
        # Fallback to a known valid set
        best = set(x for x in range(1, 256) if x % 3 == 0)

    return sorted(best)

# EVOLVE-BLOCK-END
