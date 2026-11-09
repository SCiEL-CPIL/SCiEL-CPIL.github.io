"""
Constructs a maximum-cardinality subset of [1, 255] containing 255
where every distinct pair has gcd > 1.

Uses Bron-Kerbosch with pivot to find the maximum clique in the
compatibility graph (edges where gcd > 1), restricted to candidates
that share a factor with 255 (i.e., divisible by 3, 5, or 17).
Falls back to multiples of 3 (size 85) if search exceeds time budget.
"""

from math import gcd
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b) > 1
    for all distinct pairs, maximizing cardinality.
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Candidate pool: all numbers in [1,255] that share a factor with 255
    # 255 = 3 * 5 * 17, so candidates are multiples of 3, 5, or 17
    candidates = [x for x in range(1, 256) if gcd(x, 255) > 1]
    
    # Build adjacency: adj[i] = set of j > i such that gcd(candidates[i], candidates[j]) > 1
    n = len(candidates)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find index of 255 in candidates (must be included)
    idx_255 = candidates.index(255)
    
    # Bron-Kerbosch with pivot, forced inclusion of 255
    # We search for max clique containing idx_255
    best_clique = set([idx_255])
    
    # Use Bron-Kerbosch: R = {255}, P = candidates compatible with 255, X = empty
    P = set(adj[idx_255])  # candidates that can be added alongside 255
    R = {idx_255}
    X = set()
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = set(R)
            return
        # Pivot selection: choose u in P ∪ X with max |N(u) ∩ P|
        U = P | X
        if U:
            pivot = max(U, key=lambda u: len(adj[u] & P))
        else:
            pivot = None
        # Candidates to branch on: P \ N(pivot)
        candidates_to_branch = P - adj[pivot] if pivot is not None else P()
        for v in list(candidates_to_branch):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
            if time.time() > deadline:
                return
    
    bron_kerbosch(R, P, X)
    
    # Convert best clique back to values
    result = sorted(candidates[i] for i in best_clique)
    
    # Fallback: multiples of 3 (guaranteed valid, size 85)
    fallback = list(range(3, 256, 3))
    if len(result) < len(fallback):
        result = fallback
    
    return result
