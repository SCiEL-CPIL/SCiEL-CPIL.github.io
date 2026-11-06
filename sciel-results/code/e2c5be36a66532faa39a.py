"""Stronger algorithm for the no-element-divides-sum-of-3-others task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_K = 3
_MAX = 48

def _precompute_triples():
    """For each candidate x, precompute the set of sums of k=3 distinct other numbers
    from the full universe [1..N] that are divisible by x.
    Actually we need: for a selected set S, for each a in S, no sum of 3 distinct
    elements of S\{a} is divisible by a.
    
    We'll precompute for each a, the list of all triples (b,c,d) with b,c,d distinct,
    b,c,d != a, and (b+c+d) % a == 0. Then when building S, we check that no such
    triple is fully contained in S\{a}.
    """
    triples_divisible = {}
    for a in range(1, _N + 1):
        triples = []
        # b < c < d, all != a, b,c,d in [1..N]
        for b in range(1, _N + 1):
            if b == a:
                continue
            for c in range(b + 1, _N + 1):
                if c == a:
                    continue
                for d in range(c + 1, _N + 1):
                    if d == a:
                        continue
                    if (b + c + d) % a == 0:
                        triples.append((b, c, d))
        triples_divisible[a] = triples
    return triples_divisible

_TRIPLES_DIV = _precompute_triples()

def _is_valid_set(S):
    """Check if set S is valid: no element divides sum of any 3 distinct others."""
    S_list = list(S)
    S_set = set(S_list)
    for a in S_list:
        for (b, c, d) in _TRIPLES_DIV[a]:
            if b in S_set and c in S_set and d in S_set:
                return False
    return True

def _greedy_descending():
    """Greedy: add elements from large to small if valid."""
    selected = []
    selected_set = set()
    for x in range(_N, 0, -1):
        candidate_set = selected_set | {x}
        # Check validity: only need to check new element x and existing elements
        # that might now have a new triple completed by adding x
        valid = True
        # Check x: does x divide sum of any 3 distinct elements in selected_set?
        # i.e., is there a triple (b,c,d) in selected_set with (b+c+d)%x==0?
        for (b, c, d) in _TRIPLES_DIV[x]:
            if b in selected_set and c in selected_set and d in selected_set:
                valid = False
                break
        if not valid:
            continue
        # Check existing elements: adding x might complete a triple for some a in selected
        for a in selected:
            for (b, c, d) in _TRIPLES_DIV[a]:
                # Check if x is one of b,c,d and the other two are in selected_set
                if b == x and c in selected_set and d in selected_set:
                    valid = False
                    break
                if c == x and b in selected_set and d in selected_set:
                    valid = False
                    break
                if d == x and b in selected_set and c in selected_set:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_random_order(seed):
    """Greedy with random permutation."""
    rng = random.Random(seed)
    order = list(range(1, _N + 1))
    rng.shuffle(order)
    selected = []
    selected_set = set()
    for x in order:
        valid = True
        for (b, c, d) in _TRIPLES_DIV[x]:
            if b in selected_set and c in selected_set and d in selected_set:
                valid = False
                break
        if not valid:
            continue
        for a in selected:
            for (b, c, d) in _TRIPLES_DIV[a]:
                if b == x and c in selected_set and d in selected_set:
                    valid = False
                    break
                if c == x and b in selected_set and d in selected_set:
                    valid = False
                    break
                if d == x and b in selected_set and c in selected_set:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(S, max_iters=200, seed=0):
    """Try to improve set S by swapping elements in for out."""
    S = list(S)
    S_set = set(S)
    rng = random.Random(seed)
    best = S
    best_len = len(S)
    
    for iteration in range(max_iters):
        improved = False
        # Try removing one element and adding another
        for i in range(len(S)):
            a = S[i]
            # Try to find a replacement
            candidates = [x for x in range(1, _N + 1) if x not in S_set]
            rng.shuffle(candidates)
            for x in candidates:
                if x == a:
                    continue
                new_S = S[:i] + S[i+1:] + [x]
                new_set = (S_set - {a}) | {x}
                if _is_valid_set(new_set):
                    S = new_S
                    S_set = new_set
                    improved = True
                    if len(S) > best_len:
                        best = S[:]
                        best_len = len(S)
                    break
            if improved:
                break
        if not improved:
            break
    return best

def _try_primes_and_large():
    """Strategy: prioritize primes and large numbers."""
    selected = []
    selected_set = set()
    # First add all primes > 96 (large primes)
    primes = [x for x in range(97, _N + 1) if all(x % p != 0 for p in range(2, int(x**0.5)+1))]
    primes.sort(reverse=True)
    for x in primes:
        valid = True
        for (b, c, d) in _TRIPLES_DIV[x]:
            if b in selected_set and c in selected_set and d in selected_set:
                valid = False
                break
        if not valid:
            continue
        for a in selected:
            for (b, c, d) in _TRIPLES_DIV[a]:
                if b == x and c in selected_set and d in selected_set:
                    valid = False
                    break
                if c == x and b in selected_set and d in selected_set:
                    valid = False
                    break
                if d == x and b in selected_set and c in selected_set:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    # Then try remaining numbers descending
    for x in range(_N, 0, -1):
        if x in selected_set:
            continue
        valid = True
        for (b, c, d) in _TRIPLES_DIV[x]:
            if b in selected_set and c in selected_set and d in selected_set:
                valid = False
                break
        if not valid:
            continue
        for a in selected:
            for (b, c, d) in _TRIPLES_DIV[a]:
                if b == x and c in selected_set and d in selected_set:
                    valid = False
                    break
                if c == x and b in selected_set and d in selected_set:
                    valid = False
                    break
                if d == x and b in selected_set and c in selected_set:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Get initial solutions from different strategies
    solutions = []
    
    sol1 = _greedy_descending()
    solutions.append(sol1)
    
    if time.time() > deadline:
        return max(solutions, key=len)
    
    sol2 = _try_primes_and_large()
    solutions.append(sol2)
    
    if time.time() > deadline:
        return max(solutions, key=len)
    
    # Try several random greedy orders
    for seed in range(50):
        if time.time() > deadline:
            break
        sol = _greedy_random_order(seed)
        solutions.append(sol)
    
    # Pick best so far
    best = max(solutions, key=len)
    best_set = set(best)
    
    # Local search to try to improve
    if time.time() < deadline - 5.0:
        improved = _local_search(best, max_iters=100, seed=42)
        if len(improved) > len(best):
            best = improved
            best_set = set(best)
    
    # Try more local search from different starting points if time allows
    if time.time() < deadline - 5.0:
        for sol in solutions[:5]:
            if time.time() > deadline:
                break
            improved = _local_search(sol, max_iters=50, seed=random.randint(0, 10000))
            if len(improved) > len(best):
                best = improved
                best_set = set(best)
    
    # Final validation
    if not _is_valid_set(best_set):
        # Fallback to a known valid solution
        best = _greedy_descending()
    
    return best
# EVOLVE-BLOCK-END
