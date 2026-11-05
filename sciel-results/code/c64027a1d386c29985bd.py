"""
CSP-style backtracking for dense increasing sequences with unique block products.

Approach:
- Model each number in [2,128] by its prime-exponent vector.
- Build a strictly increasing sequence by backtracking.
- Track all consecutive-block products as exponent vectors (tuples).
- Use memoization on (last_value, frozenset_of_product_vectors) to avoid re-exploring states.
- Prune aggressively using the upper bound on remaining elements.
- Order candidates by a heuristic that favors numbers with rarer prime signatures.
"""
import sys
from functools import lru_cache

# Precompute primes up to 128
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    if n >= 0:
        sieve[0] = False
    if n >= 1:
        sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

_PRIMES = _primes_up_to(128)
_NUM_PRIMES = len(_PRIMES)

# Map each number in [2,128] to its prime exponent vector
_EXP_VECTORS = {}
for x in range(2, 129):
    v = [0] * _NUM_PRIMES
    n = x
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    _EXP_VECTORS[x] = tuple(v)

# Precompute for each number, the "rarity" score for ordering
def _rarity_score(x):
    v = _EXP_VECTORS[x]
    distinct = sum(1 for e in v if e > 0)
    total_exp = sum(v)
    return (distinct, total_exp, -x)

# Sort candidates by rarity (higher first)
_CANDIDATES_BY_RARITY = sorted(range(2, 129), key=_rarity_score, reverse=True)

def construct():
    sys.setrecursionlimit(10000)
    
    best_seq = []
    best_len = 0
    nodes_visited = 0
    MAX_NODES = 500000
    
    # Memoization: state -> best length achievable from this state
    # Key: (last_val, frozenset_of_product_vectors)
    memo = {}
    
    def dfs(last_val, prods_frozen, current_seq):
        nonlocal best_len, best_seq, nodes_visited
        
        nodes_visited += 1
        if nodes_visited > MAX_NODES:
            return 0
        
        # Update best if current is longer
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
        
        # Pruning: upper bound on remaining
        remaining_upper = 128 - last_val
        if len(current_seq) + remaining_upper <= best_len:
            return 0
        
        # Check memo
        state_key = (last_val, prods_frozen)
        if state_key in memo:
            return memo[state_key]
        
        max_additional = 0
        
        # Try candidates in order of rarity, but only those > last_val
        for next_val in range(last_val + 1, 129):
            # Compute new block product vectors
            # New blocks ending at next_val: [next_val], [next_val, last], [next_val, last, prev], ...
            new_prods_list = []
            vec = _EXP_VECTORS[next_val]
            new_prods_list.append(vec)
            
            for elem in reversed(current_seq):
                ev = _EXP_VECTORS[elem]
                vec = tuple(a + b for a, b in zip(vec, ev))
                new_prods_list.append(vec)
            
            new_prods_set = frozenset(new_prods_list)
            
            # Check for conflicts with existing products
            if prods_frozen.intersection(new_prods_set):
                continue
            
            new_prods = prods_frozen.union(new_prods_set)
            new_seq = current_seq + [next_val]
            
            additional = dfs(next_val, new_prods, new_seq)
            if additional + 1 > max_additional:
                max_additional = additional + 1
        
        memo[state_key] = max_additional
        return max_additional
    
    dfs(1, frozenset(), [])
    
    return best_seq
