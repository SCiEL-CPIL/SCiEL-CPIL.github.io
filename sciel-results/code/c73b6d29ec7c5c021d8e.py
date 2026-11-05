"""Rewritten program: CSP-style backtracking with prime-power basis and memoization."""
import sys
from functools import lru_cache
from itertools import product as iproduct

# Precompute primes up to 128
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

_PRIMES = _primes_up_to(128)
_NUM_PRIMES = len(_PRIMES)

# Map each number in [2,128] to its prime exponent vector (tuple)
_EXP_VECTORS = {}
for x in range(2, 129):
    v = [0] * _NUM_PRIMES
    n = x
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    _EXP_VECTORS[x] = tuple(v)

# We want to maximize the length of a strictly increasing sequence in [2,128]
# such that all consecutive-block products are distinct.
#
# Key insight: if we build the sequence greedily but with a smarter selection
# strategy, we can do better than the naive left-to-right greedy.
#
# Approach: Use backtracking with memoization. State = (last_value, frozenset_of_products_so_far)
# But the product set can be large. Instead, use a heuristic:
#
# The naive greedy picks the smallest possible next value. This may get stuck early.
# A better strategy: try to pick values that are "prime-rich" or have unique prime factors
# to reduce collisions in block products.
#
# However, given the 60s time limit and the small range [2,128], we can afford a
# more sophisticated search. Let's use a branch-and-bound with memoization on
# (last_value, set_of_block_products). But the set of block products can be huge.
#
# Alternative: Since the naive greedy already gives a good baseline, let's try to
# improve it by looking ahead. We'll use a recursive backtracking that tries to
# maximize the remaining length, with memoization keyed on (last_value, frozenset(products)).
#
# But frozenset(products) is too large for memoization. Instead, let's try a different
# approach: precompute for each candidate x, the set of new block products it would add,
# and use a greedy-with-lookahead strategy.
#
# Actually, let's try a simpler improvement: instead of always picking the smallest valid
# next element, we pick the one that minimizes the number of "conflicts" with future
# candidates. We'll do this via a priority queue / best-first search on the current
# sequence, expanding the most promising branches first.
#
# Given time constraints, let's implement a backtracking search with a good heuristic
# ordering: try candidates in order of their "rarity" (numbers with more distinct prime
# factors or higher prime powers are rarer and thus less likely to cause conflicts).

def _rarity_score(x):
    """Score based on how 'unique' the number is in terms of prime factorization."""
    v = _EXP_VECTORS[x]
    # Count distinct primes and total exponent sum
    distinct = sum(1 for e in v if e > 0)
    total_exp = sum(v)
    # Rarer numbers have more distinct primes or higher exponents
    return (distinct, total_exp, -x)  # Higher is rarer

def construct():
    import sys
    
    def get_tails(x, current_values):
        """Get the new block products if we append x to current_values."""
        tails = [x]
        prod = x
        for a in reversed(current_values):
            prod *= a
            tails.append(prod)
        return tails

    def baseline_greedy():
        values = []
        products = set()
        remaining = set(range(2, 129))
        
        for _ in range(127):
            best_next = None
            best_score = -1
            best_tails = None
            
            for x in sorted(remaining):
                if values and x <= values[-1]:
                    continue
                
                tails = get_tails(x, values)
                if products.intersection(tails):
                    continue
                
                new_products = products.union(tails)
                future_valid = 0
                for y in sorted(remaining):
                    if y <= x:
                        continue
                    y_tails = [y]
                    prod_y = y
                    for a in reversed(values + [x]):
                        prod_y *= a
                        y_tails.append(prod_y)
                    if not new_products.intersection(y_tails):
                        future_valid += 1
                
                score = future_valid
                if score > best_score:
                    best_score = score
                    best_next = x
                    best_tails = tails
            
            if best_next is None:
                break
            
            values.append(best_next)
            products.update(best_tails)
            remaining.discard(best_next)
            
        return values

    best_seq = baseline_greedy()
    best_len = len(best_seq)
    
    sys.setrecursionlimit(10000)
    nodes_visited = 0
    MAX_NODES = 200000
    
    def dfs(last_val, prods_frozen, current_seq):
        nonlocal best_len, best_seq, nodes_visited
        
        nodes_visited += 1
        if nodes_visited > MAX_NODES:
            return False
        
        if len(current_seq) + (128 - last_val) <= best_len:
            return True
            
        for next_val in range(last_val + 1, 129):
            suff = [1]
            p = 1
            for v in reversed(current_seq):
                p *= v
                suff.append(p)
                
            new_prods_list = [next_val * s for s in suff]
            new_prods_set = frozenset(new_prods_list)
            
            if prods_frozen.intersection(new_prods_set):
                continue
                
            new_prods = prods_frozen.union(new_prods_set)
            new_seq = current_seq + [next_val]
            
            if len(new_seq) > best_len:
                best_len = len(new_seq)
                best_seq = new_seq
                
            if not dfs(next_val, new_prods, new_seq):
                return False
                
        return True
        
    dfs(1, frozenset(), [])
    
    return best_seq
