"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

# Precompute primes up to 256 and prime factorization info
_N = 256

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

PRIMES = _sieve_primes(_N)

# For each number 1..256, compute its set of prime factors
def _prime_factors(n):
    if n == 1:
        return frozenset()
    factors = frozenset()
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            factors = factors | {p}
            while n % p == 0:
                n //= p
    if n > 1:
        factors = factors | {n}
    return factors

# Precompute prime factors for all numbers
_NUM_FACTORS = [None] + [_prime_factors(i) for i in range(1, _N + 1)]

def construct():
    """
    Find a set of distinct integers from 1..256 such that every pair is coprime,
    maximizing their sum.
    
    Strategy: Branch and bound with strong upper bounds.
    Each prime can "support" at most one selected number that has that prime factor.
    Number 1 is always included (it conflicts with nothing).
    
    We use a greedy initialization followed by local search / branch-and-bound.
    """
    deadline = time.time() + 75  # Safety margin before 90s kill
    
    # Always include 1
    # For numbers 2..256, we need to select a subset where no two share a prime factor.
    
    # Group numbers by their "primary" constraint: for each number, its prime factors.
    # Key insight: each prime p can be "used" by at most one selected number.
    # So the problem is: assign each prime to at most one number, and a number can be
    # selected only if all its prime factors are assigned to it.
    
    # Better approach: This is a maximum weight independent set in a conflict graph,
    # where two numbers conflict if they share a prime factor.
    # The conflict graph is defined by shared prime factors.
    
    # Since numbers up to 256 have at most ~4 distinct prime factors, and there are
    # only 54 primes <= 256, we can use a different formulation.
    
    # For each number n > 1, it "consumes" all primes in its factorization.
    # We want to maximize sum of selected numbers subject to no prime being consumed twice.
    
    # This is equivalent to: we have 54 "resources" (primes), each number uses a subset
    # of resources, and we want max weight subset with no resource used twice.
    
    # Greedy approach: sort numbers by value/number_of_prime_factors (efficiency),
    # then greedily select if no conflict.
    
    # But first, let's try a smarter approach:
    # For each prime p, the best number that uses p (and only p among "contested" primes)
    # Actually, let's think about it differently.
    
    # Key observation: A number n with prime factors {p1, p2, ..., pk} blocks all other
    # numbers that have any of p1..pk as factors.
    
    # Optimal strategy: For primes that appear in many numbers, we should assign them
    # to the number that gives the best "value per prime used".
    
    # Let's use a branch-and-bound approach:
    # - Sort numbers by value descending
    # - At each step, decide to include or exclude the current number
    # - Upper bound: current sum + sum of all remaining numbers that don't conflict
    
    # But this could be slow. Let's try a different approach.
    
    # Actually, for n <= 256, the structure is quite constrained. Let's use a
    # maximum weight matching-like approach on the prime-number bipartite graph,
    # but with the constraint that a number needs ALL its primes.
    
    # Alternative: Since the number of primes is small (54), and each number uses
    # at most 4 primes, we can use a recursive search with memoization on the
    # set of "available" primes.
    
    # But 2^54 is too large. Let's use a different bound.
    
    # Best practical approach for this size:
    # 1. Start with all primes "available"
    # 2. Greedily pick the number with highest value that can be formed from available primes
    # 3. Repeat until no more numbers can be picked
    # 4. Then do local search: try swapping one selected number for one or two unselected
    
    # Let's implement a strong greedy + local search:
    
    # Phase 1: Greedy construction
    # For each number, compute its "cost" = set of primes it uses
    # Sort by value descending, break ties by fewer prime factors
    
    candidates = list(range(2, _N + 1))
    
    # For greedy: try multiple orderings and keep the best
    # Ordering 1: by value descending
    # Ordering 2: by value / len(prime_factors) descending (efficiency)
    # Ordering 3: by value descending, but prefer numbers with fewer prime factors
    
    def greedy_select(order):
        """Select numbers in given order, skipping conflicts."""
        used_primes = set()
        selected = [1]
        total = 1
        for n in order:
            pf = _NUM_FACTORS[n]
            if not pf:
                continue
            if pf.isdisjoint(used_primes):
                used_primes |= pf
                selected.append(n)
                total += n
        return selected, total, used_primes
    
    # Try different orderings
    orderings = [
        sorted(candidates, key=lambda x: -x),  # by value desc
        sorted(candidates, key=lambda x: (-x / len(_NUM_FACTORS[x]), -x)),  # efficiency
        sorted(candidates, key=lambda x: (len(_NUM_FACTORS[x]), -x)),  # fewest factors first
        sorted(candidates, key=lambda x: (-x / (len(_NUM_FACTORS[x]) + 0.5), -x)),
    ]
    
    best_selected = None
    best_total = 0
    best_used = None
    
    for ordering in orderings:
        sel, tot, used = greedy_select(ordering)
        if tot > best_total:
            best_total = tot
            best_selected = sel
            best_used = used
    
    # Phase 2: Local search - try to improve by swapping
    # For each selected number, try replacing it with an unselected number
    # that doesn't conflict with the rest
    
    selected_set = set(best_selected)
    available_primes = set(PRIMES) - best_used  # primes not used by selected numbers
    
    # Actually, let's think about swaps more carefully:
    # If we remove number a (freeing its primes) and add number b,
    # b must not conflict with any other selected number.
    # b's primes must be a subset of (primes freed by a) union (primes not used by any selected)
    # Wait no: b's primes must not intersect with primes used by (selected - {a})
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Build set of primes used by each selected number
        # For each selected number a, compute primes_freed = _NUM_FACTORS[a]
        # For each unselected number b, check if _NUM_FACTORS[b] is disjoint from
        # (all_used_primes - primes_freed_by_a)
        
        all_used = set()
        for s in best_selected:
            if s > 1:
                all_used |= _NUM_FACTORS[s]
        
        for a in list(best_selected):
            if a == 1:
                continue
            freed = _NUM_FACTORS[a]
            remaining_used = all_used - freed
            # Try to find b not in selected, b != a, such that _NUM_FACTORS[b] is disjoint from remaining_used
            # and b > a (to increase sum)
            best_b = None
            best_gain = 0
            for b in candidates:
                if b in selected_set:
                    continue
                pf_b = _NUM_FACTORS[b]
                if not pf_b:
                    continue
                if pf_b.isdisjoint(remaining_used):
                    gain = b - a
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            
            if best_b is not None and best_gain > 0:
                # Perform swap
                best_selected.remove(a)
                selected_set.remove(a)
                best_selected.append(best_b)
                selected_set.add(best_b)
                best_total += best_gain
                improved = True
                # Update all_used
                all_used = set()
                for s in best_selected:
                    if s > 1:
                        all_used |= _NUM_FACTORS[s]
    
    # Phase 3: Try adding more numbers if possible
    # After swaps, some primes might be free
    all_used = set()
    for s in best_selected:
        if s > 1:
            all_used |= _NUM_FACTORS[s]
    
    free_primes = set(PRIMES) - all_used
    if free_primes:
        # Try to add any number whose primes are all in free_primes
        for n in sorted(candidates, key=lambda x: -x):
            if n in selected_set:
                continue
            pf = _NUM_FACTORS[n]
            if not pf:
                continue
            if pf.issubset(free_primes):
                best_selected.append(n)
                selected_set.add(n)
                best_total += n
                all_used |= pf
                free_primes -= pf
    
    # Verify result
    result = sorted(best_selected)
    
    # Final verification: check pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # This shouldn't happen, but if it does, fall back to seed
                from math import isqrt as _isqrt
                return [1] + [a for a in range(2, _N + 1) if all(a % d for d in range(2, _isqrt(a) + 1))]
    
    # Ensure all are Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
