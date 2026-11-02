"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

def _score_set(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    
    U_set = set(U)
    n = len(U_set)
    
    # Compute |U+U| and |U-U|
    sum_set = set()
    diff_set = set()
    
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    
    ratio = len(diff_set) / len(sum_set)
    score = 1 + math.log(ratio) / log_denom
    return score

def _greedy_prime_step_construction():
    """
    Use a greedy heuristic to build a set U using prime-step gaps.
    Start with U = {0}, then greedily add the next candidate value
    that maximizes the instantaneous score improvement.
    """
    U = [0]
    
    # Generate candidate values: use primes and their combinations
    # We'll generate candidates up to a reasonable limit
    max_candidates = 20000
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Generate a list of primes up to some limit
    primes = []
    candidate = 2
    while len(primes) < 1000:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    # Start with 0
    U_set = {0}
    current_score = _score_set(U)
    
    # Greedily add elements
    max_iterations = 50  # Keep the set size manageable
    
    for _ in range(max_iterations):
        best_candidate = None
        best_score = current_score
        
        # Try adding each prime not already in the set
        for p in primes:
            if p in U_set:
                continue
            # Check if p is a reasonable next value
            if p > max_candidates:
                break
            
            new_U_set = U_set | {p}
            new_score = _score_set(list(new_U_set))
            
            if new_score > best_score:
                best_score = new_score
                best_candidate = p
        
        if best_candidate is None:
            break
        
        U_set.add(best_candidate)
        current_score = best_score
    
    return sorted(U_set)

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    
    Uses a greedy heuristic based on prime-step gaps to exploit
    additive independence of primes.
    """
    U = _greedy_prime_step_construction()
    return U
# EVOLVE-BLOCK-END
