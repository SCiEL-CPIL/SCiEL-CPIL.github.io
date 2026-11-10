"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import math
import itertools


def _sieve_primes(limit):
    """Return list of primes up to limit using a standard sieve."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i, v in enumerate(is_prime) if v]


def _is_admissible(points, primes):
    """Check if points form an admissible tuple for the given primes."""
    for p in primes:
        if len(points) < p:
            continue
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _greedy_construct(points, primes, fixed_choices):
    """
    Run greedy construction given a starting set of points and fixed choices
    for the first few primes.
    """
    # Apply fixed choices
    for p, r in fixed_choices.items():
        points = {x for x in points if x % p != r}
        
    # Greedy for remaining primes
    for p in primes:
        if p in fixed_choices:
            continue
        if len(points) < p:
            continue
            
        # Count points in each residue class mod p
        class_counts = [0] * p
        
        for v in points:
            class_counts[v % p] += 1
        
        min_count = min(class_counts)
        if min_count > 0:
            # Find the first class with this minimum count
            for r in range(p):
                if class_counts[r] == min_count:
                    points = {x for x in points if x % p != r}
                    break
                    
    return points


def construct():
    """
    Construct the largest admissible tuple within [0, N].
    
    Strategy: Multi-start greedy construction.
    We branch on the residue class removal for the first few primes (2, 3, 5, 7, 11)
    to explore different structural paths, then use greedy min-class removal
    for the rest.
    """
    primes = _sieve_primes(N)
    
    # Primes to branch on: 2, 3, 5, 7, 11
    branch_primes = [p for p in primes if p <= 11]
    
    best_points = set()
    best_size = -1
    
    # Generate all combinations of residues to remove for these primes
    residue_ranges = [range(p) for p in branch_primes]
    
    for combo in itertools.product(*residue_ranges):
        fixed_choices = dict(zip(branch_primes, combo))
        # Start with all integers
        initial_points = set(range(N + 1))
        
        current_points = _greedy_construct(initial_points, primes, fixed_choices)
        
        if len(current_points) > best_size:
            best_size = len(current_points)
            best_points = current_points
            
    return sorted(best_points)
# EVOLVE-BLOCK-END
