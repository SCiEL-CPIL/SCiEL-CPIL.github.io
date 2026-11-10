"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import math
import itertools
import numpy as np


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


def _greedy_construct_np(points, primes, fixed_choices):
    """
    Run greedy construction using NumPy for speed.
    points: numpy array of integers.
    """
    for p in primes:
        if p in fixed_choices:
            r = fixed_choices[p]
            points = points[points % p != r]
            continue
            
        if len(points) < p:
            continue
            
        # Count points in each residue class mod p
        residues = points % p
        class_counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(class_counts)
        if min_count > 0:
            # Find the first class with this minimum count
            # We pick the first index where count is minimal
            min_indices = np.where(class_counts == min_count)[0]
            r = min_indices[0]
            points = points[points % p != r]
            
    return points


def construct():
    """
    Construct the largest admissible tuple within [0, N].
    
    Strategy: Multi-start greedy construction with NumPy acceleration.
    We branch on the residue class removal for the first few primes (2, 3, 5, 7, 11, 13)
    to explore different structural paths, then use greedy min-class removal
    for the rest.
    """
    primes = _sieve_primes(N)
    
    # Primes to branch on: 2, 3, 5, 7, 11, 13
    branch_primes = [p for p in primes if p <= 13]
    
    best_points = np.array([], dtype=int)
    best_size = -1
    
    # Generate all combinations of residues to remove for these primes
    residue_ranges = [range(p) for p in branch_primes]
    
    initial_points = np.arange(N + 1, dtype=int)
    
    for combo in itertools.product(*residue_ranges):
        fixed_choices = dict(zip(branch_primes, combo))
        
        current_points = _greedy_construct_np(initial_points, primes, fixed_choices)
        
        if len(current_points) > best_size:
            best_size = len(current_points)
            best_points = current_points
            
    return sorted(best_points.tolist())
# EVOLVE-BLOCK-END
