import math
import time
import random
from itertools import combinations

MAX_N = 10**18
DEADLINE = 140.0

def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = generate_primes(300)

def gray_code_sequence(n_bits):
    """Generate Gray code sequence for n_bits."""
    seq = [0]
    for i in range(n_bits):
        step = 1 << i
        for j in range(len(seq)-1, -1, -1):
            seq.append(seq[j] ^ step)
    return seq

def score_gray_code(primes_list):
    """
    Score using Gray-code traversal of divisor lattice.
    primes_list: list of primes (each with exponent 1)
    Returns number of adjacent coprime pairs in sorted divisor order.
    """
    k = len(primes_list)
    if k == 0:
        return 0
    if k == 1:
        return 1  # divisors: 1, p -> gcd(1,p)=1
    
    # Generate all divisors via Gray code
    # Each divisor corresponds to a bitmask
    gray = gray_code_sequence(k)
    
    # Build divisor values from bitmasks
    divs = []
    for mask in gray:
        d = 1
        for i in range(k):
            if mask & (1 << i):
                d *= primes_list[i]
        divs.append(d)
    
    # Sort divisors
    divs.sort()
    
    # Count adjacent coprime pairs
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def evaluate_structure(structure):
    """
    structure: list of (prime, exponent) pairs
    Returns (score, n)
    """
    n = 1
    for p, e in structure:
        n *= p ** e
        if n > MAX_N:
            return -1, n
    
    # Generate all divisors
    divs = [1]
    for p, e in structure:
        new_divs = []
        for d in divs:
            for exp in range(e + 1):
                new_divs.append(d * (p ** exp))
        divs = new_divs
    
    divs.sort()
    
    # Count adjacent coprime pairs
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    
    return c, n

def construct():
    start_time = time.time()
    
    # Start with first 15 primes, exponent 1 each
    init_structure = [(PRIMES[i], 1) for i in range(15)]
    best_score, best_n = evaluate_structure(init_structure)
    best_structure = init_structure[:]
    
    # Local search: try modifications
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        current_structure = best_structure[:]
        current_score = best_score
        
        # Try adding a new prime
        for p in PRIMES:
            if any(p == cp for cp, _ in current_structure):
                continue
            new_struct = current_structure + [(p, 1)]
            s, n = evaluate_structure(new_struct)
            if s > best_score:
                best_score = s
                best_n = n
                best_structure = new_struct
                current_structure = new_struct
                current_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing a prime
        for i in range(len(current_structure)):
            new_struct = current_structure[:i] + current_structure[i+1:]
            if not new_struct:
                continue
            s, n = evaluate_structure(new_struct)
            if s > best_score:
                best_score = s
                best_n = n
                best_structure = new_struct
                current_structure = new_struct
                current_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try increasing exponent of a prime
        for i in range(len(current_structure)):
            p, e = current_structure[i]
            new_struct = current_structure[:]
            new_struct[i] = (p, e + 1)
            s, n = evaluate_structure(new_struct)
            if s > best_score:
                best_score = s
                best_n = n
                best_structure = new_struct
                current_structure = new_struct
                current_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping a prime
        for i in range(len(current_structure)):
            p_in = current_structure[i][0]
            for p_out in PRIMES:
                if p_out == p_in or any(p_out == cp for cp, _ in current_structure):
                    continue
                new_struct = current_structure[:]
                new_struct[i] = (p_out, current_structure[i][1])
                s, n = evaluate_structure(new_struct)
                if s > best_score:
                    best_score = s
                    best_n = n
                    best_structure = new_struct
                    current_structure = new_struct
                    current_score = s
                    improved = True
                    break
            if improved:
                break
    
    return best_n

construct()
