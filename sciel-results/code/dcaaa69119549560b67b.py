"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Find 12 distinct positive integers <= 10000 that minimize the number of
    distinct prime factors across all pairwise sums (distinct indices).
    
    Strategy:
    1. Precompute smallest prime factor (SPF) table up to 20000.
    2. For small prime sets P (starting with {2,3}), find a clique of size 12
       in the compatibility graph where edge (u,v) exists if u+v is P-smooth.
    3. If no clique of size 12 is found for {2,3}, try larger prime sets.
    4. Use greedy maximal clique heuristic with randomization.
    """
    deadline = time.time() + 75  # Safety margin before 90s kill
    
    MAX_VAL = 10000
    N = 12
    
    # Step 1: Build SPF table up to 2*MAX_VAL
    limit = 2 * MAX_VAL
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        """Get distinct prime factors of n using SPF table."""
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors
    
    def is_p_smooth(n, prime_set):
        """Check if all prime factors of n are in prime_set."""
        if n <= 1:
            return True
        while n > 1:
            p = spf[n]
            if p not in prime_set:
                return False
            while n % p == 0:
                n //= p
        return True
    
    # Try different prime sets in order of increasing size
    # Start with the most promising: {2, 3}
    prime_sets_to_try = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
    ]
    
    # Also try some other small sets
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for size in range(2, 6):
        for combo in combinations(all_primes[:10], size):
            s = set(combo)
            if s not in prime_sets_to_try:
                prime_sets_to_try.append(s)
    
    # Sort by size, then by sum of primes (smaller primes = more smooth numbers)
    prime_sets_to_try.sort(key=lambda s: (len(s), sum(s)))
    
    best_result = None
    best_prime_count = float('inf')
    
    for prime_set in prime_sets_to_try:
        if time.time() > deadline:
            break
        
        # Build candidate list: numbers 1..10000
        # For a clique, we need all pairwise sums to be P-smooth
        # This is very restrictive for small P
        
        # Strategy: find all pairs (u,v) where u+v is P-smooth
        # Then find a clique of size 12 in this graph
        
        # For efficiency, let's think about structure:
        # If P = {2, 3}, then u+v must be of form 2^a * 3^b
        # The possible values of u+v range from 3 to 19999
        
        # Let's generate all P-smooth numbers up to 2*MAX_VAL
        smooth_numbers = []
        for s in range(3, 2 * MAX_VAL + 1):
            if is_p_smooth(s, prime_set):
                smooth_numbers.append(s)
        
        if len(smooth_numbers) < N * (N - 1) // 2:
            continue  # Not enough smooth numbers to support a clique of size N
        
        # For each smooth number s, the pairs (u, s-u) where 1 <= u < s-u <= MAX_VAL
        # We need to select N numbers such that all C(N,2) sums are in smooth_numbers
        
        # Greedy approach: 
        # 1. Start with a random seed number
        # 2. Add compatible numbers one by one
        # 3. Repeat with different seeds
        
        # Precompute for each number x, which other numbers y are compatible
        # (i.e., x+y is P-smooth)
        
        # This is expensive to precompute fully, so let's be smart:
        # For a given x, y must satisfy: x+y in smooth_numbers
        # So y in {s - x for s in smooth_numbers if 1 <= s-x <= MAX_VAL and s-x != x}
        
        def find_compatible(x, smooth_set):
            """Find all y != x such that x+y is P-smooth."""
            compat = []
            for s in smooth_set:
                y = s - x
                if 1 <= y <= MAX_VAL and y != x:
                    compat.append(y)
            return compat
        
        smooth_set = set(smooth_numbers)
        
        # Try multiple random starting points
        random.seed(42)
        found = False
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Start with a random number
            start = random.randint(1, MAX_VAL)
            clique = [start]
            
            # Greedily add compatible numbers
            for _ in range(N - 1):
                # Find candidates compatible with all current clique members
                candidates = set(range(1, MAX_VAL + 1))
                for c in clique:
                    candidates &= set(find_compatible(c, smooth_set))
                
                if not candidates:
                    break
                
                # Choose a random candidate (or the one with highest degree)
                # For speed, just pick randomly from a subset
                cand_list = list(candidates)
                if len(cand_list) > 100:
                    cand_list = random.sample(cand_list, 100)
                
                next_node = random.choice(cand_list)
                clique.append(next_node)
            
            if len(clique) == N:
                found = True
                # Verify
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not is_p_smooth(clique[i] + clique[j], prime_set):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_result = clique
                    best_prime_count = len(prime_set)
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: use the seed program's result
        best_result = list(range(1, 13))
    
    return best_result
# EVOLVE-BLOCK-END
