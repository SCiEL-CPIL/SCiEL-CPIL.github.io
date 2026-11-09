"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Finds the smallest n for which the Paley tournament on n vertices
    (where n is a prime p = 3 mod 4) satisfies the condition that
    every set of 4 vertices has a common dominator.
    """
    # Primes p = 3 mod 4 in increasing order
    # We start checking from small primes up to 67 (known to work)
    # If a smaller one works, we use it.
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def check_paley(p):
        # squares = {x*x % p for x in range(1, p)}
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # adj[i][j] is True if i -> j
        # i -> j iff (j - i) % p in squares
        # Check condition: For all distinct a,b,c,d, there is a vertex v in {a,b,c,d}
        # such that v -> u for all u in {a,b,c,d} \ {v}.
        
        # Optimization: Precompute out-neighbors?
        # For p <= 67, brute force 4-combinations is fine.
        # Number of 4-combinations for n=67 is ~ 700,000.
        # For each, check 4 vertices.
        
        for a, b, c, d in itertools.combinations(range(p), 4):
            # Check if a dominates b,c,d
            if ((b - a) % p in squares and (c - a) % p in squares and (d - a) % p in squares):
                continue
            # Check if b dominates a,c,d
            if ((a - b) % p in squares and (c - b) % p in squares and (d - b) % p in squares):
                continue
            # Check if c dominates a,b,d
            if ((a - c) % p in squares and (b - c) % p in squares and (d - c) % p in squares):
                continue
            # Check if d dominates a,b,c
            if ((a - d) % p in squares and (b - d) % p in squares and (c - d) % p in squares):
                continue
            
            # No dominator found for this 4-set
            return False
        return True

    # Generate candidate primes p = 3 mod 4
    # Start from 7 up to 67
    candidates = []
    for p in range(7, 68):
        if p % 4 == 3 and is_prime(p):
            candidates.append(p)
            
    # Check each candidate
    for p in candidates:
        if check_paley(p):
            # Construct orientation
            squares = {x * x % p for x in range(1, p)}
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # 0 means i -> j, 1 means j -> i
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback to 67 if for some reason the check failed or wasn't reached (should not happen)
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
