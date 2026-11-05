"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # Generate primes up to 1000
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = gen_primes(1000)
    MAX_L = 100000
    
    # We'll build a covering system greedily to maximize contiguous prefix coverage
    # Track coverage as a boolean array
    covered = [False] * (MAX_L + 1)
    used_primes = set()
    classes = []
    budget = Fraction(1, 1)
    
    # Greedy: at each step, find the prime-residue that extends the prefix the most
    # First, find current prefix length
    def get_prefix(covered):
        L = 0
        while L + 1 <= MAX_L and covered[L + 1]:
            L += 1
        return L
    
    current_prefix = 0
    
    # We'll iterate: try to extend coverage
    # For efficiency, we'll consider primes in order of size (smaller = cheaper)
    # and for each prime, try residues that cover the first uncovered number
    
    for _ in range(len(primes)):  # at most one class per prime
        best_gain = 0
        best_prime = None
        best_residue = None
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget - cost < 0:
                continue
            
            # Try each residue
            for r in range(p):
                # Check how much this extends the prefix
                # First, find the first uncovered number
                first_uncovered = current_prefix + 1
                if first_uncovered > MAX_L:
                    break
                
                # If this residue covers first_uncovered, see how far it extends
                if first_uncovered % p == r:
                    # Count how many consecutive numbers from first_uncovered are covered
                    # by this class (considering existing coverage)
                    gain = 0
                    n = first_uncovered
                    while n <= MAX_L and (covered[n] or n % p == r):
                        if not covered[n]:
                            gain += 1
                        n += 1
                        if n - first_uncovered > 10000:  # cap for speed
                            break
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_prime = p
                        best_residue = r
        
        if best_prime is None or best_gain == 0:
            break
        
        # Apply the best choice
        p = best_prime
        r = best_residue
        used_primes.add(p)
        classes.append([p, r])
        budget -= Fraction(1, p)
        
        # Update coverage
        for n in range(r if r > 0 else p, MAX_L + 1, p):
            covered[n] = True
        
        # Update prefix
        new_prefix = get_prefix(covered)
        if new_prefix <= current_prefix:
            # This didn't help, but we already committed
            pass
        current_prefix = new_prefix
    
    # Verify the result
    # Find the actual maximum L
    L = 0
    while L + 1 <= MAX_L and covered[L + 1]:
        L += 1
    
    # Double-check budget
    total = sum(Fraction(1, p) for p, r in classes)
    assert total <= 1, f"Budget exceeded: {total}"
    
    return classes
# EVOLVE-BLOCK-END
