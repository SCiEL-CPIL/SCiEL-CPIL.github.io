"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _compute_coverage(classes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by the classes."""
    if not classes:
        return 0
    covered = [False] * (max_check + 1)
    for p, r in classes:
        for x in range(r if r > 0 else p, max_check + 1, p):
            if x >= 1:
                covered[x] = True
    # Find the largest L such that 1..L are all covered
    L = 0
    for i in range(1, max_check + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _greedy_cover(primes, deadline):
    """Greedy algorithm to find prime residue classes that maximize coverage."""
    best_L = 0
    best_classes = []
    
    # Sort primes: try larger primes first (they cover more per unit budget)
    # But also try different orderings
    for order_name, prime_order in [
        ("desc", sorted(primes, reverse=True)),
        ("asc", sorted(primes)),
        ("middle_out", sorted(primes, key=lambda p: abs(p - 500))),
    ]:
        if time.time() > deadline:
            break
        
        classes = []
        budget = Fraction(1)
        covered = [False] * 100001  # 1..100000
        
        for p in prime_order:
            if time.time() > deadline:
                break
            
            cost = Fraction(1, p)
            if budget < cost:
                continue
            
            # Try all possible residues for this prime
            best_r = None
            best_gain = 0
            
            # Find the first uncovered number
            first_uncovered = None
            for i in range(1, 100001):
                if not covered[i]:
                    first_uncovered = i
                    break
            
            if first_uncovered is None:
                break
            
            # Try residues that cover first_uncovered
            # For residue r, it covers numbers: r, r+p, r+2p, ...
            # We want r such that first_uncovered ≡ r (mod p)
            r_candidate = first_uncovered % p
            if r_candidate == 0:
                r_candidate = p  # But we need 0 <= r < p, so r=0 means multiples of p
            
            # Actually, let's try a few candidate residues around the first uncovered
            candidates = set()
            # The residue that directly covers first_uncovered
            candidates.add(first_uncovered % p)
            # Also try nearby residues
            for delta in [-2, -1, 0, 1, 2]:
                r = (first_uncovered + delta) % p
                candidates.add(r)
            
            for r in candidates:
                # Count how many new numbers this would cover
                gain = 0
                for x in range(r if r > 0 else p, 100001, p):
                    if x >= 1 and not covered[x]:
                        gain += 1
                    if x > first_uncovered + p * 2:
                        break  # Early termination
                
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r is not None and best_gain > 0:
                classes.append([p, best_r])
                budget -= cost
                # Update covered array
                for x in range(best_r if best_r > 0 else p, 100001, p):
                    if x >= 1:
                        covered[x] = True
        
        # Compute the actual coverage
        L = _compute_coverage(classes, 100000)
        if L > best_L:
            best_L = L
            best_classes = classes[:]
    
    return best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s limit)
    
    primes = _sieve(1000)
    
    # Try greedy approach
    classes = _greedy_cover(primes, deadline)
    
    # Verify the result
    if classes:
        L = _compute_coverage(classes, 100000)
    else:
        L = 0
    
    return classes
# EVOLVE-BLOCK-END
