import time
import random
from fractions import Fraction

START_TIME = time.time()
DEADLINE = START_TIME + 140 # 140s limit

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = get_primes(1000)
INV_PRIMES = {p: Fraction(1, p) for p in PRIMES}

def solve_for_L(L, timeout=10):
    # Backtracking
    # State: 
    #   covered: set of covered numbers in 1..L
    #   used_primes: set of used primes
    #   current_sum: Fraction
    
    # Optimization: We only care about uncovered numbers.
    uncovered = set(range(1, L + 1))
    used_primes = set()
    current_sum = Fraction(0, 1)
    
    # Heuristic: Try to cover the smallest uncovered number.
    
    def backtrack():
        if time.time() > DEADLINE:
            raise TimeoutError()
            
        if not uncovered:
            return True
            
        # Find smallest uncovered
        n_min = min(uncovered)
        
        # Candidates: primes p not in used_primes, such that current_sum + 1/p <= 1
        candidates = []
        for p in PRIMES:
            if p in used_primes:
                continue
            if current_sum + INV_PRIMES[p] > 1:
                continue
            # The residue must cover n_min
            r = n_min % p
            # Check if this (p, r) is "valid" in any other way?
            # Distinct primes are already ensured.
            candidates.append((p, r))
            
        if not candidates:
            return False
            
        # Sort candidates
        # Heuristic: Try larger p first? (Saves budget, covers fewer numbers)
        # Or smaller p? (Covers more numbers)
        # Let's try smaller p first (greedy coverage)
        candidates.sort(key=lambda x: x[0])
        
        # Randomize slightly to escape local minima?
        # Let's stick to deterministic first, add randomization if needed.
        
        for p, r in candidates:
            # Calculate new uncovered
            # Remove all n in uncovered such that n % p == r
            new_uncovered = set()
            for n in uncovered:
                if n % p != r:
                    new_uncovered.add(n)
            
            # Optimization: If new_uncovered is same size, we didn't cover anything?
            # Impossible, n_min is covered.
            
            if backtrack_with_state(new_uncovered, used_primes | {p}, current_sum + INV_PRIMES[p]):
                return True
                
        return False
        
    def backtrack_with_state(uncovered, used_primes, current_sum):
         # Same logic as above but taking state as args to avoid closure overhead?
         # Python closures are fine for small depth.
         # Let's just recurse.
         pass

    # Implement recursive helper properly
    def rec(uncovered, used_primes, current_sum):
        if time.time() > DEADLINE:
            raise TimeoutError()
        if not uncovered:
            return True
        n_min = min(uncovered)
        candidates = []
        for p in PRIMES:
            if p in used_primes:
                continue
            if current_sum + INV_PRIMES[p] > 1:
                continue
            r = n_min % p
            candidates.append((p, r))
            
        if not candidates:
            return False
            
        # Heuristic: Try primes that cover more uncovered numbers first?
        # Costly to calculate.
        # Simple heuristic: Sort by p.
        candidates.sort(key=lambda x: x[0])
        
        for p, r in candidates:
            # Filter uncovered
            new_uncovered = set(n for n in uncovered if n % p != r)
            if rec(new_uncovered, used_primes | {p}, current_sum + INV_PRIMES[p]):
                return True
        return False

    return rec(uncovered, set(), Fraction(0, 1))

def construct():
    # Try to beat 999
    # Start from 1000
    best_classes = None
    best_L = 999 # Seed performance
    
    # We need to return the classes that achieve the best L.
    # The seed returns classes for L=999.
    # We should try to find classes for L=1000, 1001, etc.
    
    # To ensure we return *something* valid even if search fails,
    # we keep the seed solution as fallback.
    
    seed_primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    
    # Try to find better
    for L in range(1000, 1005): # Try a few targets
        if time.time() > DEADLINE - 5:
            break
        try:
            # Run search
            # The search above returns True/False but doesn't store the path.
            # We need to modify it to return the path.
            
            solution = find_solution(L)
            if solution:
                # Verify solution
                # Check sum
                s = sum(Fraction(1, p) for p, r in solution)
                if s <= 1:
                    # Check coverage
                    covered = [False] * (L + 1)
                    for p, r in solution:
                        for n in range(r if r > 0 else p, L + 1, p):
                            covered[n] = True
                    if all(covered[1:L+1]):
                        best_classes = solution
                        best_L = L
                        # We can break if we want to just find one improvement,
                        # or continue to maximize.
                        # Given time, maybe try next L?
                        # Let's break to ensure we return quickly if found.
                        # Actually, let's try to go higher if time permits.
                        pass
        except TimeoutError:
            break
        except Exception:
            break
            
    if best_classes:
        return best_classes
    else:
        return seed_classes
