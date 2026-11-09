"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
import time as _time

def _primes_up_to(n):
    """Return list of primes up to n using sieve."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _sum_recip(classes):
    """Compute sum of 1/p for given classes [[p, r], ...] as Fraction."""
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def _covered_length(classes, cap=100000):
    """Return the largest L such that every integer 1..L is in some class.
    A class [p, r] covers integers n where n ≡ r (mod p).
    """
    if not classes:
        return 0
    # Build coverage array up to cap
    covered = bytearray(cap + 1)  # 0-indexed, covered[i] means i+1 is covered
    for p, r in classes:
        # Integers n ≡ r (mod p), i.e., n = r + k*p for k >= 0
        # We need n in 1..cap
        # r is in 0..p-1
        # The smallest positive n ≡ r (mod p): if r == 0, n = p; else n = r
        if r == 0:
            start = p
        else:
            start = r
        if start > cap:
            continue
        for n in range(start, cap + 1, p):
            covered[n - 1] = 1  # index n-1 corresponds to integer n
    
    # Find largest L such that all 1..L are covered
    L = 0
    for i in range(cap):
        if covered[i]:
            L = i + 1
        else:
            break
    return L

def construct():
    """Construct a set of prime residue classes covering 1..L maximally,
    with sum of 1/p <= 1."""
    start_time = _time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    primes = _primes_up_to(1000)
    
    # Strategy: We want to maximize the covered length L.
    # The key insight is that we need to cover consecutive integers starting from 1.
    # A greedy approach: try to cover as many consecutive integers as possible.
    
    # First, let's use a heuristic: assign each prime a residue that helps cover
    # the current "frontier" (the first uncovered integer).
    
    # Start with all primes, try to find a good assignment.
    # We'll use a greedy algorithm:
    # 1. Start with no classes.
    # 2. Repeatedly add the prime/residue that extends the covered length the most,
    #    while keeping sum of 1/p <= 1.
    
    # Actually, let's think about this differently.
    # The seed program uses primes from 13 upward, assigning p_i to cover i+1.
    # That gives a sum of about 0.92 and covers up to some L.
    
    # Let's try a different approach: use a subset of primes with carefully chosen residues.
    
    # Key idea: To cover 1, we need some class [p, r] where 1 ≡ r (mod p), i.e., r = 1 (since 1 < p for p >= 2).
    # To cover 2, we need some class where 2 ≡ r (mod p).
    # etc.
    
    # Let's try a greedy approach:
    # - Maintain a set of used primes and their residues.
    # - Maintain the current covered length L.
    # - At each step, try adding a new prime p with residue r that extends L the most,
    #   while keeping sum(1/p) <= 1.
    
    # For efficiency, we'll precompute which integers each (p, r) pair covers.
    
    # Let's start with the seed approach as a baseline, then try to improve.
    
    # Seed approach: primes from 13 upward, p_i covers i+1
    seed_classes = []
    seed_sum = Fraction(0, 1)
    for i, p in enumerate(primes):
        if p < 13:
            continue
        r = (i + 1) % p  # This is wrong in the seed; let me re-examine
        # Actually, the seed does: for i, p in enumerate(primes) where primes starts at 13
        # primes[0] = 13, so r = (0+1) % 13 = 1, covers 1, 14, 27, ...
        # primes[1] = 17, r = (1+1) % 17 = 2, covers 2, 19, 36, ...
        # etc.
        # This is a valid construction.
        pass
    
    # Let me just implement the seed and compute its L, then try to improve.
    
    # Seed implementation:
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    seed_sum = sum(Fraction(1, p) for p, r in seed_classes)
    seed_L = _covered_length(seed_classes)
    
    best_classes = seed_classes
    best_L = seed_L
    best_sum = seed_sum
    
    # Now try to improve:
    # Idea: We have some slack in the budget (sum < 1). We can try to add more classes
    # or reassign residues to extend the covered length.
    
    # Let's try a local search:
    # 1. Try changing residues of existing classes to extend L.
    # 2. Try adding new classes (primes not yet used) if budget allows.
    # 3. Try removing classes that don't help and using their budget for better ones.
    
    # For simplicity, let's try a greedy addition of small primes with good residues.
    
    # Actually, let's try a different strategy: use all primes up to some limit,
    # and find residues that maximize the covered length.
    
    # This is a hard combinatorial problem. Let's use a heuristic:
    # For each prime p, choose residue r to cover the current frontier (first uncovered int).
    
    # Let's implement a greedy frontier-filling algorithm:
    
    def greedy_construct(prime_list, max_time=100):
        """Greedy algorithm to fill the frontier."""
        start_t = _time.time()
        classes = []
        used_primes = set()
        current_sum = Fraction(0, 1)
        
        # Precompute for each prime p, the residues that cover specific integers
        # For a given target integer t, p covers t if t mod p == r, so r = t % p
        
        # We'll maintain the covered set as a boolean array
        cap = 100000
        covered = bytearray(cap + 1)
        current_L = 0
        
        # Sort primes by size (smaller primes are more "valuable" per unit budget)
        sorted_primes = sorted(prime_list, key=lambda p: p)
        
        for p in sorted_primes:
            if _time.time() - start_t > max_time:
                break
            if current_sum + Fraction(1, p) > Fraction(1, 1):
                continue
            
            # Find the best residue r for this prime
            # Try to extend the frontier
            best_r = None
            best_new_L = current_L
            
            # Try all residues
            for r in range(p):
                # Simulate adding this class
                # Check if it extends the frontier
                new_L = current_L
                # The new class covers r, r+p, r+2p, ...
                # Check if it helps extend beyond current_L
                # First, find the first uncovered integer after current_L
                # If the new class covers it, extend further
                
                # Quick check: does this residue cover current_L + 1?
                if current_L < cap:
                    target = current_L + 1
                    if target % p == r:
                        # This residue covers the frontier
                        # How far does it extend?
                        # It covers target, but we need to check if consecutive integers are covered
                        # by the combination of existing classes and this new one
                        # This is expensive, so let's just take it if it covers the frontier
                        best_r = r
                        best_new_L = current_L + 1  # optimistic
                        break
                else:
                    break
            
            if best_r is not None:
                classes.append([p, best_r])
                used_primes.add(p)
                current_sum += Fraction(1, p)
                # Update covered array
                if best_r == 0:
                    start_n = p
                else:
                    start_n = best_r
                if start_n <= cap:
                    for n in range(start_n, cap + 1, p):
                        covered[n - 1] = 1
                
                # Recompute current_L
                new_L = 0
                for i in range(cap):
                    if covered[i]:
                        new_L = i + 1
                    else:
                        break
                current_L = new_L
        
        return classes, current_L
    
    # Try greedy
    greedy_classes, greedy_L = greedy_construct(primes, max_time=100)
    if greedy_L > best_L:
        best_classes = greedy_classes
        best_L = greedy_L
        best_sum = sum(Fraction(1, p) for p, r in greedy_classes)
    
    # Try to improve by local search on the best solution
    if _time.time() < deadline - 5:
        # Try changing residues
        improved = True
        while improved and _time.time() < deadline - 5:
            improved = False
            for idx in range(len(best_classes)):
                p, r = best_classes[idx]
                # Try other residues
                for new_r in range(p):
                    if new_r == r:
                        continue
                    # Test if this improves L
                    test_classes = best_classes[:]
                    test_classes[idx] = [p, new_r]
                    test_L = _covered_length(test_classes, cap=min(100000, best_L + 10000))
                    if test_L > best_L:
                        best_classes = test_classes
                        best_L = test_L
                        improved = True
                        break
    
    # Ensure the sum constraint is met
    final_sum = sum(Fraction(1, p) for p, r in best_classes)
    if final_sum > Fraction(1, 1):
        # Fallback to seed
        best_classes = seed_classes
        best_L = seed_L
    
    return best_classes
# EVOLVE-BLOCK-END
