"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def _coverage_length(classes, limit=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    covered = [False] * (limit + 1)
    for p, r in classes:
        # Mark all n in 1..limit where n % p == r
        start = r
        if start == 0:
            start = p
        # Actually: n ≡ r (mod p), n >= 1
        # First n >= 1 with n ≡ r (mod p): if r == 0, first is p; else first is r (if r>=1)
        if r == 0:
            first = p
        else:
            first = r
        n = first
        while n <= limit:
            covered[n] = True
            n += p
    # Find largest L where 1..L all covered
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    all_primes = _primes_upto(1000)
    
    # Strategy: Greedy construction + local search refinement
    # Start with a reasonable set of primes and residues, then improve.
    
    # Initial heuristic: use primes from some range, assign residues to cover early integers
    # Key idea: for each prime p, choosing residue r covers r, r+p, r+2p, ...
    # To cover 1..L, we need every integer in 1..L hit by at least one class.
    
    # Phase 1: Build initial solution using greedy approach
    # Use primes >= 13 (smaller primes have large 1/p cost)
    # Try to maximize coverage
    
    best_classes = []
    best_L = 0
    
    # Try different starting prime ranges
    for start_idx in range(3, len(all_primes)):  # start from prime index 3 (p=7) onwards
        if time.time() > deadline:
            break
        p_start = all_primes[start_idx]
        if p_start > 1000:
            break
        
        # Select primes from p_start to 1000
        selected_primes = [p for p in all_primes[start_idx:] if p <= 1000]
        
        # Check if sum of reciprocals <= 1
        cost = sum(Fraction(1, p) for p in selected_primes)
        if cost > Fraction(1, 1):
            continue
        
        # Assign residues greedily: for each prime, pick residue that covers most uncovered early integers
        classes = []
        covered = [False] * 10001  # track up to 10000
        for p in selected_primes:
            best_r = 0
            best_gain = -1
            for r in range(p):
                # Count how many new integers in 1..10000 this would cover
                gain = 0
                n = r if r > 0 else p
                while n <= 10000:
                    if not covered[n]:
                        gain += 1
                    n += p
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([p, best_r])
            # Update covered
            n = best_r if best_r > 0 else p
            while n <= 10000:
                covered[n] = True
                n += p
        
        # Compute actual coverage length
        L = _coverage_length(classes, 10000)
        if L > best_L:
            best_L = L
            best_classes = classes[:]
    
    # Phase 2: Local search - try swapping residues to improve coverage
    if best_classes and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_classes)):
                p, r = best_classes[i]
                # Try all other residues for this prime
                for r_new in range(p):
                    if r_new == r:
                        continue
                    # Temporarily change and measure
                    old_r = best_classes[i][1]
                    best_classes[i][1] = r_new
                    L_new = _coverage_length(best_classes, 10000)
                    if L_new > best_L:
                        best_L = L_new
                        improved = True
                        break
                    else:
                        best_classes[i][1] = old_r
                if improved:
                    break
    
    # Phase 3: Try adding/removing primes
    if time.time() < deadline:
        current_cost = sum(Fraction(1, p) for p, r in best_classes)
        remaining = Fraction(1, 1) - current_cost
        
        # Try adding primes that fit in remaining budget
        for p in all_primes:
            if time.time() > deadline:
                break
            if any(bp == p for bp, br in best_classes):
                continue
            if Fraction(1, p) <= remaining:
                # Try adding this prime with best residue
                new_classes = best_classes + [[p, 0]]
                # Find best residue for new prime
                covered = [False] * 10001
                for pp, rr in best_classes:
                    n = rr if rr > 0 else pp
                    while n <= 10000:
                        covered[n] = True
                        n += pp
                best_r = 0
                best_gain = 0
                for r in range(p):
                    gain = 0
                    n = r if r > 0 else p
                    while n <= 10000:
                        if not covered[n]:
                            gain += 1
                        n += p
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                new_classes[-1] = [p, best_r]
                L_new = _coverage_length(new_classes, 10000)
                if L_new > best_L:
                    best_L = L_new
                    best_classes = new_classes
                    remaining = Fraction(1, 1) - sum(Fraction(1, pp) for pp, rr in best_classes)
    
    # Final verification
    final_cost = sum(Fraction(1, p) for p, r in best_classes)
    if final_cost > Fraction(1, 1):
        # Fallback to seed-like solution
        primes = [p for p in all_primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]
    
    return best_classes
# EVOLVE-BLOCK-END
