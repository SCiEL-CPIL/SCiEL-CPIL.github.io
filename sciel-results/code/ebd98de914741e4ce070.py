"""Structural change: greedy set-cover over prime residue classes with budget 1."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _coverage_length(used, L):
    """Return the length of the initial segment 1..L that is covered."""
    # used: dict mapping prime -> residue
    # We check 1, 2, 3, ... until we find a gap
    for n in range(1, L + 1):
        covered = False
        for p, r in used.items():
            if n % p == r:
                covered = True
                break
        if not covered:
            return n - 1
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    primes = _sieve(1000)
    L_max = 100000
    
    # We'll try to cover 1..L by selecting residue classes [p, r]
    # with sum(1/p) <= 1
    # 
    # Strategy: Greedy set cover. At each step, pick the class [p,r] that
    # covers the most currently-uncovered integers in 1..L_max,
    # subject to the budget constraint.
    #
    # To make this tractable, we limit to covering up to some L_target,
    # and use a greedy approach that re-evaluates coverage incrementally.
    
    # First, let's establish a baseline: the seed covers 1..~999 with primes 13+.
    # Let's try to do better with a smarter selection.
    
    # Key insight: small primes have low cost (1/p) and cover many residues.
    # But we need distinct primes. Let's use a priority-based greedy.
    
    # For a target L, the problem is: choose a subset of (p,r) pairs, one r per p,
    # such that every n in 1..L is hit, and sum 1/p <= 1.
    
    # Greedy approach for a given L:
    # 1. Start with empty set, uncovered = {1..L}
    # 2. Repeatedly add the (p,r) that maximizes |uncovered ∩ {n: n≡r mod p}|
    #    among all p not yet used and r in 0..p-1, subject to budget
    # 3. Stop when uncovered is empty or budget exceeded
    
    # This is expensive for large L. Let's use a different structural approach:
    # Use the Chinese Remainder Theorem insight and cover in blocks.
    
    # Alternative structural approach: 
    # The seed uses primes 13,17,19,... with r = i+1 mod p.
    # The sum of 1/p for primes 13..~1000 is about 0.92.
    # We can afford to swap some large primes for smaller ones.
    
    # Let me try: use small primes (2,3,5,7,11,...) with carefully chosen residues
    # to maximize coverage of 1..L.
    
    # For a given set of primes P, the best residue r_p for each p to cover
    # 1..L is the one that hits the most uncovered numbers. This is a set cover.
    
    # Let's do iterative improvement:
    # 1. Start with seed solution
    # 2. Try to swap each prime for a smaller unused prime with a better residue
    # 3. Try to change residues for existing primes to improve coverage
    
    # Actually, let me try a cleaner greedy from scratch:
    
    best_classes = []
    best_L = 0
    
    # Greedy set cover with budget
    # We'll work with a target range and see how far we can cover.
    
    # For efficiency, let's limit our search to L up to 5000 initially,
    # then see if we can push further.
    
    def greedy_cover(target_L, max_time):
        """Greedy set cover for 1..target_L with reciprocal budget 1."""
        t0 = time.time()
        used = {}  # prime -> residue
        cost = Fraction(0)
        
        # Uncovered set: we'll track which numbers are covered
        covered = [False] * (target_L + 1)
        num_covered = 0
        
        # Precompute for each prime p and residue r, which numbers in 1..target_L are hit
        # This is expensive, so we'll compute on the fly.
        
        # For each candidate (p, r), the number of newly covered elements is:
        # count of n in 1..target_L such that n % p == r and not covered[n]
        
        # To speed up, maintain for each prime p a count of uncovered n ≡ r (mod p)
        # Update when we add a class.
        
        # Let's use a different approach: iterate and pick best each time.
        
        primes_sorted = sorted(primes, key=lambda p: p)  # small first (cheaper)
        
        while True:
            if time.time() - t0 > max_time:
                break
            if num_covered >= target_L:
                break
            
            best_gain = -1
            best_p = None
            best_r = None
            
            for p in primes_sorted:
                if p in used:
                    continue
                # Check if adding p would exceed budget
                if cost + Fraction(1, p) > Fraction(1):
                    continue
                
                # Find residue r that maximizes coverage of uncovered numbers
                # For each r in 0..p-1, count uncovered n in 1..target_L with n%p==r
                # We can do this by iterating through uncovered numbers
                # But that's O(target_L * num_primes) which is too slow.
                
                # Alternative: for each p, compute the best r by checking
                # which residue class has the most uncovered numbers.
                # We can precompute this using a sieve-like approach.
                
                # For prime p, the numbers 1..target_L fall into residue classes 0..p-1.
                # We need to count how many uncovered numbers are in each class.
                # Let's maintain an array cnt_p of length p for each p we care about.
                
                # Actually, let's just compute it directly but efficiently.
                # For a given p, iterate r from 0 to p-1, and for each r,
                # the numbers are r, r+p, r+2p, ... up to target_L.
                # Count how many of these are not covered.
                
                # This is O(target_L/p) per r, so O(target_L) per p.
                # With ~168 primes, this is O(168 * target_L) per step.
                # With target_L=5000, that's ~840K per step. With ~50 steps, ~42M. Tight but ok.
                
                best_r_for_p = -1
                best_count_for_p = -1
                for r in range(p):
                    # Count uncovered n in {r, r+p, r+2p, ...} ∩ [1, target_L]
                    # Note: r=0 means multiples of p
                    count = 0
                    start = r
                    if start < 1:
                        start = r + p  # smallest positive n ≡ r (mod p)
                    # Actually if r == 0, numbers are p, 2p, 3p, ...
                    # if r > 0, numbers are r, r+p, r+2p, ...
                    if r == 0:
                        start = p
                    else:
                        start = r
                    if start > target_L:
                        continue
                    for n in range(start, target_L + 1, p):
                        if not covered[n]:
                            count += 1
                    if count > best_count_for_p:
                        best_count_for_p = count
                        best_r_for_p = r
                
                if best_count_for_p > best_gain:
                    best_gain = best_count_for_p
                    best_p = p
                    best_r = best_r_for_p
            
            if best_p is None or best_gain <= 0:
                break
            
            # Add (best_p, best_r)
            used[best_p] = best_r
            cost += Fraction(1, best_p)
            # Mark covered
            r = best_r
            p = best_p
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, target_L + 1, p):
                if not covered[n]:
                    covered[n] = True
                    num_covered += 1
        
        return used, cost
    
    # Try greedy cover for increasing targets
    # But greedy might not be optimal. Let's try target = 1000, 2000, 5000, etc.
    
    # Actually, let's just run greedy for target = 10000 and see how far it covers.
    # Then measure the actual coverage.
    
    used, cost = greedy_cover(10000, 30)
    
    if used:
        classes = [[p, r] for p, r in used.items()]
        # Measure actual coverage
        # Check up to L_max
        cov = [False] * (min(100000, 20000) + 1)
        for p, r in classes:
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, min(100000, 20000) + 1, p):
                cov[n] = True
        L = 0
        for n in range(1, len(cov)):
            if cov[n]:
                L = n
            else:
                break
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Also try the seed as fallback
    primes_seed = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
    # Check cost
    seed_cost = sum(Fraction(1, p) for p, _ in seed_classes)
    if seed_cost <= Fraction(1):
        # Measure seed coverage
        cov = [False] * (min(100000, 20000) + 1)
        for p, r in seed_classes:
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, min(100000, 20000) + 1, p):
                cov[n] = True
        L = 0
        for n in range(1, len(cov)):
            if cov[n]:
                L = n
            else:
                break
        if L > best_L:
            best_L = L
            best_classes = seed_classes
    
    # Try to improve by local search: swap primes, change residues
    if time.time() < deadline and best_classes:
        # Local search: try changing one residue at a time to extend coverage
        # First, compute current coverage
        cov = [False] * (min(100000, 50000) + 1)
        for p, r in best_classes:
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, min(100000, 50000) + 1, p):
                cov[n] = True
        # Find the gap
        gap_pos = None
        for n in range(1, len(cov)):
            if not cov[n]:
                gap_pos = n
                break
        
        if gap_pos is not None:
            # Try to fix the gap by changing a residue
            # For each used prime p, try all residues r and see if it covers gap_pos
            # and doesn't create new gaps before the current end
            for idx, (p, old_r) in enumerate(best_classes):
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    # Check if new_r covers gap_pos
                    if gap_pos % p != new_r:
                        continue
                    # Tentatively change
                    # Recompute coverage to check for regressions
                    # This is expensive, so let's be selective
                    # Just check if gap_pos is covered and no new gaps in 1..gap_pos
                    # Actually, changing residue might uncover some numbers
                    # Let's just try it and measure
                    best_classes[idx] = [p, new_r]
                    cov2 = [False] * (min(100000, 50000) + 1)
                    for pp, rr in best_classes:
                        if rr == 0:
                            start = pp
                        else:
                            start = rr
                        for nn in range(start, min(100000, 50000) + 1, pp):
                            cov2[nn] = True
                    L2 = 0
                    for nn in range(1, len(cov2)):
                        if cov2[nn]:
                            L2 = nn
                        else:
                            break
                    if L2 > best_L:
                        best_L = L2
                        gap_pos = None
                        break
                    else:
                        best_classes[idx] = [p, old_r]
                if gap_pos is None:
                    break
                if time.time() > deadline:
                    break
    
    if not best_classes:
        # Fallback to seed
        best_classes = seed_classes
    
    # Final validation
    # Check cost
    final_cost = sum(Fraction(1, p) for p, r in best_classes)
    if final_cost > Fraction(1):
        # Remove classes until cost is OK
        # Sort by 1/p descending (largest cost first) and remove
        best_classes.sort(key=lambda x: -1.0 / x[0])
        while final_cost > Fraction(1) and best_classes:
            p, r = best_classes.pop()
            final_cost -= Fraction(1, p)
    
    # Ensure distinct primes
    seen = set()
    filtered = []
    for p, r in best_classes:
        if p not in seen:
            seen.add(p)
            filtered.append([p, r])
    best_classes = filtered
    
    return best_classes

# EVOLVE-BLOCK-END
