"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    if n == 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    N = 512
    
    # Precompute prime factors for all numbers 1..511
    pf = [set() for _ in range(N)]
    for a in range(1, N):
        pf[a] = _prime_factors(a)
    
    # Precompute contribution values as Fractions for exact comparison
    contrib = {a: Fraction(1, N - a) for a in range(1, N)}
    
    # Precompute compatibility matrix
    # compat[i][j] is True if gcd(i, j) == 1
    # We can use a list of sets of compatible numbers for each i
    compatible_with = [set() for _ in range(N)]
    for i in range(1, N):
        for j in range(i, N):
            if i == j:
                continue
            if gcd(i, j) == 1:
                compatible_with[i].add(j)
                compatible_with[j].add(i)
    
    # Greedy initialization: pick numbers in order of highest contribution (largest a)
    # that don't conflict with already chosen numbers.
    chosen = set()
    # Sort by contribution descending (largest a first)
    candidates = sorted(range(1, N), key=lambda x: -contrib[x])
    
    for a in candidates:
        # Check if a is compatible with all in chosen
        is_compatible = True
        for b in chosen:
            if a not in compatible_with[b]:
                is_compatible = False
                break
        if is_compatible:
            chosen.add(a)
            
    best_score = sum(contrib[a] for a in chosen)
    best_chosen = set(chosen)
    unchosen = set(range(1, N)) - chosen
    
    def is_compatible_with_set(a, s):
        for b in s:
            if a not in compatible_with[b]:
                return False
        return True

    # Local search
    # We will try to improve the score by:
    # 1. Adding any unchosen number if it fits.
    # 2. Swapping one chosen number with one unchosen number if it improves score.
    # 3. Swapping one chosen number with two unchosen numbers if it improves score.
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # 1. Try adding
        for b in sorted(unchosen, key=lambda x: -contrib[x]):
            if is_compatible_with_set(b, chosen):
                new_score = best_score + contrib[b]
                if new_score > best_score:
                    chosen.add(b)
                    unchosen.remove(b)
                    best_score = new_score
                    best_chosen = set(chosen)
                    improved = True
                    # Restart loop to try adding more
                    break
        
        if improved:
            continue

        # 2. Try 1-for-1 swaps
        chosen_list = list(chosen)
        for a in chosen_list:
            if time.time() > DEADLINE:
                return list(best_chosen)
            
            # Remove a temporarily
            chosen.remove(a)
            unchosen.add(a)
            
            # Try adding any b from unchosen (excluding a)
            best_b = None
            best_gain = Fraction(0, 1)
            for b in unchosen:
                if b == a:
                    continue
                if is_compatible_with_set(b, chosen):
                    gain = contrib[b] - contrib[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            
            if best_b is not None:
                chosen.add(best_b)
                unchosen.remove(best_b)
                best_score += best_gain
                best_chosen = set(chosen)
                improved = True
                # Restart loop
                break
            else:
                # Restore a
                chosen.add(a)
                unchosen.remove(a)
        
        if improved:
            continue

        # 3. Try 1-for-2 swaps
        # Remove a, add b and c
        # This is more expensive. We only do this if we have time.
        if time.time() < DEADLINE - 10:
            chosen_list = list(chosen)
            for a in chosen_list:
                if time.time() > DEADLINE:
                    return list(best_chosen)
                
                chosen.remove(a)
                unchosen.add(a)
                
                # Find best pair (b, c) in unchosen (excluding a)
                # b and c must be compatible with each other and with chosen
                best_pair_gain = Fraction(0, 1)
                best_pair = None
                
                # To make this faster, sort unchosen by contrib
                unchosen_sorted = sorted(unchosen, key=lambda x: -contrib[x])
                # Limit search to top K candidates for b and c to keep time down
                # If K is too small, we might miss good pairs. 
                # Let's try top 50.
                top_unchosen = [x for x in unchosen_sorted if x != a][:50]
                
                for i in range(len(top_unchosen)):
                    b = top_unchosen[i]
                    if not is_compatible_with_set(b, chosen):
                        continue
                    # Check c
                    for j in range(i + 1, len(top_unchosen)):
                        c = top_unchosen[j]
                        if c == b:
                            continue
                        if b not in compatible_with[c]:
                            continue
                        if not is_compatible_with_set(c, chosen):
                            continue
                        
                        gain = contrib[b] + contrib[c] - contrib[a]
                        if gain > best_pair_gain:
                            best_pair_gain = gain
                            best_pair = (b, c)
                
                if best_pair is not None:
                    b, c = best_pair
                    chosen.add(b)
                    chosen.add(c)
                    unchosen.remove(b)
                    unchosen.remove(c)
                    best_score += best_pair_gain
                    best_chosen = set(chosen)
                    improved = True
                    break
                else:
                    chosen.add(a)
                    unchosen.remove(a)
        
        if improved:
            continue
            
    return list(best_chosen)

# EVOLVE-BLOCK-END
