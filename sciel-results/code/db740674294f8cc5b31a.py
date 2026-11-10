"""Combined approach: Prime-based initialization + Macro-Move Local Search."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    """Return list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    N = 512
    
    # Precompute values and prime factors for all numbers 1..511
    values = [0.0] * N
    prime_factors = [[] for _ in range(N)]
    
    for a in range(1, N):
        values[a] = 1.0 / (N - a)
        prime_factors[a] = _get_prime_factors(a)
        
    # Identify primes
    primes = [a for a in range(1, N) if _is_prime(a)]
    
    # Initialize with all primes
    selected = set(primes)
    current_score = sum(values[a] for a in selected)
    
    # Precompute a set of selected primes for quick lookup?
    # Actually, we just need to check if a prime factor is in `selected`.
    # Since we start with primes, `selected` contains only primes initially.
    # As we add composites, `selected` will contain composites.
    # The conflict check: 
    # For a candidate composite c, it conflicts with any b in selected if gcd(b, c) > 1.
    # Since b might be prime or composite, we can't just check prime factors of c against selected primes.
    # We must check gcd(b, c) for all b in selected.
    
    # Optimization: 
    # If we maintain a set of "used prime factors" UPF, then a candidate c is valid 
    # only if no prime factor of c is in UPF? 
    # No, that's for checking if c can be ADDED without removing anything.
    # For the macro-move, we remove conflicts.
    # Conflict set S_conflict = { b in selected | gcd(b, c) > 1 }.
    
    # To speed up conflict detection:
    # If b is prime, gcd(b, c) > 1 iff b is in prime_factors[c].
    # If b is composite, it's harder.
    # However, note that we start with primes. 
    # When we swap a set of primes P for a composite C, 
    # the new selected set will contain C and the remaining primes.
    # Will we ever have two composites in selected? 
    # Yes, if we do 2-for-k moves or if a composite doesn't conflict with another composite.
    # But initially, we only have primes.
    # Let's assume we only do 1-for-k moves first.
    # Then selected = (Primes \ P) U {C}.
    # The elements in (Primes \ P) are primes. C is composite.
    # So selected contains at most one composite at any step if we only do 1-for-k?
    # No, if we add C1, then later add C2, we might have two composites.
    # But C1 and C2 must be coprime.
    
    # Let's implement the generic 1-for-k move search.
    
    best_gain = 0
    best_move = None
    
    # We will iterate until no improvement is found
    improved = True
    while improved:
        if time.time() > deadline:
            break
            
        improved = False
        best_gain = 0
        best_move = None
        
        # Candidates are all numbers NOT in selected
        # We can restrict to composites, as adding a prime that is already "blocked" 
        # by a composite is unlikely to be beneficial compared to the composite's value,
        # but theoretically possible. 
        # However, primes have small values (1/(512-p) is small for small p, large for large p).
        # Large primes are likely already in selected (since we started with all).
        # So candidates are mostly composites.
        
        for c in range(1, N):
            if c in selected:
                continue
                
            # Find conflicts
            conflicts = []
            for b in selected:
                if gcd(b, c) > 1:
                    conflicts.append(b)
            
            if not conflicts:
                # This means c is coprime to all selected. 
                # We can just add it. Gain = value[c].
                # But wait, if c is coprime to all selected, why wasn't it added earlier?
                # Maybe it was blocked by a prime that we removed?
                # Or maybe we just haven't tried it yet.
                # In our initialization, we took ALL primes. 
                # Any composite c has prime factors. 
                # If all its prime factors were in selected (as primes), it would have conflicts.
                # So conflicts should never be empty for a composite c if selected contains all its prime factors.
                # If we removed some primes, conflicts might be empty.
                pass
            
            # Calculate gain
            conflict_cost = sum(values[b] for b in conflicts)
            gain = values[c] - conflict_cost
            
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_move = (c, conflicts)
        
        if best_move is not None:
            c, conflicts = best_move
            # Apply move
            for b in conflicts:
                selected.remove(b)
            selected.add(c)
            current_score += best_gain
            improved = True
    
    # Optional: Try 2-for-k moves if time permits
    # This is more complex. Given the strong effect of 1-for-k, 
    # and the 150s limit, 1-for-k might be sufficient to find a very good solution.
    # Let's add a simple check for 2 composites if time remains.
    
    if time.time() < deadline - 10:
        # Simple 2-for-k search
        # We look for pairs (c1, c2) not in selected, coprime to each other.
        # Conflicts = { b in selected | gcd(b, c1)>1 or gcd(b, c2)>1 }
        # Gain = v[c1] + v[c2] - sum(v[b] for b in Conflicts)
        
        # To make this fast, we only consider "high value" composites.
        # Top 50 composites by value.
        composites = [a for a in range(1, N) if not _is_prime(a) and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:50]
        
        for i in range(len(top_composites)):
            c1 = top_composites[i]
            if c1 in selected:
                continue
            for j in range(i + 1, len(top_composites)):
                c2 = top_composites[j]
                if c2 in selected:
                    continue
                if gcd(c1, c2) != 1:
                    continue
                
                conflicts = []
                for b in selected:
                    if gcd(b, c1) > 1 or gcd(b, c2) > 1:
                        conflicts.append(b)
                
                conflict_cost = sum(values[b] for b in conflicts)
                gain = values[c1] + values[c2] - conflict_cost
                
                if gain > 1e-12:
                    # Apply
                    for b in conflicts:
                        selected.remove(b)
                    selected.add(c1)
                    selected.add(c2)
                    current_score += gain
                    # Restart 2-for-k? Or just break and go back to 1-for-k?
                    # Let's break and let the outer loop handle it if we structure it that way.
                    # For simplicity, we just do one pass of 2-for-k.
                    break
            if time.time() > deadline - 5:
                break

    # Final Verification
    result = list(selected)
    if len(result) != len(set(result)):
        result = list(set(result))
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")
    
    return result
# EVOLVE-BLOCK-END
