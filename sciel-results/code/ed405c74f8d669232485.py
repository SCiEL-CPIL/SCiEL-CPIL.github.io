import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate primes up to 200
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    # Factorization cache
    def get_prime_factors(n):
        factors = set()
        d = 2
        x = n
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors

    # Precompute factors for all m
    factors_map = {m: get_prime_factors(m) for m in range(20, 201)}
    
    # Greedy selection based on prime count constraints
    selected = []
    counts = {p: 0 for p in primes}
    
    # Sort moduli by value (ascending) to maximize reciprocal sum
    for m in range(20, 201):
        if time.time() > deadline:
            break
        # Check if adding m violates any prime count constraint
        can_add = True
        for p in factors_map[m]:
            if counts[p] >= p:
                can_add = False
                break
        
        if can_add:
            selected.append(m)
            for p in factors_map[m]:
                counts[p] += 1

    # Now assign residues
    # We need r_m for each m in selected such that r_i != r_j mod gcd(m_i, m_j)
    # Sort selected by m to assign in order
    # Or sort by "constrainedness" (number of conflicts)
    
    # Simple greedy assignment
    residues = {}
    # Order: maybe by m, or by number of prime factors
    # Let's try assigning in order of m
    for m in sorted(selected):
        found = False
        for r in range(m):
            valid = True
            for m_j in selected:
                if m_j == m:
                    continue
                if m_j not in residues:
                    continue
                r_j = residues[m_j]
                g = math.gcd(m, m_j)
                if r % g == r_j % g:
                    valid = False
                    break
            if valid:
                residues[m] = r
                found = True
                break
        
        if not found:
            # This shouldn't happen if constraints are sufficient, but if it does, we have a problem.
            # We might need to backtrack or remove m.
            # For now, we'll just leave it out or try a random residue?
            # If we can't find a residue, the set is invalid.
            # We should have pruned more.
            # Let's try to remove the last added element and retry?
            # Given the time, we'll just skip this m if it fails, reducing score.
            pass
            
    # If some residues were not found, we need to handle it.
    # But the prime count constraint $k_p \le p$ is necessary.
    # Is it sufficient?
    # For $p=2$, $k_2 \le 2$.
    # If we have 2 evens, $r_1 \not\equiv r_2 \pmod 2$.
    # We can pick $r_1=0, r_2=1$.
    # If we have 3 odds, $k_2$ doesn't apply.
    # It seems sufficient for small sets.
    
    # Construct output
    result = []
    for m in selected:
        if m in residues:
            result.append([m, residues[m]])
            
    return result
