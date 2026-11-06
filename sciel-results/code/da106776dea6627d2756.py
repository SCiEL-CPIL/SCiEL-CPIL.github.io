import random
import time
import math

def construct():
    start_time = time.time()
    timeout = 85  # Leave 5s margin for safety
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_count = 0
    best_solution = []
    
    # Precompute gcds for speed
    gcd_cache = {}
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            gcd_cache[(i, j)] = math.gcd(i, j)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        if m1 > m2: m1, m2 = m2, m1
        return gcd_cache[(m1, m2)]

    def try_add(selected_moduli, residues, m):
        # Check against existing moduli
        # We need r such that for all m' in selected: r % gcd(m, m') != residues[m'] % gcd(m, m')
        # We can try residues 0..m-1
        # To speed up, we can check constraints one by one and prune?
        # Simple loop is probably fine for m <= 128
        
        # Randomize start point
        start = random.randint(0, m - 1)
        for k in range(m):
            r = (start + k) % m
            valid = True
            for m_prime in selected_moduli:
                g = get_gcd(m, m_prime)
                if r % g == residues[m_prime] % g:
                    valid = False
                    break
            if valid:
                residues[m] = r
                return True
        return False

    def try_add_csp(selected_moduli, residues, m, domains):
        # CSP approach: use arc consistency to prune domains
        # For each existing modulus m', we have a constraint on r mod gcd(m, m')
        # The constraint is: r mod gcd(m, m') != residues[m'] mod gcd(m, m')
        
        # Start with full domain for m
        domain = set(range(m))
        
        # Apply arc consistency: for each existing m', remove values that violate constraint
        for m_prime in selected_moduli:
            g = get_gcd(m, m_prime)
            forbidden = residues[m_prime] % g
            new_domain = set()
            for r in domain:
                if r % g != forbidden:
                    new_domain.add(r)
            domain = new_domain
            if not domain:
                return False
        
        if not domain:
            return False
            
        # Choose a value from the remaining domain
        # Prefer values that leave more flexibility for future additions
        # Simple heuristic: choose randomly from remaining
        r = random.choice(list(domain))
        residues[m] = r
        return True

    # Heuristic: Sort moduli by "difficulty"
    # Moduli with many divisors are harder? Or prime powers?
    # Let's try different strategies:
    # 1. Random order
    # 2. Sort by number of divisors (ascending - easier first)
    # 3. Sort by value (ascending)
    
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    # Precompute divisor counts
    div_counts = {m: count_divisors(m) for m in moduli}
    
    for restart in range(2000):
        if time.time() - start_time > timeout:
            break
            
        selected = []
        residues = {}
        
        # Choose ordering strategy based on restart
        strategy = restart % 3
        if strategy == 0:
            available = moduli[:]
            random.shuffle(available)
        elif strategy == 1:
            # Sort by number of divisors (ascending)
            available = sorted(moduli, key=lambda m: div_counts[m])
        else:
            # Sort by value (ascending)
            available = moduli[:]
        
        for m in available:
            if time.time() - start_time > timeout:
                break
            
            # Use CSP approach for better pruning
            if try_add_csp(selected, residues, m, None):
                selected.append(m)
                if len(selected) > best_count:
                    best_count = len(selected)
                    # Store copy
                    best_solution = [[residues[m2], m2] for m2 in selected]
        
        # Optional: Local search to improve current best
        # Try to swap residues of existing to allow adding more?
        # Too complex for now.
        
    if not best_solution:
        return [[(1<<k)-1,1<<(k+1)] for k in range(7)]
        
    return best_solution
