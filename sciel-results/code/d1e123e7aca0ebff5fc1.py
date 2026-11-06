import math
import random
import time

def construct():
    start_time = time.time()
    timeout = 85  # Leave 5s margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds
    gcd_cache = {}
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            gcd_cache[(i, j)] = math.gcd(i, j)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        if m1 > m2: m1, m2 = m2, m1
        return gcd_cache[(m1, m2)]

    def is_valid_assignment(selected_moduli, residues):
        """Check if the current assignment is valid."""
        for i in range(len(selected_moduli)):
            m1 = selected_moduli[i]
            r1 = residues[m1]
            for j in range(i + 1, len(selected_moduli)):
                m2 = selected_moduli[j]
                r2 = residues[m2]
                g = get_gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    def try_add_modulus(selected_moduli, residues, m):
        """Try to find a residue for m that is valid against all selected_moduli."""
        # Collect constraints
        constraints = []
        for m_prime in selected_moduli:
            g = get_gcd(m, m_prime)
            forbidden = residues[m_prime] % g
            constraints.append((g, forbidden))
        
        # Try all residues 0..m-1
        for r in range(m):
            valid = True
            for g, forbidden in constraints:
                if r % g == forbidden:
                    valid = False
                    break
            if valid:
                residues[m] = r
                return True
        return False

    def try_add_modulus_csp(selected_moduli, residues, m):
        """CSP approach with arc consistency for adding a new modulus."""
        domain = set(range(m))
        
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
            
        # Choose a residue that is "least constraining" for future additions?
        # For now, pick randomly or the first one.
        # Picking randomly helps with diversity in restarts.
        r = random.choice(list(domain))
        residues[m] = r
        return True

    best_count = 0
    best_solution = []
    
    # Heuristic: Moduli with small values or specific factors might be easier?
    # Let's try different orderings.
    
    def run_search(order, use_csp):
        selected = []
        residues = {}
        
        for m in order:
            if time.time() - start_time > timeout:
                break
                
            if use_csp:
                if try_add_modulus_csp(selected, residues, m):
                    selected.append(m)
            else:
                if try_add_modulus(selected, residues, m):
                    selected.append(m)
                    
        return selected, residues

    # Strategy 1: Greedy with random restarts and different orderings
    # We want to maximize the count.
    
    # Precompute a few static orderings
    order_asc = moduli[:]
    order_desc = moduli[::-1]
    order_random = moduli[:]
    random.shuffle(order_random)
    
    # Sort by number of divisors?
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    div_counts = {m: count_divisors(m) for m in moduli}
    order_divs = sorted(moduli, key=lambda m: div_counts[m])
    
    # Run multiple iterations
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        # Cycle through strategies
        strategy = iterations % 4
        if strategy == 0:
            order = order_asc
            use_csp = False
        elif strategy == 1:
            order = order_desc
            use_csp = False
        elif strategy == 2:
            order = order_divs
            use_csp = True
        else:
            order = order_random
            random.shuffle(order)
            use_csp = True
            
        selected, residues = run_search(order, use_csp)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_solution = [[residues[m], m] for m in selected]
            
        # Local search: try to swap residues to fix conflicts? 
        # Or try to add more moduli to the current best?
        
        if best_count > 10: # If we have a good solution, try to polish
             # Try to add any remaining modulus to the best solution
             current_mods = set(selected)
             for m in moduli:
                 if m in current_mods:
                     continue
                 if time.time() - start_time > timeout:
                     break
                 # Try to add m to the current best
                 temp_residues = dict(residues)
                 if try_add_modulus_csp(selected, temp_residues, m):
                     selected.append(m)
                     residues = temp_residues
                     if len(selected) > best_count:
                         best_count = len(selected)
                         best_solution = [[residues[m2], m2] for m2 in selected]

    if not best_solution:
        # Fallback: A known small solution
        # {6, 10, 15, 30}
        return [[0, 6], [1, 10], [2, 15], [3, 30]]
        
    return best_solution
