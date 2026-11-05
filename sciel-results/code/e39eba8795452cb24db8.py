import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Candidate sets of moduli
    # 1. All even numbers
    even_moduli = [m for m in moduli if m % 2 == 0]
    # 2. All multiples of 3
    mult3_moduli = [m for m in moduli if m % 3 == 0]
    # 3. All multiples of 4
    mult4_moduli = [m for m in moduli if m % 4 == 0]
    
    # We will try to solve the residue assignment for these sets.
    # If a set fails, we try smaller subsets or other sets.
    
    def solve_residues(modulus_set):
        """
        Given a list of moduli (values), find residues such that
        for all i != j, (r_i - r_j) % gcd(m_i, m_j) != 0.
        Returns list of (modulus, residue) or None if no solution.
        """
        # Map modulus value to index in the set for gcd lookup
        # We need gcd between any two moduli in the set.
        k = len(modulus_set)
        if k == 0:
            return []
        
        # Precompute gcds for this specific set
        # idx in modulus_set corresponds to position
        set_gcd = [[0]*k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                set_gcd[i][j] = math.gcd(modulus_set[i], modulus_set[j])
        
        # Domains: list of sets of valid residues
        domains = [set(range(modulus_set[i])) for i in range(k)]
        
        # Order variables by MRV (initially, size of domain is the modulus value)
        # We will use a recursive backtracking with forward checking.
        
        def backtrack(assigned, domains):
            if time.time() > deadline - 2:
                return None
            
            # Find unassigned variable with smallest domain
            min_idx = -1
            min_size = float('inf')
            for i in range(k):
                if i not in assigned:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        min_idx = i
                    if min_size == 0:
                        return None
            
            if min_idx == -1:
                # All assigned
                return {i: assigned[i] for i in range(k)}
            
            # Try each residue in the domain of min_idx
            # Order residues: maybe try those that cause least pruning?
            # For now, just iterate.
            for r in sorted(list(domains[min_idx])):
                new_assigned = assigned.copy()
                new_assigned[min_idx] = r
                
                # Forward check: prune domains of unassigned variables
                new_domains = [set(d) for d in domains]
                valid = True
                for j in range(k):
                    if j in new_assigned:
                        continue
                    g = set_gcd[min_idx][j]
                    # Remove residues r_j such that (r - r_j) % g == 0
                    # i.e., r_j % g == r % g
                    target_mod = r % g
                    to_remove = [x for x in new_domains[j] if x % g == target_mod]
                    if len(to_remove) == len(new_domains[j]):
                        valid = False
                        break
                    new_domains[j].difference_update(to_remove)
                
                if valid:
                    result = backtrack(new_assigned, new_domains)
                    if result is not None:
                        return result
            
            return None
        
        initial_assigned = {}
        result = backtrack(initial_assigned, domains)
        if result is None:
            return None
        
        # Convert result to list of (modulus, residue)
        return [(modulus_set[i], result[i]) for i in range(k)]

    best_solution = []
    best_len = 0
    
    # Try the candidate sets
    candidates = [even_moduli, mult3_moduli, mult4_moduli]
    
    # Also try to generate a maximal clique greedily
    # Start with 2, add 4, 6, etc.
    # Actually, even_moduli is already a clique in the non-coprime graph.
    # Let's try to see if we can add any odds to even_moduli? No, gcd(odd, even) can be 1.
    # But if we remove 2, we can add 3?
    # Let's try a few random subsets of even_moduli + some odds?
    # Given time, stick to structured sets.
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        sol = solve_residues(cand)
        if sol and len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)
            
            # If we found a large solution, we might try to improve it?
            # For now, just record.
            
    # If no solution found for large sets, try smaller ones?
    # The even set should have a solution.
    
    # If best_len is still 0, fallback to greedy single elements
    if best_len == 0:
        for m in moduli:
            best_solution.append([0, m])
            best_len = 1
            break # Just return one if all fail

    # Format output: list of [residue, modulus]
    return [[r, m] for (m, r) in best_solution]

