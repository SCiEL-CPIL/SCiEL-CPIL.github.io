"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 85
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    def prime_factors(n):
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
    
    pf = {m: prime_factors(m) for m in moduli}
    all_primes = sorted(set(p for m in moduli for p in pf[m]))
    
    # Precompute gcd table
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            if m1 <= m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def get_gcd(m1, m2):
        if m1 > m2:
            m1, m2 = m2, m1
        return gcd_table[(m1, m2)]
    
    def count_prime_usage(selected):
        pc = defaultdict(int)
        for m in selected:
            for p in pf[m]:
                pc[p] += 1
        return pc
    
    def is_valid_selection(selected):
        pc = count_prime_usage(selected)
        for p, c in pc.items():
            if c > p:
                return False
        return True
    
    def greedy_select(order):
        selected = []
        pc = defaultdict(int)
        for m in order:
            can_add = True
            for p in pf[m]:
                if pc[p] >= p:
                    can_add = False
                    break
            if can_add:
                selected.append(m)
                for p in pf[m]:
                    pc[p] += 1
        return selected
    
    # Multiple greedy orderings
    orderings = [
        sorted(moduli, key=lambda m: (len(pf[m]), m)),
        sorted(moduli, key=lambda m: (len(pf[m]), -m)),
        sorted(moduli, key=lambda m: m),
        sorted(moduli, key=lambda m: -m),
        sorted(moduli, key=lambda m: (max(pf[m], default=1), m)),
        sorted(moduli, key=lambda m: (sum(1 for p in pf[m] if p <= 7), m)),
    ]
    
    best_set = []
    for order in orderings:
        s = greedy_select(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search: try additions and swaps
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        current = list(best_set)
        pc = count_prime_usage(current)
        current_set = set(current)
        
        # Try to add any modulus
        for m in moduli:
            if m in current_set:
                continue
            can_add = all(pc[p] < p for p in pf[m])
            if can_add:
                current.append(m)
                current_set.add(m)
                for p in pf[m]:
                    pc[p] += 1
                improved = True
        
        # Try 1-for-1 swaps
        for m_in in list(current):
            for m_out in moduli:
                if m_out in current_set:
                    continue
                # Check if removing m_in and adding m_out is valid
                new_pc = dict(pc)
                for p in pf[m_in]:
                    new_pc[p] -= 1
                    if new_pc[p] == 0:
                        del new_pc[p]
                valid = True
                for p in pf[m_out]:
                    if new_pc.get(p, 0) >= p:
                        valid = False
                        break
                if valid:
                    # Only swap if it helps (same size but maybe better structure)
                    # For now, only do swaps that maintain or improve count
                    # Actually, 1-for-1 keeps same count, so skip unless we can then add more
                    pass
        
        best_set = current
    
    # AC-3 + Backtracking for residue assignment
    def assign_residues_csp(selected):
        if not selected:
            return {}
        
        n = len(selected)
        # Variables: indices 0..n-1 corresponding to selected[i]
        # Domains: range(selected[i]) for each i
        
        domains = [list(range(selected[i])) for i in range(n)]
        
        # Build constraint graph: for each pair (i,j) with gcd > 1, constraint is
        # domains[i][r] % g != domains[j][s] % g where g = gcd(selected[i], selected[j])
        
        # Precompute constraints
        constraints = {}  # (i,j) -> gcd value (only if gcd > 1)
        adj = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                g = get_gcd(selected[i], selected[j])
                if g > 1:
                    constraints[(i, j)] = g
                    constraints[(j, i)] = g
                    adj[i].append(j)
                    adj[j].append(i)
        
        # AC-3
        def ac3():
            queue = deque()
            for i in range(n):
                for j in adj[i]:
                    queue.append((i, j))
            
            while queue:
                i, j = queue.popleft()
                g = constraints[(i, j)]
                # For each value in domain[i], check if there exists a value in domain[j]
                # such that val_i % g != val_j % g
                new_domain_i = []
                for vi in domains[i]:
                    vi_mod = vi % g
                    # Check if there exists vj in domains[j] with vj % g != vi_mod
                    found = False
                    for vj in domains[j]:
                        if vj % g != vi_mod:
                            found = True
                            break
                    if found:
                        new_domain_i.append(vi)
                if len(new_domain_i) < len(domains[i]):
                    domains[i] = new_domain_i
                    if not domains[i]:
                        return False
                    # Requeue all neighbors of i except j
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
            return True
        
        if not ac3():
            return None
        
        # Backtracking with MRV
        assignment = {}
        
        def backtrack():
            if len(assignment) == n:
                return True
            
            # MRV: pick variable with smallest domain
            unassigned = [i for i in range(n) if i not in assignment]
            if not unassigned:
                return True
            mrv_var = min(unassigned, key=lambda i: len(domains[i]))
            
            for val in domains[mrv_var]:
                # Check consistency with assigned neighbors
                ok = True
                for j in adj[mrv_var]:
                    if j in assignment:
                        g = constraints[(mrv_var, j)]
                        if val % g == assignment[j] % g:
                            ok = False
                            break
                if ok:
                    assignment[mrv_var] = val
                    if backtrack():
                        return True
                    del assignment[mrv_var]
            
            return False
        
        if backtrack():
            return {selected[i]: assignment[i] for i in range(n)}
        return None
    
    # Try to assign residues
    result = assign_residues_csp(best_set)
    
    if result is None:
        # Fallback: try smaller subsets
        for size in range(len(best_set) - 1, 0, -1):
            subset = best_set[:size]
            result = assign_residues_csp(subset)
            if result is not None:
                break
    
    if result is None:
        # Ultimate fallback
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return [[result[m], m] for m in best_set if m in result]
# EVOLVE-BLOCK-END
