import math
import time
import random

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Strategy: Use all even moduli (2,4,6,...,128) = 64 classes
    # For each pair (m1,m2) both even, gcd >= 2
    # We need r1 NOT congruent to r2 mod gcd(m1,m2)
    
    # Key insight: If we set residue r for modulus m, the constraint on another modulus m'
    # is that r' must not be congruent to r mod gcd(m,m')
    
    # Better approach: Process moduli in order of increasing number of prime factors
    # or use a greedy with backtracking that's more efficient
    
    moduli = list(range(2, 129, 2))  # 64 even moduli
    n = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
    
    # Try multiple strategies with different orderings
    
    def try_greedy(order, seed_offset=0):
        """Greedy with backtracking on given order"""
        if time.time() > deadline:
            return None
        
        residues = [None] * n
        assigned = [False] * n
        
        # For each variable, track which residues are forbidden by each assigned variable
        # forbidden[i] = set of residues forbidden for variable i
        forbidden = [set() for _ in range(n)]
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            i = order[idx]
            m_i = moduli[i]
            
            # Find valid residues
            valid_res = []
            for r in range(m_i):
                if r in forbidden[i]:
                    continue
                valid = True
                for j in range(n):
                    if assigned[j]:
                        if not _is_disjoint(r, m_i, residues[j], moduli[j]):
                            valid = False
                            break
                if valid:
                    valid_res.append(r)
            
            if not valid_res:
                return False
            
            # Heuristic: try residues that are "central" or random
            random.seed(seed_offset + idx)
            random.shuffle(valid_res)
            
            for r in valid_res:
                residues[i] = r
                assigned[i] = True
                
                # Update forbidden sets for unassigned variables
                new_forbidden = []
                ok = True
                for j in range(n):
                    if not assigned[j] and j != i:
                        new_forbidden.append(j)
                
                # Check if any unassigned variable has no valid residue
                for j in new_forbidden:
                    m_j = moduli[j]
                    has_valid = False
                    for rj in range(m_j):
                        if rj in forbidden[j]:
                            continue
                        # Check against all assigned
                        valid_j = True
                        for k in range(n):
                            if assigned[k] and k != j:
                                if not _is_disjoint(rj, m_j, residues[k], moduli[k]):
                                    valid_j = False
                                    break
                        if valid_j:
                            has_valid = True
                            break
                    if not has_valid:
                        ok = False
                        break
                
                if ok:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                assigned[i] = False
                residues[i] = None
            
            return False
        
        if backtrack(0):
            return residues
        return None
    
    # Try different orderings
    # Order 1: increasing modulus
    order1 = list(range(n))
    order1.sort(key=lambda i: moduli[i])
    
    # Order 2: decreasing modulus
    order2 = list(range(n))
    order2.sort(key=lambda i: -moduli[i])
    
    # Order 3: by number of divisors (more constrained first)
    def num_divisors(x):
        cnt = 0
        for d in range(1, int(x**0.5)+1):
            if x % d == 0:
                cnt += 1
                if d != x//d:
                    cnt += 1
        return cnt
    
    order3 = list(range(n))
    order3.sort(key=lambda i: (-num_divisors(moduli[i]), moduli[i]))
    
    # Try each ordering with multiple seeds
    best_result = None
    best_len = 0
    
    for order in [order1, order2, order3]:
        for seed in range(50):
            if time.time() > deadline:
                break
            result = try_greedy(order, seed)
            if result is not None:
                if len(result) > best_len:
                    best_len = len(result)
                    best_result = [[result[i], moduli[i]] for i in range(n)]
                if best_len == n:
                    return best_result
        if time.time() > deadline:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: try a simpler approach - use only moduli that are powers of 2
    # 2,4,8,16,32,64,128 - but we need distinct moduli and they must be pairwise disjoint
    # gcd(2^a, 2^b) = 2^min(a,b), so we need r_a NOT congruent to r_b mod 2^min(a,b)
    
    # Even simpler fallback: just return a small valid set
    return [[0, 2], [1, 4]]
