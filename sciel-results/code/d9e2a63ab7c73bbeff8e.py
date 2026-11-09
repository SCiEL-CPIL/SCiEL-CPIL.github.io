"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Find the maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Two residue classes (a, m) and (b, n) are disjoint iff
    a mod gcd(m,n) != b mod gcd(m,n).
    
    Strategy:
    1. Identify candidate moduli sets where every pair has gcd >= 2
       (necessary condition for disjointness to be possible).
    2. For each candidate set, try to assign residues via greedy + local search.
    3. Maximize the number of classes.
    
    Key insight: All even moduli in [2,128] have pairwise gcd >= 2.
    There are 63 even numbers. We need to assign residues so that
    for every pair (m_i, m_j): r_i mod gcd(m_i,m_j) != r_j mod gcd(m_i,m_j).
    """
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    max_mod = 128
    available_moduli = list(range(2, max_mod + 1))
    
    def is_compatible(residue_new, modulus_new, assigned):
        """Check if (residue_new, modulus_new) is compatible with all assigned pairs."""
        for (r, m) in assigned:
            g = math.gcd(modulus_new, m)
            if residue_new % g == r % g:
                return False
        return True
    
    def greedy_assign(moduli_list, assigned, start_idx=0):
        """Greedy: try to add moduli from moduli_list[start_idx:] to assigned."""
        for i in range(start_idx, len(moduli_list)):
            if time.time() > deadline:
                break
            m = moduli_list[i]
            # Skip if already in assigned
            if any(am == m for (_, am) in assigned):
                continue
            found = False
            for r in range(m):
                if is_compatible(r, m, assigned):
                    assigned.append((r, m))
                    found = True
                    break
            if not found:
                continue
    
    def local_search(assigned, max_iters=50000):
        """Local search: try to improve by swapping residues or adding moduli."""
        assigned = assigned[:]
        used_moduli = set(m for (_, m) in assigned)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a new modulus
            for m in range(2, max_mod + 1):
                if m in used_moduli:
                    continue
                for r in range(m):
                    if is_compatible(r, m, assigned):
                        assigned.append((r, m))
                        used_moduli.add(m)
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try to swap residues to free up space
            # Pick a random pair and try swapping their residues
            if len(assigned) >= 2:
                i, j = random.sample(range(len(assigned)), 2)
                r_i, m_i = assigned[i]
                r_j, m_j = assigned[j]
                
                # Try swapping: assign r_j to m_i and r_i to m_j
                if r_j < m_i and r_i < m_j:
                    new_assigned = assigned[:]
                    new_assigned[i] = (r_j, m_i)
                    new_assigned[j] = (r_i, m_j)
                    
                    # Check if the new assignment is valid for these two
                    g = math.gcd(m_i, m_j)
                    if r_j % g != r_i % g:
                        # Check against all others
                        valid = True
                        for k in range(len(new_assigned)):
                            if k == i or k == j:
                                continue
                            r_k, m_k = new_assigned[k]
                            g_ik = math.gcd(m_i, m_k)
                            g_jk = math.gcd(m_j, m_k)
                            if r_j % g_ik == r_k % g_ik or r_i % g_jk == r_k % g_jk:
                                valid = False
                                break
                        if valid:
                            assigned = new_assigned
                            improved = True
            
            if not improved and iteration % 1000 == 0:
                # Try a more aggressive perturbation: randomly reassign a few residues
                for _ in range(3):
                    idx = random.randint(0, len(assigned) - 1)
                    r_old, m_old = assigned[idx]
                    # Try a random new residue
                    r_new = random.randint(0, m_old - 1)
                    if r_new == r_old:
                        continue
                    temp_assigned = assigned[:]
                    temp_assigned[idx] = (r_new, m_old)
                    valid = True
                    for k in range(len(temp_assigned)):
                        if k == idx:
                            continue
                        r_k, m_k = temp_assigned[k]
                        g = math.gcd(m_old, m_k)
                        if r_new % g == r_k % g:
                            valid = False
                            break
                    if valid:
                        assigned = temp_assigned
                        improved = True
                        break
        
        return assigned
    
    # Strategy 1: Use all even moduli (63 moduli)
    even_moduli = [m for m in available_moduli if m % 2 == 0]
    
    # Try different orderings for greedy
    best_result = []
    
    orderings = [
        sorted(even_moduli),  # ascending
        sorted(even_moduli, reverse=True),  # descending
        sorted(even_moduli, key=lambda m: -m.bit_count()),  # more bits first
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        assigned = []
        greedy_assign(ordering, assigned)
        assigned = local_search(assigned)
        if len(assigned) > len(best_result):
            best_result = assigned[:]
    
    # Strategy 2: Try subsets that include odd moduli with shared factors
    # Numbers divisible by 3: 3, 6, 9, 12, ...
    # Numbers divisible by 5: 5, 10, 15, ...
    # But we need pairwise gcd >= 2 for all pairs.
    
    # Let's try: all multiples of 2 (even) - already done
    # All multiples of 3: only 42 in [2,128], but not all pairs have gcd>=2 with evens
    # Actually, 3 and 2 have gcd 1, so we can't mix 3 with 2.
    # But 6 and 2 have gcd 2, 6 and 3 have gcd 3, etc.
    
    # The largest set with pairwise gcd >= 2 is likely all evens or all multiples of some small prime.
    # All evens: 63 moduli.
    # All multiples of 3: 42 moduli.
    # All multiples of 5: 25 moduli.
    
    # So evens give the most candidates. Let's also try:
    # All numbers divisible by 2 or 3? No, 2 and 3 have gcd 1.
    
    # What about numbers that are all divisible by 2? That's just evens.
    # What about numbers where every pair shares at least one prime factor?
    # This is a "pairwise intersecting" family.
    
    # For now, let's stick with evens and do more local search.
    
    # Strategy 3: More aggressive local search with multiple restarts
    for restart in range(5):
        if time.time() > deadline:
            break
        assigned = []
        # Random order
        random.shuffle(even_moduli)
        greedy_assign(even_moduli, assigned)
        assigned = local_search(assigned, max_iters=20000)
        if len(assigned) > len(best_result):
            best_result = assigned[:]
    
    # Strategy 4: Try to add odd moduli that are compatible
    # For an odd modulus m to be compatible with an even modulus e,
    # we need gcd(m, e) >= 2, which means m must share a factor with e.
    # Since e is even, gcd(m,e) >= 2 means m must be even... but m is odd.
    # So gcd(m, e) is odd, and for it to be >= 2, it must be >= 3.
    # So m must be divisible by 3, 5, 7, etc., and e must share that factor.
    
    # This gets complex. Let's try adding odd moduli one by one.
    odd_candidates = [m for m in available_moduli if m % 2 != 0]
    used_moduli = set(m for (_, m) in best_result)
    
    for m in sorted(odd_candidates, key=lambda x: -x):  # larger first (more residue choices)
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        for r in range(m):
            if is_compatible(r, m, best_result):
                best_result.append((r, m))
                used_moduli.add(m)
                break
    
    # Final local search
    best_result = local_search(best_result, max_iters=30000)
    
    # Convert to list of [residue, modulus]
    result = [[r, m] for (r, m) in best_result]
    
    if not result:
        result = [[1, 2]]
    
    return result
# EVOLVE-BLOCK-END
