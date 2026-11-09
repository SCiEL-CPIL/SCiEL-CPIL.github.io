"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    n = len(all_moduli)  # 127 moduli
    
    # Precompute non-coprime adjacency (bitset)
    # Two moduli are "compatible" if gcd > 1 (they can be made disjoint)
    non_coprime_adj = [0] * n
    
    for i in range(n):
        mi = all_moduli[i]
        for j in range(i + 1, n):
            mj = all_moduli[j]
            if gcd(mi, mj) > 1:
                non_coprime_adj[i] |= (1 << j)
                non_coprime_adj[j] |= (1 << i)
    
    # Find maximum clique in non-coprime graph using branch and bound
    # A clique is a set where every pair has gcd > 1
    best_clique = []
    best_size = 0
    
    # Order vertices by degree (descending) for better pruning
    vertex_order = list(range(n))
    vertex_order.sort(key=lambda i: bin(non_coprime_adj[i]).count('1'), reverse=True)
    
    # Map from original index to position in sorted order
    pos_of = [0] * n
    for pos, orig in enumerate(vertex_order):
        pos_of[orig] = pos
    
    # Rebuild adjacency in sorted order
    adj_sorted = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (non_coprime_adj[vertex_order[i]] >> vertex_order[j]) & 1:
                adj_sorted[i] |= (1 << j)
                adj_sorted[j] |= (1 << i)
    
    def branch_and_bound(candidates, current_clique, current_size, depth):
        nonlocal best_size, best_clique
        
        if time.time() > deadline - 3:
            return
        
        # Pruning: if we can't beat best, stop
        if current_size + len(candidates) <= best_size:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_clique = current_clique[:]
            return
        
        # Try each candidate vertex
        cand_list = list(candidates)
        
        for idx in cand_list:
            if time.time() > deadline - 3:
                return
            
            # New candidates must be compatible with idx and already in candidates after idx
            new_candidates = [v for v in cand_list if v > idx and (adj_sorted[idx] >> v) & 1]
            
            new_clique = current_clique + [vertex_order[idx]]
            branch_and_bound(new_candidates, new_clique, current_size + 1, depth + 1)
            
            # Remove idx from candidates for subsequent iterations
            cand_list.remove(idx)
    
    # Initial candidates: all vertices
    branch_and_bound(list(range(n)), [], 0, 0)
    
    # Fallback: if B&B didn't find anything good, use greedy
    if best_size < 10:
        # Greedy: pick vertices in degree order, add if compatible with all chosen
        greedy_clique = []
        for i in vertex_order:
            if time.time() > deadline - 5:
                break
            compatible = True
            for j in greedy_clique:
                if not (adj_sorted[i] >> j) & 1:
                    compatible = False
                    break
            if compatible:
                greedy_clique.append(i)
        if len(greedy_clique) > best_size:
            best_size = len(greedy_clique)
            best_clique = greedy_clique
    
    # Convert clique back to moduli
    clique_moduli = [all_moduli[i] for i in best_clique]
    
    if not clique_moduli:
        # Fallback: just use multiples of 2
        clique_moduli = [m for m in range(2, MAX_MOD + 1) if m % 2 == 0]
    
    # Now assign residues: for each modulus, find a residue compatible with all others
    # Two classes (r1 mod m1) and (r2 mod m2) are disjoint iff r1 % gcd(m1,m2) != r2 % gcd(m1,m2)
    # We need to assign residues such that for every pair, this condition holds.
    
    chosen = []  # list of (r, m)
    
    # Process moduli in order of increasing number of constraints
    # Sort by the maximum gcd with other moduli (fewer constraints first)
    def constraint_score(m):
        max_g = 0
        for other_m in clique_moduli:
            if other_m != m:
                g = gcd(m, other_m)
                if g > max_g:
                    max_g = g
        return max_g
    
    sorted_moduli = sorted(clique_moduli, key=constraint_score)
    
    for m in sorted_moduli:
        if time.time() > deadline - 3:
            break
        
        found = False
        for r in range(m):
            compatible = True
            for r_prev, m_prev in chosen:
                g = gcd(m, m_prev)
                if r % g == r_prev % g:
                    compatible = False
                    break
            if compatible:
                chosen.append((r, m))
                found = True
                break
        
        if not found:
            # Try shuffled residues
            import random
            random.seed(hash(m) % 10000)
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                compatible = True
                for r_prev, m_prev in chosen:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    chosen.append((r, m))
                    found = True
                    break
    
    # Try to add back any moduli we couldn't assign residues to
    chosen_mod_set = set(m for r, m in chosen)
    remaining = [m for m in clique_moduli if m not in chosen_mod_set]
    
    for m in remaining:
        if time.time() > deadline - 2:
            break
        for r in range(m):
            compatible = True
            for r_prev, m_prev in chosen:
                g = gcd(m, m_prev)
                if r % g == r_prev % g:
                    compatible = False
                    break
            if compatible:
                chosen.append((r, m))
                break
    
    if chosen:
        return [[r, m] for r, m in chosen]
    
    return [[1, 2]]
# EVOLVE-BLOCK-END
