"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcd matrix
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def check_compatible(m_idx, r_new, selected_indices, residues):
        """Check if residue r_new for modulus m_idx is compatible with all selected."""
        for s_idx in selected_indices:
            g = gcd_table[m_idx][s_idx]
            if (r_new - residues[s_idx]) % g == 0:
                return False
        return True
    
    def get_valid_residues(m_idx, selected_indices, residues):
        """Get all valid residues for modulus m_idx given selected."""
        valid = []
        m = moduli[m_idx]
        for r in range(m):
            if check_compatible(m_idx, r, selected_indices, residues):
                valid.append(r)
        return valid
    
    def count_conflicts_if_add(m_idx, r_new, selected_indices, residues):
        """Count how many future moduli would be blocked by adding (m_idx, r_new)."""
        # For each unselected modulus, check if this residue blocks ALL its residues
        blocked = 0
        for j in range(n_moduli):
            if j in selected_indices or j == m_idx:
                continue
            g = gcd_table[m_idx][j]
            # Check if r_new blocks all residues of moduli[j]
            # Residue r of j conflicts if (r - r_new) % g == 0
            # Number of conflicting residues = g (since 0 <= r < moduli[j], and g divides moduli[j])
            # Wait, that's not right. Let me think...
            # Actually, for modulus m_j, residues 0..m_j-1, the ones conflicting with r_new mod g
            # are those r where r ≡ r_new (mod g). There are floor(m_j/g) or ceil(m_j/g) such.
            # Since g divides m_j (g = gcd(m_i, m_j)), there are exactly m_j/g such residues.
            # So if m_j/g == m_j, i.e., g == 1, no conflict. Otherwise, m_j/g residues conflict.
            # The modulus is fully blocked only if ALL its residues conflict, which never happens for g >= 2.
            # So this metric isn't quite right. Let me use a different one.
            # Better: count how many residues of j become unavailable.
            # Actually for our purposes, let's just check if adding this makes any future modulus have 0 valid residues.
            pass
        return 0
    
    def score_of(selected_indices, residues):
        return sum(Fraction(1, moduli[i]) for i in selected_indices)
    
    # Strategy 1: Greedy with residue choice that maximizes future options
    def greedy_with_flexibility():
        selected_indices = []
        residues = [0] * n_moduli
        
        # Process moduli in increasing order (smaller moduli have more constraints on others)
        # Actually, let's try: process in order of 1/m (largest first = smallest m first)
        order = sorted(range(n_moduli), key=lambda i: moduli[i])  # increasing m
        
        for m_idx in order:
            valid_r = get_valid_residues(m_idx, selected_indices, residues)
            if not valid_r:
                continue
            
            # Choose the residue that blocks the fewest future options
            # For each valid residue, count how many unselected moduli would have fewer valid residues
            best_r = valid_r[0]
            best_score = -1
            
            # For speed, use a simpler heuristic: choose residue that minimizes conflicts
            # with the most "important" (small) unselected moduli
            unselected = [j for j in range(n_moduli) if j not in selected_indices and j != m_idx]
            
            for r in valid_r:
                # Score: sum over unselected moduli of (1 if this residue doesn't fully block them)
                # Actually, just pick the one that has the fewest gcd conflicts
                conflict_count = 0
                for j in unselected:
                    g = gcd_table[m_idx][j]
                    if g > 1:
                        # This residue conflicts with g residues of modulus j
                        # But j still has moduli[j] - g valid residues
                        # Penalize more for larger g relative to moduli[j]
                        conflict_count += g / moduli[j]
                if conflict_count < best_score or best_score == -1:
                    best_score = conflict_count
                    best_r = r
            
            selected_indices.append(m_idx)
            residues[m_idx] = best_r
        
        return selected_indices, residues
    
    # Strategy 2: Greedy with different ordering
    def greedy_order(order, residue_strategy='min_conflict'):
        selected_indices = []
        residues = [0] * n_moduli
        
        for m_idx in order:
            valid_r = get_valid_residues(m_idx, selected_indices, residues)
            if not valid_r:
                continue
            
            if residue_strategy == 'first':
                best_r = valid_r[0]
            elif residue_strategy == 'last':
                best_r = valid_r[-1]
            elif residue_strategy == 'random':
                best_r = random.choice(valid_r)
            else:  # min_conflict
                unselected = [j for j in range(n_moduli) if j not in selected_indices and j != m_idx]
                best_r = valid_r[0]
                best_conf = float('inf')
                for r in valid_r:
                    conf = 0
                    for j in unselected:
                        g = gcd_table[m_idx][j]
                        if g > 1:
                            conf += g / moduli[j]
                    if conf < best_conf:
                        best_conf = conf
                        best_r = r
            
            selected_indices.append(m_idx)
            residues[m_idx] = best_r
        
        return selected_indices, residues
    
    # Local search: try to add a removed modulus or swap residues
    def local_search(selected_indices, residues, max_iters=50000):
        sel_set = set(selected_indices)
        current_score = score_of(selected_indices, residues)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Try adding a non-selected modulus
            for m_idx in range(n_moduli):
                if m_idx in sel_set:
                    continue
                valid_r = get_valid_residues(m_idx, selected_indices, residues)
                if valid_r:
                    # Add it
                    selected_indices.append(m_idx)
                    sel_set.add(m_idx)
                    # Pick best residue
                    unselected = [j for j in range(n_moduli) if j not in sel_set and j != m_idx]
                    best_r = valid_r[0]
                    best_conf = float('inf')
                    for r in valid_r:
                        conf = 0
                        for j in unselected:
                            g = gcd_table[m_idx][j]
                            if g > 1:
                                conf += g / moduli[j]
                        if conf < best_conf:
                            best_conf = conf
                            best_r = r
                    residues[m_idx] = best_r
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try changing residue of a selected modulus to free up space
            if selected_indices:
                # Pick a random selected modulus and try different residues
                m_idx = random.choice(selected_indices)
                old_r = residues[m_idx]
                valid_r = get_valid_residues(m_idx, [i for i in selected_indices if i != m_idx], residues)
                if len(valid_r) > 1:
                    new_r = random.choice([r for r in valid_r if r != old_r])
                    residues[m_idx] = new_r
                    # Check if this allows adding a new modulus
                    for j in range(n_moduli):
                        if j in sel_set:
                            continue
                        valid_j = get_valid_residues(j, selected_indices, residues)
                        if valid_j:
                            selected_indices.append(j)
                            sel_set.add(j)
                            residues[j] = valid_j[0]
                            improved = True
                            break
                    if not improved:
                        residues[m_idx] = old_r  # revert
            
            # Try removing a modulus and adding two better ones
            if len(selected_indices) > 2 and random.random() < 0.1:
                remove_idx = random.choice(selected_indices)
                old_r = residues[remove_idx]
                selected_indices.remove(remove_idx)
                sel_set.discard(remove_idx)
                
                # Try to add back different moduli
                added_any = False
                for j in range(n_moduli):
                    if j in sel_set or j == remove_idx:
                        continue
                    valid_j = get_valid_residues(j, selected_indices, residues)
                    if valid_j:
                        selected_indices.append(j)
                        sel_set.add(j)
                        residues[j] = valid_j[0]
                        added_any = True
                        break
                
                if not added_any:
                    # Revert
                    selected_indices.append(remove_idx)
                    sel_set.add(remove_idx)
                    residues[remove_idx] = old_r
        
        return selected_indices, residues, score_of(selected_indices, residues)
    
    # Try multiple strategies
    best_selected = None
    best_residues = None
    best_score = Fraction(0)
    
    # Strategy A: Greedy with flexibility
    sel_a, res_a = greedy_with_flexibility()
    score_a = score_of(sel_a, res_a)
    if score_a > best_score:
        best_score = score_a
        best_selected = sel_a[:]
        best_residues = res_a[:]
    
    # Strategy B: Greedy with different orderings
    orders = [
        sorted(range(n_moduli), key=lambda i: moduli[i]),           # increasing m
        sorted(range(n_moduli), key=lambda i: -moduli[i]),           # decreasing m
        sorted(range(n_moduli), key=lambda i: -1.0/moduli[i]),       # same as increasing
        list(range(n_moduli)),
    ]
    
    for order in orders:
        for strat in ['min_conflict', 'first', 'random']:
            if time.time() > deadline - 10.0:
                break
            sel_b, res_b = greedy_order(order, strat)
            score_b = score_of(sel_b, res_b)
            if score_b > best_score:
                best_score = score_b
                best_selected = sel_b[:]
                best_residues = res_b[:]
    
    # Local search from best solution
    if time.time() < deadline - 15.0 and best_selected is not None:
        sel_ls, res_ls, score_ls = local_search(best_selected[:], best_residues[:], max_iters=100000)
        if score_ls > best_score:
            best_score = score_ls
            best_selected = sel_ls
            best_residues = res_ls
    
    # Multiple local search restarts
    if time.time() < deadline - 10.0:
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            # Random greedy
            order = list(range(n_moduli))
            random.shuffle(order)
            sel_r, res_r = greedy_order(order, 'random')
            score_r = score_of(sel_r, res_r)
            
            # Local search
            if time.time() < deadline - 5.0:
                sel_ls, res_ls, score_ls = local_search(sel_r, res_r, max_iters=50000)
                if score_ls > best_score:
                    best_score = score_ls
                    best_selected = sel_ls
                    best_residues = res_ls
    
    if best_selected is None:
        best_selected, best_residues = greedy_with_flexibility()
    
    # Build result
    result = [[moduli[i], best_residues[i]] for i in best_selected]
    return result
# EVOLVE-BLOCK-END
