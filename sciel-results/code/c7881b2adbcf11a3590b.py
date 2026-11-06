import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table for all pairs of moduli
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute prime factor counts (omega) for ordering heuristics
    def omega(m):
        count = 0
        d = 2
        while d * d <= m:
            if m % d == 0:
                count += 1
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            count += 1
        return count
    
    omega_cache = {m: omega(m) for m in moduli}
    
    # Precompute compatibility count: for each modulus, how many others share gcd > 1
    compat_count = [0] * n_mod
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            if gcd_table[i][j] > 1:
                compat_count[i] += 1
                compat_count[j] += 1
    
    # Precompute a few deterministic orderings
    orderings = []
    orderings.append(sorted(moduli, key=lambda m: (-compat_count[mod_to_idx[m]], m)))  # Most compatible first
    orderings.append(sorted(moduli, key=lambda m: (compat_count[mod_to_idx[m]], m)))   # Least compatible first
    orderings.append(sorted(moduli, key=lambda m: (-omega_cache[m], m)))  # Most constrained first
    orderings.append(sorted(moduli, key=lambda m: (omega_cache[m], m)))   # Least constrained first
    orderings.append(sorted(moduli))                                       # Ascending
    orderings.append(sorted(moduli, reverse=True))                         # Descending
    
    def _find_residue_fast(m_idx, selected_pairs, selected_mod_indices):
        """
        Find a residue r in [0, moduli[m_idx]) that is disjoint from all selected classes.
        Uses a bitset approach for efficiency.
        """
        m = moduli[m_idx]
        # Build constraints: for each selected class with gcd > 1, we have (g, r_i)
        constraints = []
        for k, (r_i, m_i) in enumerate(selected_pairs):
            m_i_idx = selected_mod_indices[k]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))
        
        if not constraints:
            return 0  # Any residue works, pick 0
        
        # Mark forbidden residues
        forbidden = [False] * m
        for (g, r_i) in constraints:
            # Residues r where (r - r_i) % g == 0 are forbidden
            start = r_i % g
            for r in range(start, m, g):
                forbidden[r] = True
        
        # Find first non-forbidden residue
        for r in range(m):
            if not forbidden[r]:
                return r
        return None
    
    def greedy_pass(order_indices):
        """
        One greedy pass given an ordering of modulus indices.
        Returns (list_of_pairs, list_of_mod_indices)
        """
        selected_pairs = []
        selected_mod_indices = []
        used = set()
        
        for m_idx in order_indices:
            if moduli[m_idx] in used:
                continue
            # Check if this modulus can even be added: gcd must be > 1 with all selected
            compatible = True
            for m_sel_idx in selected_mod_indices:
                if gcd_table[m_idx][m_sel_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            
            r = _find_residue_fast(m_idx, selected_pairs, selected_mod_indices)
            if r is not None:
                selected_pairs.append((r, moduli[m_idx]))
                selected_mod_indices.append(m_idx)
                used.add(moduli[m_idx])
        
        return selected_pairs, selected_mod_indices
    
    best_pairs = []
    best_count = 0
    
    # Run deterministic orderings
    for order in orderings:
        if time.time() > deadline:
            break
        order_indices = [mod_to_idx[m] for m in order]
        pairs, _ = greedy_pass(order_indices)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]
    
    # Run randomized restarts with various strategies
    random.seed(42)
    num_restarts = 3000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Mix different randomization strategies
        strategy = restart % 4
        all_indices = list(range(n_mod))
        
        if strategy == 0:
            # Pure random
            random.shuffle(all_indices)
        elif strategy == 1:
            # Sort by compat count, then perturb
            all_indices.sort(key=lambda i: compat_count[i])
            for i in range(len(all_indices) - 1):
                if random.random() < 0.15:
                    j = random.randint(i + 1, len(all_indices) - 1)
                    all_indices[i], all_indices[j] = all_indices[j], all_indices[i]
        elif strategy == 2:
            # Sort by omega, then perturb
            all_indices.sort(key=lambda i: omega_cache[moduli[i]])
            for i in range(len(all_indices) - 1):
                if random.random() < 0.15:
                    j = random.randint(i + 1, len(all_indices) - 1)
                    all_indices[i], all_indices[j] = all_indices[j], all_indices[i]
        else:
            # Reverse compat count, then perturb
            all_indices.sort(key=lambda i: -compat_count[i])
            for i in range(len(all_indices) - 1):
                if random.random() < 0.15:
                    j = random.randint(i + 1, len(all_indices) - 1)
                    all_indices[i], all_indices[j] = all_indices[j], all_indices[i]
        
        pairs, _ = greedy_pass(all_indices)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]
    
    # Local search: try to add more moduli and swap existing ones
    if time.time() < deadline - 5.0 and best_pairs:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_pairs = best_pairs[:]
            current_mod_indices = [mod_to_idx[m] for r, m in current_pairs]
            used = set(m for r, m in current_pairs)
            
            # Try adding each unused modulus
            for m_idx in range(n_mod):
                if time.time() > deadline - 2.0:
                    break
                m = moduli[m_idx]
                if m in used:
                    continue
                
                # Check gcd compatibility
                compatible = True
                for m_sel_idx in current_mod_indices:
                    if gcd_table[m_idx][m_sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                r = _find_residue_fast(m_idx, current_pairs, current_mod_indices)
                if r is not None:
                    current_pairs.append((r, m))
                    current_mod_indices.append(m_idx)
                    used.add(m)
                    improved = True
            
            # Try swapping: remove one modulus and try to add two others
            if not improved and len(current_pairs) >= 2:
                for remove_idx in range(len(current_pairs)):
                    if time.time() > deadline - 2.0:
                        break
                    
                    # Remove this modulus
                    removed_pair = current_pairs[remove_idx]
                    current_pairs.remove(removed_pair)
                    removed_idx = current_mod_indices[remove_idx]
                    current_mod_indices.remove(removed_idx)
                    used.remove(removed_pair[1])
                    
                    # Try to add two new moduli
                    added_count = 0
                    for m_idx in range(n_mod):
                        if time.time() > deadline - 2.0:
                            break
                        m = moduli[m_idx]
                        if m in used:
                            continue
                        
                        compatible = True
                        for m_sel_idx in current_mod_indices:
                            if gcd_table[m_idx][m_sel_idx] == 1:
                                compatible = False
                                break
                        if not compatible:
                            continue
                        
                        r = _find_residue_fast(m_idx, current_pairs, current_mod_indices)
                        if r is not None:
                            current_pairs.append((r, m))
                            current_mod_indices.append(m_idx)
                            used.add(m)
                            added_count += 1
                            if added_count >= 2:
                                break
                    
                    if added_count > 1:
                        improved = True
                    else:
                        # Restore the removed pair
                        current_pairs.append(removed_pair)
                        current_mod_indices.append(removed_idx)
                        used.add(removed_pair[1])
            
            if len(current_pairs) > len(best_pairs):
                best_pairs = current_pairs
    
    return [[r, m] for r, m in best_pairs]

# EVOLVE-BLOCK-END
