import time
from math import gcd
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    moduli_range = list(range(2, 129))
    n_moduli = len(moduli_range)

    # Precompute gcd matrix and conflict graph
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    neighbors = [[] for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            g = gcd(moduli_range[i], moduli_range[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            if g > 1:
                neighbors[i].append(j)
                neighbors[j].append(i)
    
    # Sort neighbors by degree (descending) to help clique finding
    neighbor_degrees = [len(neighbors[i]) for i in range(n_moduli)]
    for i in range(n_moduli):
        neighbors[i].sort(key=lambda x: neighbor_degrees[x], reverse=True)

    def is_compatible_residue(residues, selected_idx, new_r, new_m_idx):
        """Check if new residue r with modulus at new_m_idx is compatible with all existing."""
        for i in selected_idx:
            g = gcd_table[new_m_idx][i]
            if new_r % g == residues[i] % g:
                return False
        return True

    def solve_residue_csp(moduli_indices, residues, selected_idx):
        """
        Given a set of moduli (indices), find valid residues.
        Uses backtracking with forward checking / MRV heuristic.
        moduli_indices: list of indices of moduli to assign.
        Returns True if a valid assignment is found in 'residues' and 'selected_idx' is updated.
        """
        if not moduli_indices:
            return True
        
        # MRV Heuristic: Pick the variable (modulus) with the fewest remaining valid values
        # To simplify, we just pick the one with the highest degree in the current subgraph
        # or just the first one if we want speed. Let's try sorting by degree in the current set.
        
        # Calculate degrees within the current set
        current_set = set(moduli_indices)
        degrees = {}
        for i in moduli_indices:
            deg = 0
            for j in moduli_indices:
                if i != j and gcd_table[i][j] > 1:
                    deg += 1
            degrees[i] = deg
        
        # Pick the most constrained variable
        current_var = max(moduli_indices, key=lambda x: degrees[x])
        
        m = moduli_range[current_var]
        
        # Try all residues for this variable
        for r in range(m):
            if is_compatible_residue(residues, selected_idx, r, current_var):
                residues[current_var] = r
                selected_idx.append(current_var)
                
                # Recurse on the remaining variables
                remaining = [v for v in moduli_indices if v != current_var]
                if solve_residue_csp(remaining, residues, selected_idx):
                    return True
                
                # Backtrack
                selected_idx.pop()
                residues[current_var] = 0
        
        return False

    def greedy_clique():
        """
        Greedy clique finding.
        Start with the node with highest degree, add neighbors that are connected to all current clique members.
        """
        # Order nodes by degree descending
        order = sorted(range(n_moduli), key=lambda i: neighbor_degrees[i], reverse=True)
        
        best_clique = []
        for start_node in order:
            clique = [start_node]
            candidates = [n for n in neighbors[start_node] if n > start_node] # Avoid duplicates
            
            for node in candidates:
                # Check if node is connected to all in clique
                if all(gcd_table[node][c] > 1 for c in clique):
                    clique.append(node)
            
            if len(clique) > len(best_clique):
                best_clique = clique
                
        return best_clique

    # Strategy 1: Find a large clique and solve CSP
    best_result = []
    
    # Get initial clique
    initial_clique = greedy_clique()
    
    # Try to solve CSP for this clique
    residues = [0] * n_moduli
    selected_idx = []
    
    # We need to be careful with the 'residues' and 'selected_idx' state management
    # The CSP solver modifies them in place
    
    # Let's create a wrapper to handle the state
    def try_solve(clique):
        residues_local = [0] * n_moduli
        selected_local = []
        if solve_residue_csp(clique, residues_local, selected_local):
            result = [[residues_local[i], moduli_range[i]] for i in selected_local]
            return result
        return []

    result = try_solve(initial_clique)
    if result:
        best_result = result
        print(f"Found clique of size {len(result)}")

    # Strategy 2: Try to expand the clique
    # If we found a solution, try to add more moduli
    if best_result:
        used_moduli = set(m for _, m in best_result)
        residues_arr = [0] * n_moduli
        selected_idx = []
        for [r, m] in best_result:
            m_idx = moduli_range.index(m)
            residues_arr[m_idx] = r
            selected_idx.append(m_idx)
        
        # Try adding unused moduli one by one
        for m_idx in range(n_moduli):
            if moduli_range[m_idx] in used_moduli:
                continue
            if time.time() > deadline - 5:
                break
            
            # Check if m_idx can be added to the clique (connected to all existing)
            if all(gcd_table[m_idx][i] > 1 for i in selected_idx):
                # Try to find a residue for m_idx that is compatible
                m = moduli_range[m_idx]
                found = False
                for r in range(m):
                    if is_compatible_residue(residues_arr, selected_idx, r, m_idx):
                        residues_arr[m_idx] = r
                        selected_idx.append(m_idx)
                        used_moduli.add(m)
                        best_result.append([r, m])
                        found = True
                        break
                # If not found, we might need to re-solve the whole CSP
                # For now, just skip if greedy residue assignment fails
                if not found:
                     # Try full re-solve with this new modulus included
                     new_clique = selected_idx + [m_idx]
                     # Note: selected_idx was not modified in the failed attempt, so this is safe
                     # But we need to pass the *original* clique + new node
                     orig_clique = [i for i in selected_idx]
                     new_clique = orig_clique + [m_idx]
                     
                     res = try_solve(new_clique)
                     if res and len(res) > len(best_result):
                         best_result = res
                         # Update internal state if we want to continue expanding
                         used_moduli = set(m for _, m in best_result)
                         residues_arr = [0] * n_moduli
                         selected_idx = []
                         for [r, m] in best_result:
                             m_idx2 = moduli_range.index(m)
                             residues_arr[m_idx2] = r
                             selected_idx.append(m_idx2)

    # Strategy 3: Randomized restarts for clique finding
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Randomized greedy clique
        order = list(range(n_moduli))
        random.shuffle(order)
        # Sort by degree within the shuffled list? No, just pick random start
        start_node = random.choice(order)
        clique = [start_node]
        candidates = [n for n in neighbors[start_node] if n not in clique]
        random.shuffle(candidates)
        
        for node in candidates:
            if all(gcd_table[node][c] > 1 for c in clique):
                clique.append(node)
                
        if len(clique) > len(best_result):
            res = try_solve(clique)
            if res and len(res) > len(best_result):
                best_result = res

    if not best_result:
        # Fallback: Single modulus
        best_result = [[0, 2]]

    return best_result
