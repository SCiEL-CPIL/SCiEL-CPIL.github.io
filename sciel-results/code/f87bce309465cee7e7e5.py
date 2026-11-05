import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Precompute gcds for all pairs in [20, 200]
    moduli_range = list(range(20, 201))
    n_moduli = len(moduli_range)
    
    # gcd_matrix[i][j] = gcd(moduli_range[i], moduli_range[j])
    gcd_matrix = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            g = math.gcd(moduli_range[i], moduli_range[i+1]) # placeholder, will fix
            gcd_matrix[i][j] = math.gcd(moduli_range[i], moduli_range[j])
            gcd_matrix[j][i] = gcd_matrix[i][j]

    def check_feasibility(moduli_set):
        """
        Check if there exists a set of residues for the given moduli that are pairwise disjoint.
        We use a simple backtracking with constraint propagation.
        Since the number of moduli is small (< 30 typically in high score sets), this is fast.
        """
        k = len(moduli_set)
        if k == 0:
            return []
        
        # Order moduli by size (smaller moduli have fewer residues, might be tighter?)
        # Actually, let's just pick an order. Maybe larger moduli first?
        # Let's try ordering by modulus value descending.
        indices = sorted(range(k), key=lambda i: -moduli_set[i])
        
        residues = [None] * k
        
        def backtrack(idx):
            if idx == k:
                return True
            
            m = moduli_set[indices[idx]]
            # Try each residue for this modulus
            # Optimization: we can prune if a residue is incompatible with already assigned ones
            
            for r in range(m):
                compatible = True
                for j in range(idx):
                    m_other = moduli_set[indices[j]]
                    r_other = residues[indices[j]]
                    g = gcd_matrix[moduli_range.index(m)][moduli_range.index(m_other)]
                    if r % g == r_other % g:
                        compatible = False
                        break
                if compatible:
                    residues[indices[idx]] = r
                    if backtrack(idx + 1):
                        return True
                    residues[indices[idx]] = None
            return False

        if backtrack(0):
            return residues
        return None

    def get_score(moduli_set):
        return sum(1.0 / m for m in moduli_set)

    # Strategy:
    # 1. Start with a greedy set of moduli that are "easy" to satisfy (e.g., prime, or coprime to many others)
    # 2. Try to add more moduli if feasible
    # 3. Try to swap moduli to improve score
    
    # Initial construction: Greedy by 1/m (smallest moduli first) but checking feasibility
    # Actually, to maximize sum 1/m, we want small moduli.
    # But small moduli are more likely to conflict?
    # Let's start with the smallest moduli that can form a feasible set.
    
    # A better heuristic: 
    # Iterate moduli from 20 to 200.
    # Maintain a current set.
    # Try adding m to the set. If feasible, add it.
    # This might miss some combinations where skipping a small m allows many slightly larger ms?
    # Unlikely, since 1/m decreases. But conflicts might prevent adding small m.
    
    # Let's use a randomized restart approach with the greedy add heuristic.
    
    best_set = []
    best_score = 0.0
    
    for _ in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        random.seed()
        # Shuffle moduli? No, we want small moduli. 
        # But maybe randomize the order of checking residues in check_feasibility?
        # Or randomize which moduli we attempt to add?
        
        # Heuristic: Try to build a set by adding moduli in increasing order of 1/m (i.e., increasing m? No, decreasing 1/m is increasing m).
        # We want high 1/m, so we prefer small m.
        # Let's try adding moduli in increasing order of m, but with randomization in feasibility check.
        
        current_set = []
        for m in moduli_range:
            # Try to add m
            new_set = current_set + [m]
            # Check feasibility
            res = check_feasibility(new_set)
            if res is not None:
                current_set = new_set
            # Else skip m
        
        score = get_score(current_set)
        if score > best_score:
            best_score = score
            best_set = current_set[:]
            
        # Local improvement: Try to swap a modulus in best_set with one not in it
        # to improve score.
        # This is complex. Let's stick to the greedy add with randomization in residue selection.
        
        # To improve, we can randomize the order of residues tried in check_feasibility.
        # But check_feasibility is deterministic.
        
    # Alternative: Simulated Annealing on the set of moduli.
    # State: set of moduli.
    # Move: add a modulus, remove a modulus, swap a modulus.
    # Energy: -sum(1/m) if feasible, +infinity if infeasible.
    
    # Let's implement a simple SA on the moduli set.
    
    # Start with a feasible set from greedy
    if not best_set:
        # Fallback: just pick primes?
        best_set = [m for m in moduli_range if all(math.gcd(m, other) == 1 for other in moduli_range[20:201] if other != m)]
        # This is too strict.
        # Just use the greedy result.
        pass

    # Refine best_set with SA
    current_set = best_set[:]
    current_res = check_feasibility(current_set)
    if current_res is None:
        # Should not happen if best_set was feasible
        current_set = []
        current_res = []
    
    current_score = get_score(current_set)
    best_sa_set = current_set[:]
    best_sa_score = current_score
    
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Generate a neighbor
        move_type = random.choice(['add', 'remove', 'swap'])
        neighbor_set = current_set[:]
        
        if move_type == 'add':
            # Pick a random modulus not in set
            available = [m for m in moduli_range if m not in neighbor_set]
            if available:
                m_add = random.choice(available)
                neighbor_set.append(m_add)
                
        elif move_type == 'remove':
            if neighbor_set:
                m_remove = random.choice(neighbor_set)
                neighbor_set.remove(m_remove)
                
        elif move_type == 'swap':
            if neighbor_set:
                available = [m for m in moduli_range if m not in neighbor_set]
                if available:
                    m_remove = random.choice(neighbor_set)
                    m_add = random.choice(available)
                    neighbor_set.remove(m_remove)
                    neighbor_set.append(m_add)
                    
        # Check feasibility of neighbor
        neighbor_res = check_feasibility(neighbor_set)
        
        if neighbor_res is not None:
            neighbor_score = get_score(neighbor_set)
            delta = neighbor_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_set = neighbor_set
                current_res = neighbor_res
                current_score = neighbor_score
                
                if current_score > best_sa_score:
                    best_sa_score = current_score
                    best_sa_set = current_set[:]
                    
        T *= cooling
        
    # Final result
    # Ensure best_sa_set is feasible and return it
    final_res = check_feasibility(best_sa_set)
    if final_res is None:
        # Fallback to initial best_set
        final_res = check_feasibility(best_set)
        if final_res is None:
            # Absolute fallback: empty set? No, must have distinct moduli.
            # Just return the first feasible set found.
            # This should not happen.
            return []
        best_sa_set = best_set
        
    # Map residues back to original moduli order?
    # check_feasibility returns residues in the order of the input moduli_set.
    # We need to return [[m, r], ...]
    
    # best_sa_set is a list of moduli.
    # final_res is a list of residues corresponding to best_sa_set.
    
    result = []
    for i, m in enumerate(best_sa_set):
        result.append([m, final_res[i]])
        
    return result

