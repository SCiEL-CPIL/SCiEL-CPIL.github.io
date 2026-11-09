import random
import time
import math
from collections import defaultdict

def _find_max_clique_moduli():
    """Find largest subset of {2..128} where every pair has gcd > 1."""
    moduli = list(range(2, 129))
    n = len(moduli)
    # Build adjacency: adj[i] = set of j where gcd(moduli[i], moduli[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(moduli[i], moduli[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy: start with all evens (known large clique), try to expand
    # All evens: indices where moduli[i] % 2 == 0
    even_indices = [i for i in range(n) if moduli[i] % 2 == 0]
    best_clique = set(even_indices)
    
    # Try to add non-even moduli to the clique
    for i in range(n):
        if i in best_clique:
            continue
        # Check if i is compatible with all in best_clique
        if all(j in adj[i] for j in best_clique):
            best_clique.add(i)
    
    # Try different starting points
    for start in range(n):
        clique = {start}
        for j in range(n):
            if j in clique:
                continue
            if all(k in adj[j] for k in clique):
                clique.add(j)
        if len(clique) > len(best_clique):
            best_clique = clique
    
    return [moduli[i] for i in sorted(best_clique)]

def _assign_residues_backtrack(moduli_subset, time_limit):
    """Assign residues to moduli_subset using backtracking with forward checking."""
    start = time.time()
    n = len(moduli_subset)
    residues = [None] * n
    # Order by increasing modulus (fewer choices first? or most constrained?)
    # Actually, process in order of increasing number of compatible residue options
    # For now, sort by modulus ascending
    order = sorted(range(n), key=lambda i: moduli_subset[i])
    
    def backtrack(idx):
        if time.time() - start > time_limit:
            return False
        if idx == n:
            return True
        i = order[idx]
        m = moduli_subset[i]
        # Try each residue
        for r in range(m):
            ok = True
            for j in range(idx):
                j_actual = order[j]
                g = math.gcd(m, moduli_subset[j_actual])
                if (r - residues[j_actual]) % g == 0:
                    ok = False
                    break
            if ok:
                residues[i] = r
                if backtrack(idx + 1):
                    return True
                residues[i] = None
        return False
    
    if backtrack(0):
        return [[residues[i], moduli_subset[i]] for i in range(n)]
    return None

def _greedy_assign(moduli_subset, num_restarts=100, time_limit=10.0):
    """Randomized greedy residue assignment."""
    start = time.time()
    n = len(moduli_subset)
    best = []
    
    for _ in range(num_restarts):
        if time.time() - start > time_limit:
            break
        random.shuffle(moduli_subset)
        current = []
        for m in moduli_subset:
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                ok = True
                for r_i, m_i in current:
                    g = math.gcd(m, m_i)
                    if (r - r_i) % g == 0:
                        ok = False
                        break
                if ok:
                    current.append([r, m])
                    break
        if len(current) > len(best):
            best = current
            if len(best) == n:
                break
    
    return best

def construct():
    start_time = time.time()
    timeout = 55.0
    
    # Step 1: Find a large clique of moduli with pairwise gcd > 1
    clique = _find_max_clique_moduli()
    
    # Step 2: Try to assign residues to the clique
    # First try backtracking with a time limit
    bt_result = _assign_residues_backtrack(clique, time_limit=5.0)
    if bt_result is not None:
        return bt_result
    
    # Step 3: Fall back to greedy on the clique
    greedy_result = _greedy_assign(clique, num_restarts=200, time_limit=10.0)
    
    # Step 4: If greedy didn't fill the clique, try to add more moduli
    # Use the original moduli list for additional search
    remaining = [m for m in range(2, 129) if m not in clique]
    full_moduli = clique + remaining
    
    # Try greedy on all moduli as fallback
    all_greedy = _greedy_assign(full_moduli, num_restarts=100, time_limit=15.0)
    
    best = max(greedy_result, all_greedy, key=len)
    
    # Continue with randomized restarts if time allows
    while time.time() - start_time < timeout:
        random.shuffle(full_moduli)
        current_list = []
        for m in full_moduli:
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                compatible = True
                for r_i, m_i in current_list:
                    g = math.gcd(m, m_i)
                    if (r - r_i) % g == 0:
                        compatible = False
                        break
                if compatible:
                    current_list.append([r, m])
                    break
        if len(current_list) > len(best):
            best = current_list
            if len(best) == 127:
                break
    
    return best
