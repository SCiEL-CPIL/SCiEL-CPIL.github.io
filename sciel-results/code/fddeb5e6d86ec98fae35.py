import time
from math import gcd
from fractions import Fraction
import random

def _check_feasibility(moduli):
    """
    Check if there exists an assignment of residues for the given moduli
    such that they are pairwise disjoint.
    Returns True if feasible, False otherwise.
    Uses backtracking.
    """
    if not moduli:
        return True
    
    # Sort moduli to help with pruning (smaller moduli have fewer residues, but constraints might be tighter? 
    # Actually, larger gcds mean tighter constraints. Let's sort by gcd with others? 
    # Simple heuristic: sort by modulus size.
    moduli_sorted = sorted(moduli)
    n = len(moduli_sorted)
    residues = [None] * n
    
    # Precompute gcds
    g = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            g[i][j] = gcd(moduli_sorted[i], moduli_sorted[j])
            if g[i][j] == 1:
                return False # Infeasible immediately

    def is_consistent(i, r_i):
        for j in range(i):
            if residues[j] is not None:
                if r_i % g[i][j] == residues[j] % g[i][j]:
                    return False
        return True

    def backtrack(i):
        if i == n:
            return True
        m = moduli_sorted[i]
        # Try residues in a specific order? 
        # To maximize flexibility, maybe try residues that are "central"? 
        # Or just 0 to m-1.
        for r in range(m):
            if is_consistent(i, r):
                residues[i] = r
                if backtrack(i + 1):
                    return True
                residues[i] = None
        return False

    return backtrack(0)

def _score(moduli):
    return sum(1.0 / m for m in moduli)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    all_moduli = list(range(20, 201))
    
    # Initial Greedy Construction
    # Sort by 1/m descending (m ascending)
    # We want to pick moduli that share factors.
    
    def greedy_initial():
        selected = []
        # Try to build a set where all moduli share a common factor? 
        # Or just pairwise.
        # Let's try to pick moduli divisible by 2 first, then 3, etc.
        # Actually, a mixed set like {6, 10, 15, 30...} is good.
        
        # Strategy: Pick smallest m that is compatible with current set.
        for m in all_moduli:
            if time.time() > deadline - 5:
                break
            # Check pairwise gcd > 1 with all in selected
            ok = True
            for m_sel in selected:
                if gcd(m, m_sel) == 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check feasibility of selected + [m]
            # To speed up, only check if we add m. 
            # We can maintain the residue assignment? 
            # For now, just check full feasibility. 
            # This is O(n * m * depth). Might be slow if we do it for every m.
            
            # Optimization: Only check feasibility if the set size is small or periodically.
            # Let's check feasibility.
            if _check_feasibility(selected + [m]):
                selected.append(m)
                
        return selected

    best_moduli = greedy_initial()
    best_score = _score(best_moduli)
    
    # Local Search / Simulated Annealing
    current_moduli = best_moduli[:]
    current_score = best_score
    
    def try_move(moduli, move_type, m_target):
        """
        move_type: 'add', 'remove', 'swap'
        m_target: modulus to add/remove/swap with
        """
        new_moduli = moduli[:]
        
        if move_type == 'add':
            # Check pairwise gcd
            for m in new_moduli:
                if gcd(m, m_target) == 1:
                    return None, -1
            new_moduli.append(m_target)
            
        elif move_type == 'remove':
            if m_target not in new_moduli:
                return None, -1
            new_moduli.remove(m_target)
            
        elif move_type == 'swap':
            if m_target not in new_moduli:
                return None, -1
            # m_target is the one to remove. We need a new one. 
            # This function signature is ambiguous for swap. 
            # Let's handle swap outside.
            return None, -1
        
        # Check feasibility
        if not _check_feasibility(new_moduli):
            return None, -1
            
        return new_moduli, _score(new_moduli)

    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    # Precompute compatibility matrix for speed? 
    # gcd(m1, m2) > 1
    
    while time.time() < deadline - 5:
        improved = False
        
        # Randomly pick a move
        # 1. Add a random modulus
        if random.random() < 0.5 and len(current_moduli) < 180:
            candidates = [m for m in all_moduli if m not in current_moduli]
            if candidates:
                m_add = random.choice(candidates)
                # Check gcd constraint first
                gcd_ok = all(gcd(m, m_add) > 1 for m in current_moduli)
                if gcd_ok:
                    new_moduli, new_score = try_move(current_moduli, 'add', m_add)
                    if new_moduli is not None:
                        if new_score > current_score:
                            current_moduli = new_moduli
                            current_score = new_score
                            improved = True
                        elif random.random() < pow(2.718, (new_score - current_score) / T):
                            current_moduli = new_moduli
                            current_score = new_score
                            improved = True

        # 2. Remove a random modulus
        elif random.random() < 0.3 and len(current_moduli) > 0:
            m_rem = random.choice(current_moduli)
            new_moduli, new_score = try_move(current_moduli, 'remove', m_rem)
            if new_moduli is not None:
                # Removal always improves feasibility, but lowers score. 
                # We only accept if it allows better future moves? 
                # In SA, we accept worse scores with prob exp(delta/T).
                if random.random() < pow(2.718, (new_score - current_score) / T):
                    current_moduli = new_moduli
                    current_score = new_score
                    improved = True

        # 3. Swap: Remove one, Add one
        else:
            if len(current_moduli) > 0:
                m_rem = random.choice(current_moduli)
                candidates = [m for m in all_moduli if m not in current_moduli]
                if candidates:
                    m_add = random.choice(candidates)
                    # Check gcd for m_add against (current - m_rem)
                    others = [m for m in current_moduli if m != m_rem]
                    gcd_ok = all(gcd(m, m_add) > 1 for m in others)
                    if gcd_ok:
                        new_moduli = others + [m_add]
                        if _check_feasibility(new_moduli):
                            new_score = _score(new_moduli)
                            if new_score > current_score:
                                current_moduli = new_moduli
                                current_score = new_score
                                improved = True
                            elif random.random() < pow(2.718, (new_score - current_score) / T):
                                current_moduli = new_moduli
                                current_score = new_score
                                improved = True

        if not improved:
            T *= cooling
        
        if current_score > best_score:
            best_score = current_score
            best_moduli = current_moduli[:]

    # Final verification and residue assignment
    # We need to return [[m, r], ...]
    # _check_feasibility returned True/False but didn't store residues.
    # We need a function that returns the residues.
    
    def get_residues(moduli):
        if not moduli:
            return []
        moduli_sorted = sorted(moduli)
        n = len(moduli_sorted)
        residues = [None] * n
        g = [[1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                g[i][j] = gcd(moduli_sorted[i], moduli_sorted[j])
        
        def is_consistent(i, r_i):
            for j in range(i):
                if residues[j] is not None:
                    if r_i % g[i][j] == residues[j] % g[i][j]:
                        return False
            return True

        def backtrack(i):
            if i == n:
                return True
            m = moduli_sorted[i]
            for r in range(m):
                if is_consistent(i, r):
                    residues[i] = r
                    if backtrack(i + 1):
                        return True
                    residues[i] = None
            return False

        if backtrack(0):
            return [[moduli_sorted[i], residues[i]] for i in range(n)]
        return None

    result = get_residues(best_moduli)
    if result is None:
        # Fallback to a simple valid set
        return [[20, 0], [30, 1]] # 20, 30 gcd 10. 0 != 1 mod 10. Valid.
        
    return result
