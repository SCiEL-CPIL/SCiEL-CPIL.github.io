import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety and evaluation overhead
    
    moduli = list(range(20, 201))
    # We prioritize smaller moduli, so we iterate in increasing order of m.
    # However, for backtracking, we want to explore high-value (small m) options first.
    
    best_sum = 0.0
    best_solution = []
    
    # Pre-compute gcds for speed?
    # gcd is fast in Python 3.9+ with math.gcd.
    
    # Upper bound calculation helper
    # For a given set of selected moduli, what is the max sum of reciprocals we can add?
    # A loose bound is sum(1/m for m in remaining).
    # A tighter bound is sum(1/m for m in remaining if gcd(m, m_sel) > 1 for all m_sel in selected).
    
    def is_compatible(m_new, selected):
        for m_sel, _ in selected:
            if math.gcd(m_new, m_sel) == 1:
                return False
        return True

    def get_valid_residues(m_new, selected):
        # Returns a list of valid residues for m_new
        # r is valid if for all (m_sel, r_sel) in selected:
        # r % gcd(m_new, m_sel) != r_sel % gcd(m_new, m_sel)
        
        blocked = set()
        for m_sel, r_sel in selected:
            g = math.gcd(m_new, m_sel)
            if g == 1:
                # Should not happen if is_compatible was checked, but safe
                return [] 
            r_sel_mod = r_sel % g
            # All r such that r % g == r_sel_mod are blocked
            # These are r_sel_mod, r_sel_mod + g, r_sel_mod + 2g, ...
            for r in range(r_sel_mod, m_new, g):
                blocked.add(r)
        
        valid = []
        for r in range(m_new):
            if r not in blocked:
                valid.append(r)
        return valid

    def upper_bound(start_idx, selected):
        # Sum of 1/m for m in moduli[start_idx:] that are compatible with all in selected
        s = 0.0
        for i in range(start_idx, len(moduli)):
            m = moduli[i]
            if is_compatible(m, selected):
                s += 1.0 / m
        return s

    def dfs(start_idx, selected, current_sum):
        nonlocal best_sum, best_solution
        
        if time.time() > deadline:
            return
            
        # Pruning: if current sum + upper bound <= best sum, no need to continue
        # Note: upper_bound is calculated based on current selected set.
        # This calculation is O(N * K) where K is size of selected.
        # To make it faster, we can just use the loose bound first, or optimize.
        
        # Loose bound: sum of 1/m for all remaining
        # Pre-compute suffix sums of 1/m
        # suffix_sum[i] = sum(1/m for m in moduli[i:])
        
        if current_sum + suffix_sums[start_idx] <= best_sum:
            return
            
        # Update best
        if current_sum > best_sum:
            best_sum = current_sum
            best_solution = list(selected)
            
        # Try adding each modulus from start_idx
        for i in range(start_idx, len(moduli)):
            if time.time() > deadline:
                return
                
            m = moduli[i]
            
            # Check compatibility
            if not is_compatible(m, selected):
                continue
                
            # Find valid residues
            valid_residues = get_valid_residues(m, selected)
            if not valid_residues:
                continue
                
            # Heuristic for residue selection:
            # We want to choose a residue that doesn't block too many future options.
            # This is hard to compute. 
            # Strategy: Try the first valid residue. If that fails to improve, 
            # the backtracking will try the next modulus, but it won't try other residues for the same modulus 
            # unless we explicitly loop over them.
            
            # To keep branching factor manageable, we should limit the number of residues we try per modulus.
            # If we try all, branching factor is up to m (200). Depth is ~10-20. 200^20 is too much.
            # We need a strong pruning or limited residue choice.
            
            # Let's try a few promising residues:
            # 1. Smallest valid
            # 2. Largest valid
            # 3. Maybe one in the middle
            # Actually, let's just try the first one. If the search space is tight, 
            # the "first" greedy choice is often good.
            # But to improve on the seed (which is greedy), we need to explore alternatives.
            
            # Let's try up to 3 residues per modulus to balance depth and breadth.
            residues_to_try = valid_residues[:3]
            if len(valid_residues) > 3:
                residues_to_try.append(valid_residues[-1])
                
            for r in residues_to_try:
                selected.append((m, r))
                dfs(i + 1, selected, current_sum + 1.0 / m)
                selected.pop()
                
                # If we found a solution with this m, and the next modulus is very large, 
                # we might want to break early? No, let's let backtracking handle it.

    # Pre-compute suffix sums for loose upper bound
    suffix_sums = [0.0] * (len(moduli) + 1)
    for i in range(len(moduli) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + 1.0 / moduli[i]

    # Initial call
    dfs(0, [], 0.0)
    
    # Convert to required format
    return [[m, r] for m, r in best_solution]
