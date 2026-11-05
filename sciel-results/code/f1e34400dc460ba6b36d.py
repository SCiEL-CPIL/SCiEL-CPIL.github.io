"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # The moduli are fixed in [20, 200].
    # To maximize sum(1/m), we want to include as many small moduli as possible.
    # Ideally, we include ALL moduli from 20 to 200.
    # The problem reduces to: Is there a set of residues r_m for m in [20, 200]
    # such that for all m1 != m2, r_m1 != r_m2 mod gcd(m1, m2)?
    
    # This is a CSP.
    # Variables: m in [20, 200]
    # Domains: D_m = [0, m-1]
    # Constraints: For m1, m2, r_m1 % gcd(m1, m2) != r_m2 % gcd(m1, m2)
    
    # Strategy:
    # 1. Try to solve the CSP for the full set M = {20, ..., 200}.
    # 2. If successful, return all classes. Score is maximal possible.
    # 3. If failed, we need to drop some moduli. 
    #    Which ones? Dropping a large m has small cost. Dropping a small m has large cost.
    #    So we should try to keep small moduli.
    #    However, the CSP solver with backtracking can handle "optional" variables 
    #    or we can use a greedy-add approach with a very fast incremental CSP check.
    
    # Let's first try to solve the full CSP.
    # Variable ordering is critical.
    # Most Constrained First (MCF): Variables with the smallest domain or most constraints.
    # Here, all domains are size m. Small m has smaller domain.
    # Also, moduli with many non-coprime neighbors are more constrained.
    
    all_moduli = list(range(20, 201))
    n = len(all_moduli)
    
    # Pre-compute GCDs
    gcd_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        m_i = all_moduli[i]
        for j in range(i, n):
            m_j = all_moduli[j]
            g = math.gcd(m_i, m_j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    # Degree of constraint: count of non-coprime neighbors (g > 1)
    # Actually, even if g=1, there is a constraint (always satisfied, so effectively 0).
    # So count neighbors with g > 1.
    degrees = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and gcd_matrix[i][j] > 1:
                degrees[i] += 1
                
    # Sort indices by (degree, m) descending? 
    # High degree means more constraints. Small m means smaller domain.
    # Let's sort by m ascending first, as small domains are tighter.
    # Or combine: score = degree * 1000 + m?
    # Let's try sorting by m ascending primarily.
    
    # Order for backtracking:
    # We want to assign variables that are most constrained first.
    # Let's compute a dynamic ordering or a static one.
    # Static: Sort by m ascending.
    order = list(range(n))
    order.sort(key=lambda i: all_moduli[i])
    
    # Reverse order? No, small m first.
    
    def solve_full_csp(time_limit=100.0):
        """
        Solves CSP for all moduli.
        Returns list of (m_idx, r) or None.
        """
        # Use recursion with forward checking?
        # Python recursion depth might be an issue for 181 variables? 
        # 181 is small for recursion limit (default 1000).
        
        assignment = [None] * n
        # domains[i] = set of available residues for all_moduli[i]
        # Initially range(m)
        # Forward checking: when assigning var i to r, update domains of neighbors.
        
        # To optimize, we can use a bitset or just lists.
        # Given m <= 200, lists are fine.
        
        # We will implement a backtracking solver that picks the next variable 
        # dynamically (MRV - Minimum Remaining Values) to prune faster.
        
        # However, dynamic MRV requires updating domains, which is O(N^2) per step.
        # With N=181, N^2 ~ 32k. 181 steps * 32k ~ 6M ops per path. 
        # Backtracking might explore many paths.
        
        # Alternative: Fixed order (m ascending) with simple consistency check.
        # This is faster per step but might explore more branches.
        # Let's try fixed order first with a time limit.
        
        def backtrack(k):
            if time.time() > start_time + time_limit:
                return False
            if k == n:
                return True
            
            m_idx = order[k]
            m = all_moduli[m_idx]
            
            # Try residues 0..m-1
            # Optimization: Check against already assigned variables.
            # assigned_vars are order[0..k-1]
            
            for r in range(m):
                valid = True
                for j in range(k):
                    m2_idx = order[j]
                    r2 = assignment[m2_idx]
                    g = gcd_matrix[m_idx][m2_idx]
                    if g > 1: # Only check if constraint exists
                        if (r - r2) % g == 0:
                            valid = False
                            break
                
                if valid:
                    assignment[m_idx] = r
                    if backtrack(k + 1):
                        return True
                    assignment[m_idx] = None
            
            return False

        if backtrack(0):
            return [(all_moduli[i], assignment[i]) for i in range(n)]
        return None

    # Try full CSP
    # Note: The full set might be infeasible. 
    # If it is, we need a fallback.
    # Fallback: Greedy addition with CSP check for the subset.
    
    result = solve_full_csp(time_limit=10.0)
    
    if result:
        # Verify and return
        # Score is sum(1/m)
        return result
    
    # If full set failed, we need to find a large subset.
    # Strategy: Start with empty set, add moduli in order of 1/m (m ascending).
    # For each candidate, check if it can be added to the current valid assignment.
    # If yes, add it. If no, skip it.
    # This is the same as the current program's greedy, BUT we can use the 
    # backtracking solver on the *current subset + new candidate* to see if 
    # we can adjust residues to make it fit.
    
    # However, re-solving the whole subset for each addition is expensive.
    # Better: Maintain the current assignment. When adding a new modulus, 
    # try to find a residue for it that is consistent with the current assignment.
    # If no such residue exists, we might need to change some existing residues.
    # This is hard.
    
    # Alternative Fallback: 
    # Since the full set is likely infeasible (or hard), let's try a 
    # "Maximum Weight Independent Set" on the graph of "hard conflicts"?
    # No, conflicts are not binary.
    
    # Let's try a simpler fallback: 
    # Randomized Greedy with Restarts.
    # 1. Pick a random permutation of moduli, weighted by 1/m? 
    #    No, we want to prioritize small m.
    # 2. Greedily add moduli. If a modulus cannot be added (no valid residue 
    #    w.r.t current assignment), skip it.
    # 3. To increase success rate, we can randomly shuffle the order of 
    #    residues tried for each modulus.
    
    best_score = 0.0
    best_result = None
    
    # Pre-compute score for any set? No, just sum 1/m.
    
    def try_greedy_construction(time_budget=1.0):
        # Order moduli by m ascending (highest value)
        # But randomize slightly to escape local optima?
        # Let's try a few different orderings.
        
        # Ordering 1: m ascending
        # Ordering 2: m ascending, but break ties randomly? (No ties in m)
        # Ordering 3: Random subset of top K moduli?
        
        # Let's just do m ascending with random residue selection.
        
        state = [] # list of (m, r)
        used_moduli = set()
        
        # We iterate through moduli in increasing order of m.
        # If we can find a valid r for m given state, we add it.
        
        for m in all_moduli:
            if time.time() > start_time + time_budget:
                break
            
            # Find valid r
            # Randomize order of r to find diverse solutions
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                valid = True
                for m2, r2 in state:
                    g = math.gcd(m, m2)
                    if g > 1:
                        if (r - r2) % g == 0:
                            valid = False
                            break
                if valid:
                    state.append((m, r))
                    used_moduli.add(m)
                    found = True
                    break
            
            # If not found, we skip m.
            # This might be suboptimal if a different r for an existing m2 
            # would have allowed m. But greedy is fast.
            
        score = sum(1.0 / m for m, r in state)
        return state, score
    
    # Run multiple greedy restarts
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        state, score = try_greedy_construction(time_budget=0.5)
        if score > best_score:
            best_score = score
            best_result = state
    
    # If we still have time, try to improve best_result by adding skipped moduli?
    # Or just return best_result.
    
    if best_result:
        return best_result
    else:
        # Fallback to single class
        return [[20, 0]]

# EVOLVE-BLOCK-END
