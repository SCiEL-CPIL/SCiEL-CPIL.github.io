"""Improved program for the suite task; uses backtracking feasibility and local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Candidates are 20 to 200
    candidates = list(range(20, 201))
    
    # Precompute gcds for all pairs? 
    # There are 181 candidates. 181*181/2 ~ 16k pairs. Fast enough.
    # But we only need it for pairs in the current set.
    
    def is_feasible(moduli):
        """
        Check if there exist residues r_i for moduli such that 
        r_i != r_j mod gcd(m_i, m_j) for all i != j.
        Returns True if feasible, False otherwise.
        """
        n = len(moduli)
        if n == 0:
            return True
        
        # Sort moduli by size? Smaller moduli have smaller domains, might be better to assign first.
        # But constraints depend on GCD.
        # Let's use a heuristic ordering: sort by modulus ascending.
        order = sorted(range(n), key=lambda i: moduli[i])
        
        # res[i] is the assigned residue for moduli[order[i]], or -1 if unassigned
        # Actually, let's just use an array indexed by the position in 'order'
        # assigned[i] = residue for the i-th modulus in 'order'
        assigned = [-1] * n
        
        # forbidden[i] is a set of residues forbidden for the i-th modulus in 'order'
        # We store forbidden residues as a set of integers.
        forbidden = [set() for _ in range(n)]
        
        # Precompute gcds for the pairs in 'order'
        # gcds[i][j] = gcd(moduli[order[i]], moduli[order[j]])
        gcds = [[1]*n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                g = gcd(moduli[order[i]], moduli[order[j]])
                gcds[i][j] = g
                gcds[j][i] = g

        def backtrack(k):
            """Try to assign residue to the k-th modulus in order."""
            if k == n:
                return True
            
            m_k = moduli[order[k]]
            
            # Find a valid residue for k
            # The domain is {0, ..., m_k-1} \ forbidden[k]
            # We iterate through possible residues
            # Optimization: If forbidden[k] is large, iterating through 0..m_k-1 is fine since m_k <= 200.
            
            for r in range(m_k):
                if r in forbidden[k]:
                    continue
                
                # Check if this assignment is consistent with previously assigned variables
                # It is, because we maintain forbidden sets.
                # Wait, forbidden[k] only contains residues forbidden by PREVIOUSLY assigned variables?
                # Yes, if we update correctly.
                
                # Assign r to k
                assigned[k] = r
                
                # Update forbidden sets for future variables (j > k)
                new_forbidden = []
                valid = True
                for j in range(k + 1, n):
                    g = gcds[k][j]
                    # r_k = r. Forbidden for j are values v such that v % g == r % g.
                    # We need to add these to forbidden[j].
                    # To revert, we need to know exactly what we added.
                    # Since sets don't track duplicates, and we might add the same value multiple times from different k's,
                    # simple addition is tricky for backtracking.
                    
                    # Alternative: Don't maintain dynamic forbidden sets.
                    # Instead, check constraints against all previously assigned variables directly.
                    # This is O(n^2) per assignment, but n is small (~20).
                    # Let's restart with the simpler, safer approach.
                    pass
                
                # Let's use the direct check approach for simplicity and correctness.
                # It's fast enough for n < 30.
                ok = True
                for j in range(k):
                    # Check constraint with j
                    r_j = assigned[j]
                    g = gcds[k][j]
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                
                if ok:
                    if backtrack(k + 1):
                        return True
                
                assigned[k] = -1
            
            return False
        
        return backtrack(0)

    def score_of(moduli):
        return sum(Fraction(1, m) for m in moduli)

    # Greedy Construction
    current_set = []
    # Sort candidates by 1/m descending (i.e., m ascending)
    sorted_candidates = sorted(candidates)
    
    for m in sorted_candidates:
        if time.time() > deadline - 10:
            break
            
        trial_set = current_set + [m]
        if is_feasible(trial_set):
            current_set = trial_set

    best_set = current_set[:]
    best_score = score_of(best_set)

    # Local Search / Hill Climbing
    # Try to improve by swapping or adding
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Try to add any candidate not in best_set
        for m in sorted_candidates:
            if time.time() > deadline - 5:
                break
            if m in best_set:
                continue
            trial_set = best_set + [m]
            if is_feasible(trial_set):
                s = score_of(trial_set)
                if s > best_score:
                    best_set = trial_set
                    best_score = s
                    improved = True
                    # Restart loop to check other additions?
                    # Break to restart the outer while loop
                    break
        
        if improved:
            continue

        # 2. Try to swap: remove one, add one
        # Iterate over all pairs (in, out)
        # To be efficient, only consider swaps that increase score: 1/m_out > 1/m_in => m_out < m_in
        for i, m_in in enumerate(best_set):
            if time.time() > deadline - 5:
                break
            for m_out in sorted_candidates:
                if time.time() > deadline - 5:
                    break
                if m_out in best_set:
                    continue
                if m_out >= m_in: # No score gain
                    continue
                
                trial_set = [m for j, m in enumerate(best_set) if j != i] + [m_out]
                if is_feasible(trial_set):
                    s = score_of(trial_set)
                    if s > best_score:
                        best_set = trial_set
                        best_score = s
                        improved = True
                        break
            if improved:
                break

    # 3. Try to remove one and add two? (Rarely beneficial due to 1/m decay, but possible)
    # Skip for now to save time.

    # Final check: Ensure we return a valid set.
    # If best_set is empty, return a trivial valid set
    if not best_set:
        # 20, 21, 22... are usually feasible
        for m in range(20, 201):
            if is_feasible([m]):
                best_set = [m]
                break
        if not best_set:
            best_set = [20]

    # Convert to required format [[m, r], ...]
    # We need to find the residues.
    # We can re-run the backtracking to get the residues, or modify is_feasible to return them.
    # For simplicity, let's create a function to find residues.
    
    def find_residues(moduli):
        n = len(moduli)
        if n == 0:
            return []
        
        order = sorted(range(n), key=lambda i: moduli[i])
        assigned = [-1] * n
        
        gcds = [[1]*n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                g = gcd(moduli[order[i]], moduli[order[j]])
                gcds[i][j] = g
                gcds[j][i] = g

        def backtrack(k):
            if k == n:
                return True
            
            m_k = moduli[order[k]]
            for r in range(m_k):
                ok = True
                for j in range(k):
                    r_j = assigned[j]
                    g = gcds[k][j]
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                
                if ok:
                    assigned[k] = r
                    if backtrack(k + 1):
                        return True
                    assigned[k] = -1
            
            return False
        
        if backtrack(0):
            # Map back to original indices
            res = [0] * n
            for i in range(n):
                res[order[i]] = assigned[i]
            return res
        else:
            return None # Should not happen if is_feasible returned True

    residues = find_residues(best_set)
    if residues is None:
        # Fallback, shouldn't happen
        residues = [0] * len(best_set)
        
    return [[best_set[i], residues[i]] for i in range(len(best_set))]

# EVOLVE-BLOCK-END
