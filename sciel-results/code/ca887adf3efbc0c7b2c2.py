import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 140.0  # 10s safety margin for 150s limit
    
    # The number of differences in one row is k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound for M:
    # We need total_diffs_needed distinct differences in {1, ..., M}.
    # So M >= total_diffs_needed.
    # Also, each row has max element <= M.
    # For (7,5), total_diffs = 7 * 15 = 105.
    M_lower = total_diffs_needed
    
    # Upper bound: Use a greedy construction to get a valid (though potentially suboptimal) solution.
    initial_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(initial_rows)
    best_rows = initial_rows
    
    # If the greedy solution is already at the lower bound, it's optimal.
    if best_M <= M_lower:
        return best_rows

    # Search for better M in [M_lower, best_M - 1]
    # Since we want the smallest M, we iterate upwards.
    # If we find a solution for M, it is the best for that M.
    # We stop at the first M that yields a solution.
    
    # To make the search effective, we prioritize M values close to the lower bound.
    # We will try each M in order.
    
    for M in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
        
        # Try to find a packing with max entry <= M
        rows = _solve_clique(n_rows, k, M, deadline)
        
        if rows is not None:
            best_rows = rows
            best_M = M
            break
            
    return best_rows

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    This provides a valid upper bound.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _solve_clique(n_rows, k, M, deadline):
    """
    Solves the packing problem for a fixed M using Maximum Clique on the compatibility graph.
    Nodes: Valid Sidon rows (subsets of size k+1 from {0..M} containing 0).
    Edges: Two nodes are connected if their difference sets are disjoint.
    We seek a clique of size n_rows.
    """
    # 1. Generate all valid Sidon rows within [0, M]
    # A row is a sorted tuple of k+1 integers starting with 0.
    # We generate combinations of k elements from 1..M.
    
    valid_rows = []
    row_diffs = []
    
    # Optimization: Precompute difference sets for all combinations?
    # C(M, k) can be large. For M=105, k=5, C(105,5) ~ 96M. Too many.
    # We must filter early.
    
    # However, for the "Maximum Clique" approach to work within time limits,
    # the graph must be manageable or we need very strong pruning.
    # Given the small n_rows (7) and specific structure, maybe we can prune candidates heavily?
    
    # Alternative: Instead of generating ALL valid rows, we can generate a subset of promising candidates.
    # But for a rigorous "structural change", let's try to generate all if possible, or use a smarter generation.
    
    # Let's check the size.
    # If M is small (e.g., < 50), C(M,5) is small.
    # If M is large (e.g., 100+), C(M,5) is huge.
    
    # For M=105, C(105, 5) = 96,560,844. This is too many to store in a list of tuples + sets in Python memory/time.
    
    # REVISION: The "Maximum Clique" on ALL valid rows is infeasible for M~100, k=5 due to node count.
    # We need a different structural approach or a restricted graph.
    
    # Let's reconsider the "Inspiration 1" (Simulated Annealing) vs "Inspiration 3" (Static diffs).
    # The prompt asks for a DIFFERENT strategy from the current (Backtracking/Random).
    # The current program uses random restarts with backtracking.
    
    # New Strategy Idea: **Constraint Programming / Arc Consistency**
    # Or: **Genetic Algorithm** on the rows?
    # Or: **Integer Linear Programming** (ILP) using scipy?
    # SciPy is available. `scipy.optimize` doesn't have a direct ILP solver, but `scipy.optimize.milp` exists in newer versions? 
    # Actually, standard SciPy might not have a robust ILP solver for this scale without external libs like PuLP (not allowed, stdlib only).
    
    # Let's look at **Bitset-based Backtracking with Forward Checking** more deeply, but structured differently.
    # The current code uses `random.sample` which is slow and doesn't exploit structure.
    
    # New Strategy: **Systematic Search with Bitmask Pruning**
    # Represent the set of used differences as a bitset (Python int or array of ints).
    # Represent each candidate row's differences as a bitset.
    # Compatibility is `(row_bits & used_bits) == 0`.
    # This is fast.
    
    # But we still need the candidate rows.
    # Instead of generating all C(M,k), we can generate them in a structured way?
    
    # Let's try **Simulated Annealing** (Inspiration 1) as it is a distinct family from Backtracking/Random Restart.
    # SA operates on a population of full solutions (sets of rows) and improves them via moves.
    
    # Plan for SA:
    # 1. Initialize with a valid solution (Greedy).
    # 2. Define moves:
    #    a. Replace one row with a new random valid row (that fits current used diffs).
    #    b. "Swap" differences between rows? Hard.
    #    c. Local search on a single row given others are fixed.
    # 3. Objective: Minimize max(entry).
    #    To do this, we can penalize large entries or try to shrink the max entry.
    #    Better: Fix M and try to find a valid packing. If found, decrease M.
    #    This combines the M-search with SA.
    
    # Let's implement SA for fixed M.
    
    def _sa_solve_fixed_M(M, deadline):
        """
        Uses Simulated Annealing to find a valid packing of n_rows rows with max entry <= M.
        """
        # 1. Generate a pool of candidate rows.
        # To avoid generating millions, we generate a random sample of valid Sidon rows.
        # If the sample is too small, we increase it.
        
        # For M=105, k=5, we need 7 rows.
        # We generate, say, 5000 random valid rows.
        
        candidates = []
        cand_diffs = []
        
        # Helper to check if a set of numbers is a Sidon set (distinct diffs)
        def is_sidon(vals):
            diffs = set()
            for i in range(len(vals)):
                for j in range(i+1, len(vals)):
                    d = vals[j] - vals[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
            return True, diffs

        # Generate candidates
        # We want to bias towards "compact" rows? Or random?
        # Random uniform from 1..M.
        
        attempts = 0
        target_candidates = 2000
        while len(candidates) < target_candidates and attempts < 20000:
            vals = random.sample(range(1, M + 1), k)
            vals.sort()
            row = (0,) + tuple(vals)
            ok, d_set = is_sidon(row)
            if ok:
                # Check if max entry is <= M (guaranteed by sample)
                candidates.append(row)
                cand_diffs.append(d_set)
            attempts += 1
            
        if len(candidates) < n_rows:
            return None

        # 2. SA Loop
        # State: list of indices into `candidates` representing the current rows.
        # We need to ensure the current rows are valid (disjoint diffs).
        
        # Initialization: Greedy selection from candidates
        current_rows = []
        current_diffs = set()
        current_indices = []
        
        # Sort candidates by max value to prefer compact ones initially?
        # Let's just pick randomly or greedily.
        
        # Greedy init from candidate pool
        for _ in range(n_rows):
            best_idx = -1
            best_penalty = float('inf')
            for i, row in enumerate(candidates):
                if i in current_indices:
                    continue
                # Check compatibility
                if not cand_diffs[i].isdisjoint(current_diffs):
                    continue
                # Heuristic: prefer rows with smaller max entry
                penalty = row[-1]
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_idx = i
            if best_idx == -1:
                return None
            current_indices.append(best_idx)
            current_rows.append(candidates[best_idx])
            current_diffs |= cand_diffs[best_idx]
            
        # Now we have a valid solution.
        # Objective: Minimize the maximum entry across all rows.
        # Current max entry:
        current_max = max(r[-1] for r in current_rows)
        
        # SA Parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        steps = 50000
        
        best_solution = current_rows[:]
        best_max = current_max
        
        for step in range(steps):
            if time.time() > deadline - 2:
                break
                
            # Pick a random row to replace
            idx_to_change = random.randint(0, n_rows - 1)
            old_row = current_rows[idx_to_change]
            old_diffs = cand_diffs[current_indices[idx_to_change]]
            
            # Remove it
            current_diffs.difference_update(old_diffs)
            
            # Pick a new random candidate
            new_idx = random.randint(0, len(candidates) - 1)
            new_row = candidates[new_idx]
            new_diffs = cand_diffs[new_idx]
            
            # Check if new row is compatible with the rest
            if not new_diffs.isdisjoint(current_diffs):
                # Revert
                current_diffs |= old_diffs
                continue
                
            # Accept
            current_rows[idx_to_change] = new_row
            current_indices[idx_to_change] = new_idx
            current_diffs |= new_diffs
            
            new_max = max(r[-1] for r in current_rows)
            
            # Update best
            if new_max < best_max:
                best_max = new_max
                best_solution = current_rows[:]
                
            # Annealing step:
            # We don't have a simple energy function that allows "worse" moves easily in a discrete set packing.
            # However, we can accept a move that increases max entry if it allows future improvements?
            # Hard to define.
            
            # Let's try a different move: "Swap" two rows? No, they are independent.
            # Let's try "Local Search" on the row itself.
            # Instead of picking a completely new random row, pick a row similar to the old one?
            
            # For now, this is a "Random Restart" style SA.
            # To improve, we can cool down and reduce the range of candidates?
            
            T *= cooling
            if T < T_min:
                T = T_min
                
            # Occasionally, re-seed or restart if stuck?
            # The current logic just keeps swapping.
            
        return best_solution

    # Try SA for the given M
    result = _sa_solve_fixed_M(M, deadline)
    if result is not None:
        return result
        
    return None

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True

