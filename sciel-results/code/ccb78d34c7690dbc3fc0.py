"""Improved covering solver: Greedy + iterative deletion + WalkSAT with clause weighting.
Focus on efficient data structures and smarter local search."""
import time
import random
import itertools
import numpy as np
from collections import defaultdict

# Deterministic seed for reproducibility of stochastic choices
_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    # Precompute all t-subsets and map each to an index for fast lookup
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets (candidate blocks) and the t-subsets they cover
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    # For each k-subset, list of t-subset indices it covers
    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)

    # For each t-subset, list of block indices that cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)

    # ---------------- Greedy initial solution with efficient tracking ----------------
    # Use a list to track uncovered t-subsets for O(1) access
    uncovered_list = list(range(n_t))
    uncovered_set = set(range(n_t))
    selected = set()
    
    # For each block, precompute its coverage as a set for fast intersection
    block_covers_set = [set(cov) for cov in block_covers]

    while uncovered_list:
        if time.time() > deadline - 3.0:
            break
        # Pick an uncovered t-subset (random for diversity)
        target = _RNG.choice(uncovered_list)
        
        # Choose the block covering the most uncovered t-subsets among candidates
        best_block = None
        best_gain = -1
        for bi in t_to_blocks[target]:
            if bi in selected:
                continue
            gain = len(block_covers_set[bi] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        if best_block is None:
            # Fallback: pick any block covering target
            for bi in t_to_blocks[target]:
                if bi not in selected:
                    best_block = bi
                    break
            if best_block is None:
                break
        
        selected.add(best_block)
        # Update uncovered
        newly_covered = block_covers_set[best_block] & uncovered_set
        for ti in newly_covered:
            uncovered_set.discard(ti)
        uncovered_list = [ti for ti in uncovered_list if ti in uncovered_set]

    # If still not covered, add brute-force blocks
    while uncovered_list:
        target = uncovered_list[0]
        for bi in t_to_blocks[target]:
            if bi not in selected:
                selected.add(bi)
                newly_covered = block_covers_set[bi] & uncovered_set
                for ti in newly_covered:
                    uncovered_set.discard(ti)
                uncovered_list = [ti for ti in uncovered_list if ti in uncovered_set]
                break

    # ---------------- Iterative deletion reduction ----------------
    # Maintain cover_count for efficiency
    cover_count = [0] * n_t
    for bi in selected:
        for ti in block_covers[bi]:
            cover_count[ti] += 1

    def try_remove_block(bi):
        """Try to remove block bi. Returns True if successful."""
        for ti in block_covers[bi]:
            cover_count[ti] -= 1
        # Check if any went to 0
        for ti in block_covers[bi]:
            if cover_count[ti] == 0:
                # Restore
                for ti2 in block_covers[bi]:
                    cover_count[ti2] += 1
                return False
        return True

    # Multiple passes of deletion until no more can be removed
    max_passes = 10
    for _pass in range(max_passes):
        if time.time() > deadline - 5.0:
            break
        removed_any = False
        for bi in list(selected):
            if try_remove_block(bi):
                selected.discard(bi)
                removed_any = True
        if not removed_any:
            break

    # ---------------- MILP phase (optional, time-limited) ----------------
    # Use MILP to potentially find a better solution
    milp_start = time.time()
    milp_budget = 35.0  # seconds for MILP
    milp_solution = None
    
    if time.time() < deadline - 50.0:  # Only if we have enough time left
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csr_matrix
            
            # Build constraint matrix A: n_t x n_k, A[j,i] = 1 if t-subset j is covered by block i
            A_rows = []
            A_cols = []
            A_data = []
            for ti in range(n_t):
                for bi in t_to_blocks[ti]:
                    A_rows.append(ti)
                    A_cols.append(bi)
                    A_data.append(1)
            
            A = csr_matrix((A_data, (A_rows, A_cols)), shape=(n_t, n_k))
            
            # Objective: minimize sum(x_i)
            c = np.ones(n_k)
            
            # Constraints: A @ x >= 1 for all t-subsets
            lb = np.ones(n_t)
            ub = np.full(n_t, np.inf)
            
            # Bounds: x_i in {0, 1}
            integrality = np.ones(n_k)
            bounds = Bounds(lb=np.zeros(n_k), ub=np.ones(n_k))
            
            constraints = LinearConstraint(A, lb, ub)
            
            # Solve with time limit
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                      options={"time_limit": milp_budget})
            
            milp_elapsed = time.time() - milp_start
            if res.status == 0 and res.x is not None:
                sol = res.x
                milp_solution = set(int(bi) for bi in range(n_k) if sol[bi] > 0.5)
                
                # Check if this is better than current
                if len(milp_solution) < len(selected):
                    selected = milp_solution
                    # Rebuild cover_count and do deletion reduction
                    cover_count = [0] * n_t
                    for bi in selected:
                        for ti in block_covers[bi]:
                            cover_count[ti] += 1
                    for _pass in range(5):
                        if time.time() > deadline - 10.0:
                            break
                        removed_any = False
                        for bi in list(selected):
                            if try_remove_block(bi):
                                selected.discard(bi)
                                removed_any = True
                        if not removed_any:
                            break
                    
        except Exception:
            pass  # Fall back to greedy solution if MILP fails
    
    # ---------------- WalkSAT-style local search with clause weighting ----------------
    # State: boolean array of length n_k, True = selected
    x = np.zeros(n_k, dtype=bool)
    for bi in selected:
        x[bi] = True
    
    selected_set = set(selected)
    selected_list = list(selected)  # Maintain list of selected blocks for efficiency

    # Use numpy for clause_sat_count for faster operations
    clause_sat_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected_list:
        clause_sat_count[block_covers[bi]] += 1

    unsat_clauses = np.where(clause_sat_count == 0)[0].tolist()
    
    # Track best solution
    best_selected = set(selected_list)
    best_size = len(selected_list)

    # Precompute block coverage as numpy arrays for faster intersection
    block_covers_np = [np.array(cov, dtype=np.int32) for cov in block_covers]

    def try_flip_add(bi):
        """Try to add block bi. Returns True if successful (no new unsat)."""
        nonlocal unsat_clauses, best_selected, best_size
        x[bi] = True
        clause_sat_count[block_covers_np[bi]] += 1
        selected_list.append(bi)
        selected_set.add(bi)
        # Update best if improved
        if len(selected_list) < best_size:
            best_size = len(selected_list)
            best_selected = set(selected_list)
        return True

    def try_flip_remove(bi):
        """Try to remove block bi. Returns True if successful (no new unsat)."""
        nonlocal unsat_clauses, best_selected, best_size
        x[bi] = False
        clause_sat_count[block_covers_np[bi]] -= 1
        # Check if any became 0
        if np.any(clause_sat_count[block_covers_np[bi]] == 0):
            # Restore
            x[bi] = True
            clause_sat_count[block_covers_np[bi]] += 1
            return False
        # Remove from selected_list
        try:
            selected_list.remove(bi)
        except ValueError:
            pass
        selected_set.discard(bi)
        # Update best if improved
        if len(selected_list) < best_size:
            best_size = len(selected_list)
            best_selected = set(selected_list)
        return True

    # WalkSAT loop with clause weighting
    remaining_time = deadline - 10.0 - time.time()
    final_selected = set(selected_list)  # Fallback
    
    if remaining_time >= 3.0:
        # Use clause weighting: weight each clause by 1/sat_count (harder clauses get higher weight)
        # This is a soft surrogate for the nonsmooth min objective
        
        # Geometric decay parameter for clause weights
        alpha = 0.95
        clause_weights = np.ones(n_t, dtype=np.float64)
        
        for step in range(int(remaining_time * 8000)):
            if time.time() > deadline - 10.0:
                break
            
            if unsat_clauses:
                # Pick a clause weighted by difficulty
                # Unsat clauses have weight 1, but we can prioritize based on how hard they are
                ti = _RNG.choice(unsat_clauses)
                candidates = t_to_blocks[ti]
                if candidates:
                    # Pick a candidate block, weighted by how many other unsat clauses it covers
                    # Use numpy for faster scoring
                    best_bi = None
                    best_score = -1
                    for bi in candidates:
                        if not x[bi]:
                            # Count how many unsat clauses this block would cover
                            # Use numpy for speed
                            covers = block_covers_np[bi]
                            score = np.sum(clause_sat_count[covers] == 0)
                            if score > best_score:
                                best_score = score
                                best_bi = bi
                    if best_bi is not None:
                        try_flip_add(best_bi)
                        # Rebuild unsat_clauses
                        unsat_clauses = np.where(clause_sat_count == 0)[0].tolist()
                        # Update clause weights with geometric decay
                        clause_weights = alpha * clause_weights
                        clause_weights[unsat_clauses] = 1.0
            else:
                # Satisfied. Try to reduce size.
                # Pick a random selected block and try to remove it
                if selected_list:
                    # Try multiple removals per iteration for efficiency
                    for _ in range(5):
                        if not selected_list:
                            break
                        bi = _RNG.choice(selected_list)
                        if try_flip_remove(bi):
                            pass
                        else:
                            pass
                else:
                    # Shouldn't happen if we started valid
                    break

        # Use best found
        final_selected = best_selected

    # Final deletion reduction on the best solution
    # Ensure final_selected is a set of indices
    if not isinstance(final_selected, set):
        final_selected = set(final_selected)
        
    # Re-verify coverage and do a final strict reduction
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in final_selected:
        cover_count[block_covers_np[bi]] += 1
            
    # Check validity first
    if np.any(cover_count == 0):
        # If invalid, fall back to the greedy solution which was verified
        final_selected = set(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for bi in final_selected:
            cover_count[block_covers_np[bi]] += 1

    # Final reduction pass - multiple iterations for better reduction
    for _pass in range(3):
        if time.time() > deadline - 2.0:
            break
        removed_any = False
        for bi in list(final_selected):
            if time.time() > deadline - 2.0:
                break
            # Try to remove bi
            cover_count[block_covers_np[bi]] -= 1
            
            if np.any(cover_count[block_covers_np[bi]] == 0):
                # Restore
                cover_count[block_covers_np[bi]] += 1
            else:
                final_selected.discard(bi)
                removed_any = True
        
        if not removed_any:
            break

    # Build output blocks
    blocks = [list(k_subs[bi]) for bi in sorted(final_selected)]
    
    return blocks
