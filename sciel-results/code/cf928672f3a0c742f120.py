"""Improved covering solver: Greedy + iterative deletion + WalkSAT with clause weighting.
Focus on efficient data structures and smarter local search."""
import time
import random
import itertools
import numpy as np
from collections import defaultdict, deque

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

    # Precompute block coverage as numpy arrays for fast operations
    block_covers_np = [np.array(cov, dtype=np.int32) for cov in block_covers]

    # ---------------- Greedy initial solution with efficient tracking ----------------
    # Use a list to track uncovered t-subsets
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
        # Rebuild uncovered_list only when it shrinks significantly
        if len(uncovered_set) < len(uncovered_list) * 0.9:
            uncovered_list = [ti for ti in uncovered_list if ti in uncovered_set]
        else:
            # Filter in place for efficiency
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
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    def try_remove_block(bi):
        """Try to remove block bi. Returns True if successful."""
        cover_count[block_covers_np[bi]] -= 1
        # Check if any went to 0
        if np.any(cover_count[block_covers_np[bi]] == 0):
            # Restore
            cover_count[block_covers_np[bi]] += 1
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

    # ---------------- WalkSAT-style local search with clause weighting and 2-swap ----------------
    # State: boolean array of length n_k, True = selected
    x = np.zeros(n_k, dtype=bool)
    for bi in selected:
        x[bi] = True
    
    selected_set = set(selected)
    selected_list = list(selected)  # Maintain list of selected blocks for efficiency

    # Use numpy for clause_sat_count for faster operations
    clause_sat_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected_list:
        clause_sat_count[block_covers_np[bi]] += 1

    unsat_clauses = np.where(clause_sat_count == 0)[0].tolist()
    
    # Track best solution
    best_selected = set(selected_list)
    best_size = len(selected_list)

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

    def try_swap(remove_bi, add_bi):
        """Try to swap: remove remove_bi and add add_bi. Returns True if successful."""
        nonlocal unsat_clauses, best_selected, best_size
        if remove_bi == add_bi:
            return False
        if not x[remove_bi] or x[add_bi]:
            return False
        
        # Compute net effect on clause_sat_count
        rem_covers = block_covers_np[remove_bi]
        add_covers = block_covers_np[add_bi]
        
        # Temporarily apply the swap
        clause_sat_count[rem_covers] -= 1
        clause_sat_count[add_covers] += 1
        
        # Check if any clause went to 0
        if np.any(clause_sat_count[rem_covers] == 0):
            # Restore
            clause_sat_count[rem_covers] += 1
            clause_sat_count[add_covers] -= 1
            return False
        
        # Swap is valid
        x[remove_bi] = False
        x[add_bi] = True
        try:
            selected_list.remove(remove_bi)
        except ValueError:
            pass
        selected_list.append(add_bi)
        selected_set.discard(remove_bi)
        selected_set.add(add_bi)
        
        # Update best if improved (swap keeps same size, so no improvement in size)
        # But it might enable future improvements
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
        
        # Precompute for faster access
        selected_blocks_list = selected_list[:]
        
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
                    
                    # Occasionally try a 2-swap to escape local optima
                    if selected_list and len(selected_list) > 1 and _RNG.random() < 0.1:
                        # Pick a random selected block to potentially remove
                        remove_bi = _RNG.choice(selected_list)
                        # Find a block that could replace it
                        # Look at clauses that only this block covers
                        rem_covers = block_covers_np[remove_bi]
                        # Find clauses with sat_count == 1
                        unique_clauses = rem_covers[clause_sat_count[rem_covers] == 1]
                        if len(unique_clauses) > 0:
                            # Pick one of the unique clauses
                            target_clause = _RNG.choice(unique_clauses.tolist())
                            # Find blocks that cover this clause
                            candidates = t_to_blocks[target_clause]
                            for add_bi in candidates:
                                if add_bi != remove_bi and not x[add_bi]:
                                    if try_swap(remove_bi, add_bi):
                                        break
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
