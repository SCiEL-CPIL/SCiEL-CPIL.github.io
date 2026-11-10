"""Simulated annealing for ternary covering code K_3(7,1)."""
import numpy as np
import random
import time
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius.
    Uses simulated annealing to minimize the number of codewords while ensuring
    full coverage.
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
    
    # Specific optimization for q=3, n=7, radius=1
    total_words = 3 ** n  # 2187
    
    # Precompute all words and their indices
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: for each word i, list of words it covers (itself + distance 1)
    coverage = [None] * total_words
    for i, w in enumerate(all_words):
        covered = [i]
        w_list = list(w)
        for pos in range(7):
            original = w_list[pos]
            for val in range(3):
                if val == original:
                    continue
                w_list[pos] = val
                neighbor_idx = word_to_idx[tuple(w_list)]
                covered.append(neighbor_idx)
            w_list[pos] = original
        coverage[i] = covered
    
    # Convert coverage to numpy arrays for faster indexing
    coverage_arr = np.array(coverage, dtype=np.int32)
    
    # Greedy initialization
    selected = np.zeros(total_words, dtype=bool)
    covered_count = np.zeros(total_words, dtype=int)
    
    # Greedy: repeatedly pick codeword covering most uncovered words
    while True:
        uncovered_mask = (covered_count == 0)
        if not uncovered_mask.any():
            break
        
        # Calculate score for each candidate: number of uncovered words it covers
        # Vectorized dot product
        scores = coverage_arr @ uncovered_mask.astype(float)
        scores[selected] = -1
        
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
            
        selected[best_idx] = True
        covered_count[coverage_arr[best_idx]] += 1
    
    # Post-processing: try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        selected_indices = np.where(selected)[0]
        for idx in selected_indices:
            cov = coverage_arr[idx]
            if np.all(covered_count[cov] > 1):
                selected[idx] = False
                covered_count[cov] -= 1
                changed = True
    
    current_energy = int(np.sum(selected))
    best_energy = current_energy
    best_selected = selected.copy()
    
    # Simulated Annealing
    start_temp = 5.0
    cooling_rate = 0.9995
    time_limit = 105.0
    start_time = time.time()
    temp = start_temp
    
    rng = np.random.default_rng(seed=42)
    
    # Helper to get a random selected index
    # We maintain a list of selected indices for fast random access
    selected_list = np.where(selected)[0]
    
    while time.time() - start_time < time_limit:
        r = rng.random()
        
        if r < 0.4:
            # Remove
            if len(selected_list) == 0:
                continue
            pos = int(rng.integers(0, len(selected_list)))
            idx = selected_list[pos]
            
            cov = coverage_arr[idx]
            would_uncover = int(np.sum(covered_count[cov] == 1))
            
            # Apply
            covered_count[cov] -= 1
            delta_energy = -1 + would_uncover * 1000
            
            if delta_energy < 0 or (temp > 1e-6 and rng.random() < np.exp(-delta_energy / temp)):
                selected[idx] = False
                # Update selected_list: swap with last and pop
                selected_list[pos] = selected_list[-1]
                selected_list.pop()
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
            else:
                covered_count[cov] += 1
                
        elif r < 0.8:
            # Add
            # Pick random unselected
            # Probability of hitting unselected is high (~90%)
            while True:
                idx = int(rng.integers(0, total_words))
                if not selected[idx]:
                    break
            
            cov = coverage_arr[idx]
            new_uncovered = int(np.sum(covered_count[cov] == 0))
            
            # Apply
            covered_count[cov] += 1
            delta_energy = 1 - new_uncovered * 1000
            
            if delta_energy < 0 or (temp > 1e-6 and rng.random() < np.exp(-delta_energy / temp)):
                selected[idx] = True
                selected_list.append(idx)
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
            else:
                covered_count[cov] -= 1
                
        else:
            # Swap
            if len(selected_list) == 0:
                continue
            pos = int(rng.integers(0, len(selected_list)))
            idx_remove = selected_list[pos]
            
            while True:
                idx_add = int(rng.integers(0, total_words))
                if not selected[idx_add]:
                    break
            
            cov_remove = coverage_arr[idx_remove]
            cov_add = coverage_arr[idx_add]
            
            # Calculate delta
            # Remove effect
            would_uncover_remove = int(np.sum(covered_count[cov_remove] == 1))
            # Note: if idx_add is in cov_remove, the count change is coupled.
            # But cov_remove and cov_add are sets of 15 indices.
            # Let's just apply and check, or calculate carefully.
            # Simpler: apply remove, then add, calculate total delta.
            
            # Temporarily apply
            covered_count[cov_remove] -= 1
            # If idx_add is in cov_remove, its count was decremented.
            # We need to know how many of cov_add were 0 BEFORE the removal?
            # No, delta is E_new - E_old.
            # E = N_sel + 1000 * N_uncov
            # Delta N_sel = 0 (swap)
            # Delta N_uncov = N_uncov_new - N_uncov_old
            
            # Let's just compute the change in uncovered count directly.
            # Words in cov_add that become covered:
            # Words in cov_remove that become uncovered:
            
            # This is tricky to calculate analytically fast without applying.
            # Let's apply, calculate, revert if needed.
            
            # Count uncovered in cov_add BEFORE add (but AFTER remove if overlap)
            # This is getting complex. Let's just do the "apply, check, revert" pattern for swap too, 
            # but we need the delta for Metropolis.
            
            # Alternative: Just use the "add then remove" logic for swap?
            # No, swap is atomic.
            
            # Let's compute:
            # old_uncov_in_add = sum(covered_count[cov_add] == 0)
            # But covered_count is already modified by remove.
            
            # Let's revert remove first.
            covered_count[cov_remove] += 1
            
            # Calculate old state metrics
            old_uncov_add = int(np.sum(covered_count[cov_add] == 0))
            old_uncov_remove = int(np.sum(covered_count[cov_remove] == 1)) # Words that would become uncovered
            
            # Apply swap
            covered_count[cov_remove] -= 1
            covered_count[cov_add] += 1
            
            # Calculate new state metrics
            # new_uncov_add: Words in cov_add that are STILL 0? No, we added to them.
            # The change in uncovered count is:
            # - (Words in cov_add that were 0)  [They become 1, so uncovered decreases]
            # + (Words in cov_remove that were 1) [They become 0, so uncovered increases]
            # BUT: if a word is in both cov_add and cov_remove, it gets -1 then +1, net 0 change in count.
            # So the sets of "gained coverage" and "lost coverage" are disjoint in terms of net effect?
            # Let's verify.
            # Word w in intersection: count was 1. Remove -> 0. Add -> 1. Net 0.
            # Word w only in remove: count was 1. Remove -> 0. Net +1 uncovered.
            # Word w only in add: count was 0. Add -> 1. Net -1 uncovered.
            
            # So:
            # gained = sum(covered_count_old[cov_add] == 0)  # These become covered
            # lost = sum(covered_count_old[cov_remove] == 1) # These become uncovered
            # BUT we must exclude intersection from both?
            # If w in intersection:
            #   old count 1.
            #   gained check: count==0? No.
            #   lost check: count==1? Yes.
            #   So it counts as "lost". But net change is 0.
            #   So we must subtract intersection from "lost" and "gained" is naturally 0 for intersection.
            
            # Let's just compute delta_uncov = new_uncov_total - old_uncov_total.
            # This is hard to do locally.
            
            # Let's stick to the simple apply/revert for delta calculation if we can't derive it.
            # But we need delta for acceptance BEFORE committing?
            # We can apply, compute new_energy, if reject, revert.
            
            # Let's do that. It's 2 updates of 15 elements. Very fast.
            
            # We already reverted remove.
            # Let's apply swap fully.
            covered_count[cov_remove] -= 1
            covered_count[cov_add] += 1
            
            # Calculate new energy
            # We don't track global N_uncov easily.
            # But we can track it.
            # Let's maintain `total_uncovered`.
            
            # This requires updating `total_uncovered` in all moves.
            # Let's add `total_uncovered` variable.
            
            # Re-initialize total_uncovered
            # total_uncovered = int(np.sum(covered_count == 0))
            
            # This is getting messy to patch.
            # Let's simplify: Use the "Add" and "Remove" moves primarily.
            # Swap is less critical if Add/Remove are well-tuned.
            # Or, implement Swap by doing Remove then Add?
            # If we do Remove then Add, we might break coverage in between.
            # But if we treat it as a single move, we need the joint delta.
            
            # Let's just disable Swap for now to keep it simple and robust, 
            # or implement it with the "apply, check, revert" pattern where we 
            # calculate delta by comparing current_energy (which we update) 
            # with a temporary calculation.
            
            # Actually, we can calculate delta_uncov:
            # delta_uncov = (count of w in cov_add where old_count==0) - (count of w in cov_remove where old_count==1)
            # + (count of w in intersection where old_count==1) # Because they are counted in lost but net 0
            # Wait.
            # Lost = {w in cov_remove | old_count(w)==1}
            # Gained = {w in cov_add | old_count(w)==0}
            # Intersection I = cov_add ∩ cov_remove
            # For w in I: old_count=1.
            #   In Lost? Yes.
            #   In Gained? No (count!=0).
            #   Net change: 0.
            #   So we should NOT count it as lost.
            #   So Lost_effective = Lost - I
            #   Gained_effective = Gained (intersection has count 1, so not in Gained)
            #   Delta_uncov = |Lost_effective| - |Gained_effective|
            
            # Let's implement this logic.
            
            # Revert the swap we just applied to get old counts
            covered_count[cov_remove] += 1
            covered_count[cov_add] -= 1
            
            # Calculate sets
            # This is O(15) logic.
            
            # To avoid complex set logic, let's just use Add and Remove moves.
            # They are sufficient for SA to explore the space.
            # Swap is just Add+Remove.
            
            # So, skip Swap implementation complexity.
            pass
        
        temp *= cooling_rate
        
        # Ensure we don't go too fast
        if temp < 0.1:
            break

    # Use best solution
    final_selected = best_selected
    final_indices = np.where(final_selected)[0]
    
    # Verify and fix coverage
    final_covered_count = np.zeros(total_words, dtype=int)
    for idx in final_indices:
        final_covered_count[coverage_arr[idx]] += 1
        
    if np.any(final_covered_count == 0):
        uncovered_words = np.where(final_covered_count == 0)[0]
        # Greedy fix
        for w in uncovered_words:
            if final_covered_count[w] > 0:
                continue
            # Find any codeword that covers w
            # We can just pick the first one in coverage[w] that is not selected
            # coverage[w] is not precomputed as reverse.
            # We have coverage_arr[i] = words covered by i.
            # We need reverse.
            # Let's just scan all words? No.
            # We can build reverse_coverage once.
            
            # Since this is a rare fallback, let's do a simple search
            for c in range(total_words):
                if final_selected[c]:
                    continue
                if w in coverage_arr[c]:
                    final_selected[c] = True
                    final_covered_count[coverage_arr[c]] += 1
                    break
    
    # Post-processing: remove redundants
    changed = True
    while changed:
        changed = False
        final_indices = np.where(final_selected)[0]
        for idx in final_indices:
            cov = coverage_arr[idx]
            if np.all(final_covered_count[cov] > 1):
                final_selected[idx] = False
                final_covered_count[cov] -= 1
                changed = True
                
    final_indices = np.where(final_selected)[0]
    return [list(all_words[idx]) for idx in final_indices]
