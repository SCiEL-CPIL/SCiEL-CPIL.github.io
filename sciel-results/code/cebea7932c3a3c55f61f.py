"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    
    # Precompute gcd table for moduli
    gcd_table = {}
    for m1 in candidates:
        for m2 in candidates:
            if (m1, m2) not in gcd_table:
                gcd_table[(m1, m2)] = gcd(m1, m2)
    
    # Precompute weights
    weights = {m: 1.0 / m for m in candidates}
    
    # Sort moduli by weight descending (i.e., m ascending) for greedy priority
    sorted_mods = sorted(candidates, key=lambda m: -weights[m])
    
    # Upper bound calculation helper
    # Given a set of selected moduli and a set of available moduli, 
    # the max possible additional score is sum(1/m for m in available)
    # We can precompute suffix sums of sorted moduli weights for quick upper bounds
    # if we process in sorted order.
    
    # However, the constraint is complex. 
    # Let's use a Branch and Bound over the sorted moduli.
    # State: (current_index, selected_moduli, selected_residues)
    # At each step, we decide to include sorted_mods[i] or skip it.
    # If include, we must find a valid residue.
    
    best_score = 0.0
    best_solution = []
    
    # To speed up residue finding, we use backtracking.
    # For a given list of (m, r) pairs, check validity.
    
    def check_validity(pairs):
        """Check if list of (m, r) is pairwise disjoint."""
        n = len(pairs)
        for i in range(n):
            m1, r1 = pairs[i]
            for j in range(i + 1, n):
                m2, r2 = pairs[j]
                g = gcd_table[(m1, m2)]
                if r1 % g == r2 % g:
                    return False
        return True

    def find_residue(m, existing_pairs):
        """
        Find a valid residue r for modulus m given existing_pairs.
        Returns None if no valid residue exists.
        """
        for r in range(m):
            valid = True
            for m_e, r_e in existing_pairs:
                g = gcd_table[(m, m_e)]
                if r % g == r_e % g:
                    valid = False
                    break
            if valid:
                return r
        return None

    # Branch and Bound
    # We iterate through sorted_mods (smallest to largest).
    # We maintain a list of selected (m, r).
    # We maintain an index into sorted_mods.
    
    def bnb_search(idx, current_pairs, current_score):
        nonlocal best_score, best_solution
        
        if time.time() > deadline:
            return
            
        # Pruning: Upper bound
        # Theoretical max additional score is sum of weights of all unprocessed moduli.
        # We can precompute suffix sums of weights for sorted_mods.
        # suffix_sum[i] = sum(weights[m] for m in sorted_mods[i:])
        # Upper bound = current_score + suffix_sum[idx]
        # If upper_bound <= best_score, prune.
        
        if idx >= len(sorted_mods):
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_pairs)
            return
        
        # Calculate upper bound
        # To avoid recalculating sum every time, we pass a remaining_potential or use precomputed.
        # Let's precompute suffix sums of weights.
        
        # Try including sorted_mods[idx]
        m_new = sorted_mods[idx]
        r_new = find_residue(m_new, current_pairs)
        
        if r_new is not None:
            new_pairs = current_pairs + [(m_new, r_new)]
            new_score = current_score + weights[m_new]
            bnb_search(idx + 1, new_pairs, new_score)
            
        # Try excluding sorted_mods[idx]
        # Optimization: If we exclude, we can't include any modulus that conflicts with m_new?
        # No, exclusion just means we don't use m_new. Other moduli might still be usable.
        # But wait, if we exclude m_new, we might still be able to include other moduli.
        # The search space is 2^181 which is too large.
        
        # We need a better pruning strategy.
        # The "suffix sum" upper bound is very loose because it assumes we can take ALL remaining moduli.
        # In reality, many will conflict.
        
        # Alternative: Greedy + Local Search is often better for this specific structure if B&B is weak.
        # Let's stick to the current strategy but add a heuristic: 
        # If including m_new fails, we skip it.
        # If including m_new succeeds, we recurse.
        # We also recurse on skipping m_new.
        
        # To improve B&B, we can use a tighter bound.
        # For any remaining modulus m_j, if it conflicts with ALL currently selected residues 
        # for any of its conflicting moduli, it can't be added. 
        # This is expensive to check dynamically.
        
        # Let's try a different structural change: 
        # **Greedy with Lookahead and Backtracking on Conflicts**
        
        pass # Placeholder for actual B&B logic below

    # Precompute suffix sums of weights for the sorted list
    suffix_sums = [0.0] * (len(sorted_mods) + 1)
    for i in range(len(sorted_mods) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[sorted_mods[i]]
    
    def bnb_search_v2(idx, current_pairs, current_score):
        nonlocal best_score, best_solution
        
        if time.time() > deadline:
            return
        
        # Upper bound check
        if current_score + suffix_sums[idx] <= best_score:
            return
            
        if idx >= len(sorted_mods):
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_pairs)
            return
        
        m_new = sorted_mods[idx]
        
        # Option 1: Include m_new
        r_new = find_residue(m_new, current_pairs)
        if r_new is not None:
            new_pairs = current_pairs + [(m_new, r_new)]
            new_score = current_score + weights[m_new]
            bnb_search_v2(idx + 1, new_pairs, new_score)
            
        # Option 2: Exclude m_new
        # We can prune the exclusion branch if the upper bound of the remaining items
        # (excluding m_new) cannot beat the best.
        # But we already checked the upper bound at the start.
        # The upper bound for the exclusion branch is current_score + (suffix_sums[idx] - weights[m_new]).
        # If this is <= best_score, we don't need to explore exclusion.
        if current_score + (suffix_sums[idx] - weights[m_new]) > best_score:
            bnb_search_v2(idx + 1, current_pairs, current_score)

    # Initialize B&B
    bnb_search_v2(0, [], 0.0)
    
    # If B&B didn't find a good solution (or took too long and returned partial),
    # we might need a fallback or refinement.
    # However, B&B with this pruning is usually quite effective for this size.
    
    # Let's add a local search refinement on the best solution found by B&B
    # to potentially swap residues to allow more moduli or better combinations.
    
    if best_solution:
        # Local Search: Try to swap residues to free up space for other moduli?
        # Or just try to add any remaining moduli that were skipped?
        # The B&B explores all combinations of moduli (include/exclude).
        # So if a modulus was excluded, it was either because including it didn't lead to a better score
        # or because it was pruned.
        
        # One improvement: The B&B picks the *first* valid residue for m_new.
        # A different residue might allow more future moduli.
        # We can improve find_residue to be smarter, or run B&B multiple times with different residue choices.
        
        # Given time constraints, let's do a quick "improve residues" pass.
        # For each modulus in best_solution, try all other valid residues.
        # If a change allows adding a previously skipped modulus, do it.
        
        used_mods = set(m for m, r in best_solution)
        skipped_mods = [m for m in candidates if m not in used_mods]
        
        # Sort skipped by weight descending
        skipped_mods.sort(key=lambda m: -weights[m])
        
        improved = False
        for m_add in skipped_mods:
            if time.time() > deadline - 5:
                break
            # Try to find a residue for m_add given current best_solution
            r_add = find_residue(m_add, best_solution)
            if r_add is not None:
                # We can add it!
                new_solution = best_solution + [(m_add, r_add)]
                new_score = best_score + weights[m_add]
                best_solution = new_solution
                best_score = new_score
                used_mods.add(m_add)
                improved = True
                # Continue trying to add more
                # Note: Adding one might block others, but since we check validity, it's safe.
                # We might want to re-evaluate if adding a small modulus blocks a larger one?
                # No, we are adding in order of weight (smallest m first), so we are adding the most valuable ones.
                # Wait, the loop iterates skipped_mods sorted by weight desc.
                # So we are trying to add the next most valuable skipped modulus.
                # If we add it, we update used_mods.
                # We should probably restart the loop or continue carefully.
                # Since adding a modulus only restricts future additions, and we are going down the list of skipped,
                # it's safe to continue. But the list `skipped_mods` is static.
                # We should just break and restart or just continue.
                # Actually, if we add m_add, we might have skipped it in the B&B because of a bad residue choice.
                # Now we found a valid one.
                # Let's just continue the loop.
                
    # Final check: ensure no more can be added greedily
    # (The loop above does this, but let's be sure)
    used_mods = set(m for m, r in best_solution)
    remaining = [m for m in candidates if m not in used_mods]
    remaining.sort(key=lambda m: -weights[m])
    
    for m in remaining:
        if time.time() > deadline - 2:
            break
        r = find_residue(m, best_solution)
        if r is not None:
            best_solution.append((m, r))
            best_score += weights[m]
            used_mods.add(m)

    return [[m, r] for m, r in best_solution]
# EVOLVE-BLOCK-END
