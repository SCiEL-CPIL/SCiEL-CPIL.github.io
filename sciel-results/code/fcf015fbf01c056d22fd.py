import math
import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill
    
    def is_compatible(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def is_compatible_with_list(m, r, sel):
        for m2, r2 in sel:
            if not is_compatible(m, r, m2, r2):
                return False
        return True
        
    def find_residue(m, sel):
        """Find a valid residue r for modulus m given current selection sel."""
        for r in range(m):
            if is_compatible_with_list(m, r, sel):
                return r
        return -1

    def calc_score(sel):
        return sum(1.0 / m for m, r in sel)

    all_moduli = list(range(20, 201))
    best_score = 0.0
    best_selected = []
    
    # 1. Try multiple greedy initializations
    # Strategy A: Increasing m
    sel_a = []
    for m in all_moduli:
        r = find_residue(m, sel_a)
        if r != -1:
            sel_a.append((m, r))
    score_a = calc_score(sel_a)
    if score_a > best_score:
        best_score = score_a
        best_selected = list(sel_a)
        
    # Strategy B: Randomized Greedy (multiple runs)
    # We run this for a few seconds to find a good seed
    time_budget_for_init = 2.0
    while time.time() - start_time < time_budget_for_init:
        sel_r = []
        # Shuffle moduli, but bias towards smaller ones? 
        # Actually, just shuffling is a form of random greedy.
        mods = list(all_moduli)
        random.shuffle(mods)
        # To make it more effective, we can sort by m but with random noise
        # Or just pure random order
        for m in mods:
            r = find_residue(m, sel_r)
            if r != -1:
                sel_r.append((m, r))
        score_r = calc_score(sel_r)
        if score_r > best_score:
            best_score = score_r
            best_selected = list(sel_r)
            
    # 2. Local Search on the best candidate
    current = list(best_selected)
    current_score = best_score
    used_moduli = set(m for m, r in current)
    skipped = [m for m in all_moduli if m not in used_moduli]
    
    attempts = 0
    max_attempts = 500000
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        
        # Move 1: Try to add a skipped modulus
        if skipped and random.random() < 0.6:
            m = random.choice(skipped)
            r = find_residue(m, current)
            if r != -1:
                current.append((m, r))
                used_moduli.add(m)
                skipped.remove(m)
                current_score += 1.0 / m
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
        
        # Move 2: Swap (Remove one, Add one better)
        elif len(current) > 0 and skipped and random.random() < 0.8:
            # Pick a random selected modulus to remove
            idx = random.randint(0, len(current) - 1)
            m_out, r_out = current[idx]
            val_out = 1.0 / m_out
            
            # Find a skipped modulus with higher value
            candidates = [m for m in skipped if 1.0/m > val_out]
            if not candidates:
                continue
                
            m_in = random.choice(candidates)
            
            # Temporarily remove m_out
            current.pop(idx)
            used_moduli.remove(m_out)
            
            # Try to add m_in
            r_in = find_residue(m_in, current)
            if r_in != -1:
                current.append((m_in, r_in))
                used_moduli.add(m_in)
                skipped.remove(m_in)
                current_score = current_score - val_out + 1.0/m_in
                
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
            else:
                # Revert
                current.append((m_out, r_out))
                used_moduli.add(m_out)
                # m_in is still in skipped, so no change there
                current_score = current_score # It was not updated yet in the successful branch, 
                # wait, I updated current_score only if successful. 
                # If failed, current_score is unchanged from before the move?
                # No, I should revert the score calculation if I did it.
                # In my code above, I only update current_score if r_in != -1.
                # So if r_in == -1, current_score is still the old value. Correct.
                # But I need to make sure I didn't modify `current` score variable incorrectly.
                # The code above:
                # if r_in != -1: ... current_score = ...
                # else: ...
                # So if r_in == -1, current_score is untouched. Correct.
                
        # Move 3: Kick-out (Remove one, Try to add multiple)
        else:
            if len(current) > 0 and skipped:
                idx = random.randint(0, len(current) - 1)
                m_out, r_out = current[idx]
                val_out = 1.0 / m_out
                
                # Temporarily remove
                current.pop(idx)
                used_moduli.remove(m_out)
                
                # Try to add up to 3 skipped moduli with high value
                added = []
                gain = 0.0
                # Sort skipped by value desc to prioritize best ones
                potential_adds = sorted([m for m in skipped if 1.0/m > val_out], key=lambda x: 1.0/x, reverse=True)
                
                for m_in in potential_adds[:3]:
                    r_in = find_residue(m_in, current)
                    if r_in != -1:
                        current.append((m_in, r_in))
                        used_moduli.add(m_in)
                        added.append(m_in)
                        gain += 1.0 / m_in
                        if gain > val_out:
                            break # We have already gained enough, stop adding more to save time?
                            # Actually, we might want to add more if they fit easily?
                            # But if we break, we leave others in skipped.
                            # Let's just try to add as many as possible up to 3 to maximize chance.
                            # But if we add 3, and the 3rd one blocks the 4th which is better?
                            # Greedy add is fine.
                            
                if gain > val_out:
                    current_score = current_score - val_out + gain
                    # Remove added from skipped
                    for m in added:
                        skipped.remove(m)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(current)
                else:
                    # Revert
                    # Remove added ones
                    for m in added:
                        # Find and remove from current
                        for i, (mm, rr) in enumerate(current):
                            if mm == m:
                                current.pop(i)
                                used_moduli.remove(m)
                                break
                    # Re-add m_out
                    current.append((m_out, r_out))
                    used_moduli.add(m_out)
                    # current_score is unchanged because we only updated if gain > val_out
                    # So we are back to original state.
                    
    # Final Polish: Deterministic add
    for m in sorted(skipped, key=lambda x: 1.0/x, reverse=True):
        r = find_residue(m, current)
        if r != -1:
            current.append((m, r))
            current_score += 1.0 / m
            if current_score > best_score:
                best_score = current_score
                best_selected = list(current)
            # Don't remove from skipped here if we want to return best_selected, 
            # but we are done with search.
            
    return [[m, r] for m, r in best_selected]
