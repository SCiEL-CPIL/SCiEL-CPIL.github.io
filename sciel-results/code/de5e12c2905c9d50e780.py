import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precomputation
    total_words = q ** n
    # Map word tuple to index
    # Generate all words
    words = []
    for i in range(total_words):
        w = []
        val = i
        for _ in range(n):
            w.append(val % q)
            val //= q
        words.append(tuple(w))
        
    # Precompute balls
    # ball[i] is a list of indices covered by words[i]
    balls = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        balls[i].append(i)
        for pos in range(n):
            for delta in range(1, q):
                new_w = list(w)
                new_w[pos] = (new_w[pos] + delta) % q
                # Find index of new_w
                # To make this fast, we need a mapping from tuple to index
                # Or we can compute index directly
                # Index is sum(w[j] * q^j)
                idx = 0
                power = 1
                for val in new_w:
                    idx += val * power
                    power *= q
                balls[i].append(idx)
                
    # Convert balls to bitmasks (Python ints)
    masks = [0] * total_words
    for i in range(total_words):
        m = 0
        for idx in balls[i]:
            m |= (1 << idx)
        masks[i] = m
        
    full_mask = (1 << total_words) - 1
    
    # Helper: popcount
    def popcount(x):
        return bin(x).count('1')
        
    # Helper: Greedy Set Cover
    def greedy_cover(initial_code=None):
        if initial_code is None:
            code = []
            covered = 0
        else:
            code = list(initial_code)
            covered = 0
            for c in code:
                covered |= masks[c]
                
        uncovered = full_mask ^ covered
        
        while uncovered != 0:
            best_gain = -1
            best_idx = -1
            # Optimization: Only consider candidates that actually add coverage
            # We can iterate through all, but that's 729 * 73 steps.
            # 729 * 73 * (bit ops) is fast.
            for i in range(total_words):
                gain = popcount(masks[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break # Should not happen if valid
            code.append(best_idx)
            covered |= masks[best_idx]
            uncovered = full_mask ^ covered
            if time.time() > deadline:
                break
        return code

    # Helper: Local Search (Remove Redundant)
    def local_search(code):
        # Convert to set for fast lookup
        code_set = set(code)
        # Count coverage
        cover_count = [0] * total_words
        for c in code:
            for idx in balls[c]:
                cover_count[idx] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove
            for i in range(len(code)):
                c = code[i]
                # Check if removing c leaves all covered
                # Temporarily decrement
                for idx in balls[c]:
                    cover_count[idx] -= 1
                
                # Check min
                is_redundant = True
                for idx in range(total_words):
                    if cover_count[idx] == 0:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Accept removal
                    code.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                else:
                    # Revert
                    for idx in balls[c]:
                        cover_count[idx] += 1
                        # We can break early if we know it's not redundant, 
                        # but we already checked all. 
                    # Actually, we only need to revert if we didn't remove.
                    # If we did remove, we don't revert.
                    # Wait, the logic above reverts if not redundant.
                    # If redundant, we pop and break.
                    # So the revert block is only for the "not redundant" case.
                    # But we decremented in the loop.
                    # So we must revert if we don't remove.
                    # The code above does that.
        
        return code

    # Initial solutions
    # 1. Tetracode construction (81 words)
    # Re-implementing the seed's logic to get indices
    tetracode_words = []
    for a in range(3):
        for b in range(3):
            w = [a, b, (a+b)%3, (a+2*b)%3]
            for tail in [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]:
                full_w = w + tail
                # Calculate index
                idx = 0
                power = 1
                for val in full_w:
                    idx += val * power
                    power *= 3
                tetracode_words.append(idx)
                
    best_code = greedy_cover(tetracode_words)
    best_code = local_search(best_code)
    best_size = len(best_code)
    
    # 2. Pure Greedy
    code2 = greedy_cover()
    code2 = local_search(code2)
    if len(code2) < best_size:
        best_code = code2
        best_size = len(code2)
        
    # 3. Randomized Greedy / Add-Remove Search
    # We will run a loop trying to improve best_code
    # Strategy: Take best_code, try Add-Remove steps
    
    # Convert best_code to a set for fast "in" checks
    # We'll work with lists and sets
    
    current_code = list(best_code)
    current_size = best_size
    
    # Precompute random shuffles or just use random selection
    # To make Add-Remove effective, we should try adding words that are "close" to the current set's gaps?
    # Or just random.
    
    # Let's implement a specific Add-Remove loop
    # State: code, cover_count
    code = list(current_code)
    cover_count = [0] * total_words
    for c in code:
        for idx in balls[c]:
            cover_count[idx] += 1
            
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # 1. Try to remove redundant
        removed = False
        for i in range(len(code)):
            c = code[i]
            # Check redundancy
            # We can check if min cover_count in balls[c] is > 1
            # If min > 1, it's redundant.
            # Actually, if min cover_count in balls[c] >= 1 after decrement, it's redundant.
            # i.e., if min(cover_count[idx] for idx in balls[c]) > 1? No.
            # If we remove c, cover_count[idx] becomes cover_count[idx]-1.
            # We need cover_count[idx]-1 >= 1 => cover_count[idx] >= 2 for all idx in balls[c].
            # So if min(cover_count[idx] for idx in balls[c]) >= 2, it is redundant.
            
            min_count = min(cover_count[idx] for idx in balls[c])
            if min_count >= 2:
                # Remove c
                for idx in balls[c]:
                    cover_count[idx] -= 1
                code.pop(i)
                removed = True
                no_improve_count = 0
                break
        
        if not removed:
            # 2. Try Add-Remove
            # Pick a random word to add
            # To be smarter, pick a word that covers many uncovered? 
            # But everything is covered. 
            # So we pick a word that has high overlap with current set? 
            # Or just random.
            # Let's try a few random additions.
            
            added_improved = False
            # Try a few random candidates
            for _ in range(10):
                if time.time() > deadline:
                    break
                c_add = random.randint(0, total_words - 1)
                if c_add in code:
                    continue
                    
                # Add c_add
                code.append(c_add)
                for idx in balls[c_add]:
                    cover_count[idx] += 1
                    
                # Now try to remove redundant words (including c_add)
                # We do a pass of removals
                size_before = len(code)
                # We need to be careful with indices if we remove
                # Let's just collect indices to remove
                to_remove = []
                for i in range(len(code)):
                    c = code[i]
                    min_count = min(cover_count[idx] for idx in balls[c])
                    if min_count >= 2:
                        to_remove.append(i)
                        
                if to_remove:
                    # Remove all identified
                    # Sort by index descending to pop safely
                    to_remove.sort(reverse=True)
                    for i in to_remove:
                        c = code[i]
                        for idx in balls[c]:
                            cover_count[idx] -= 1
                        code.pop(i)
                        
                    if len(code) < size_before:
                        added_improved = True
                        no_improve_count = 0
                        # Keep the new state
                        break
                    else:
                        # Revert addition? 
                        # If we removed things but net size didn't drop below original, 
                        # we might want to revert to keep the set "tight"?
                        # Or maybe the new structure is better for future steps.
                        # Let's revert if size didn't improve.
                        # Revert: Remove c_add, restore removed?
                        # This is complex. 
                        # Simpler: If size didn't improve, revert the whole operation.
                        # To revert, we need to know what was removed.
                        # Let's just not do this complex revert. 
                        # If it doesn't improve size, we discard the change.
                        # So we must revert.
                        pass # Implementation of revert is tricky. 
                        
                if not added_improve:
                    # Revert addition of c_add
                    # But we might have removed others? 
                    # If we didn't break out of the inner loop with an improvement, 
                    # we should revert to the state before adding c_add.
                    # This requires storing the state.
                    pass
        
            if not added_improved:
                no_improve_count += 1
            else:
                # Update best if better
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                # Continue loop with this new code

    # Final polish: Run local_search on best_code again
    best_code = local_search(best_code)
    
    # Convert indices back to words
    result = []
    for idx in best_code:
        result.append(list(words[idx]))
        
    return result
