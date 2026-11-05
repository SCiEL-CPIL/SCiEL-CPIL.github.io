import time
import random
import itertools

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # The evaluator passes q, n, radius. We assume q=3, n=6, radius=1 based on the problem description.
    # However, the function signature is generic. We should handle the general case or assume the specific instance.
    # The prompt says: "The evaluator passes this task's alphabet size q, length n and covering radius."
    # And "Keep the function names and signatures".
    # The seed program hardcodes the tetracode logic for n=6, q=3.
    # I will implement the local search for the specific case q=3, n=6, radius=1, 
    # but fall back to the seed logic if parameters differ, or just assume the task is fixed.
    # Given the constraints and the "Football pool problem for 6 matches", it's safe to assume q=3, n=6, radius=1.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other parameters, though unlikely to be tested
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    # Precompute neighbors for each word (Hamming distance <= 1)
    # Word representation: integer 0 to 728
    
    # Convert integer to base-3 digits
    def int_to_digits(num, n):
        digits = [0] * n
        for i in range(n):
            digits[i] = num % 3
            num //= 3
        return digits

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        digits = int_to_digits(w, n)
        # Distance 0
        neighbors[w].append(w)
        # Distance 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    # Convert back to int
                    new_w = 0
                    power = 1
                    for i in range(n):
                        new_w += new_digits[i] * power
                        power *= 3
                    neighbors[w].append(new_w)
    
    # Initial code: Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            c = h + list(tail)
            # Convert to int
            w = 0
            power = 1
            for i in range(n):
                w += c[i] * power
                power *= 3
            initial_codewords.append(w)
            
    C = set(initial_codewords)
    cover_count = [0] * total_words
    for c in C:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    best_C = set(C)
    best_size = len(C)
    
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    # Local Search
    while time.time() < deadline:
        # Try to remove a codeword
        removed = False
        # Shuffle to avoid bias? Or just iterate.
        # To make it a random walk, we should pick a random order or random candidate.
        # But first, check if any is removable.
        removable_candidates = []
        for c in C:
            # Check if c is removable
            # c is removable if all its neighbors are covered by count > 1
            # Note: if we remove c, cover_count[nb] decreases by 1.
            # So we need cover_count[nb] >= 2 for all nb in neighbors[c]
            is_removable = True
            for nb in neighbors[c]:
                if cover_count[nb] < 2:
                    is_removable = False
                    break
            if is_removable:
                removable_candidates.append(c)
                
        if removable_candidates:
            # Remove one
            c_to_remove = random.choice(removable_candidates)
            C.remove(c_to_remove)
            for nb in neighbors[c_to_remove]:
                cover_count[nb] -= 1
            if len(C) < best_size:
                best_size = len(C)
                best_C = set(C)
            removed = True
            continue # Restart the loop to check for more removals
        
        if not removed:
            # Stuck. Perform a kick.
            # Add a random codeword not in C
            # To be efficient, we can pick from a list of all words
            # But C is small (~80), total 729.
            # We can just pick a random word from 0..728 and check if in C.
            # Or maintain a list of non-C words? Updating that list is O(N).
            # Just pick random.
            
            # Try a few kicks
            kick_success = False
            for _ in range(10): # Try 10 random kicks
                # Pick random word
                w = random.randint(0, total_words - 1)
                if w in C:
                    continue
                    
                # Add w
                C.add(w)
                for nb in neighbors[w]:
                    cover_count[nb] += 1
                    
                # Now try to remove a codeword (other than w if possible, to change state)
                # Actually, if we remove w, we are back to start.
                # We want to remove some c != w to change the set.
                # Or remove w and then we are stuck again.
                # So, look for a removable c in C.
                # If we find c != w, remove it.
                # If we only find w, remove w (undo).
                
                removable_now = []
                for c in C:
                    is_removable = True
                    for nb in neighbors[c]:
                        if cover_count[nb] < 2:
                            is_removable = False
                            break
                    if is_removable:
                        removable_now.append(c)
                        
                if removable_now:
                    # Prefer removing a codeword that is NOT w, to ensure state change
                    # If we remove w, we are back to the same set.
                    # If we remove c != w, we move to a new set of same size.
                    candidates = [c for c in removable_now if c != w]
                    if candidates:
                        c_to_remove = random.choice(candidates)
                    else:
                        c_to_remove = w # Only w is removable, so we must remove it (undo)
                        
                    C.remove(c_to_remove)
                    for nb in neighbors[c_to_remove]:
                        cover_count[nb] -= 1
                    
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = set(C)
                    kick_success = True
                    break # Restart main loop
                    
                else:
                    # No codeword is removable. Remove w to undo
                    C.remove(w)
                    for nb in neighbors[w]:
                        cover_count[nb] -= 1
                    # Continue trying other kicks
                    
            # If no kick worked, we are stuck in a local minimum for this set of kicks.
            # We can try a "double kick" or just restart with a different random seed?
            # Or just keep trying single kicks.
            
            # To escape deep local minima, we might need to allow temporary invalidity?
            # No, the "Add-then-Remove" keeps validity.
            # If we are stuck, it means no single addition allows a removal.
            # This implies the code is "irreducible".
            # We might need to add TWO codewords and remove ONE? (Size +1)
            # Or add ONE and remove TWO? (Size -1) -> This is the goal!
            # My current kick adds 1 and tries to remove 1.
            # If I add 1, I have k+1 codewords.
            # If I can remove 2, I get k-1.
            # So, after adding w, I should check if I can remove 2 codewords.
            # How? 
            # 1. Find a removable c1. Remove it. Now size k.
            # 2. Check if another c2 is removable. Remove it. Now size k-1.
            # This is just doing two removal steps.
            # My main loop already does "Remove one, then loop, then remove next".
            # So if adding w makes TWO codewords removable, the main loop will remove them one by one.
            # The only issue is if adding w makes NO codeword removable, but adding w + v makes one removable.
            # That's a 2-add kick.
            
            # For now, let's stick to 1-add kicks.
            # If stuck, we can try adding a random word, and if no removal, try adding ANOTHER random word?
            # That increases size by 2. Then try to remove 2?
            # This is getting complex.
            
            # Let's optimize the removal check.
            # The check "for nb in neighbors[c]: if cover_count[nb] < 2" is O(16).
            # C has ~80 elements. 80 * 16 = 1280 ops per check.
            # This is very fast.
            
            # If we are stuck, we might just be unlucky.
            # We can also try to swap: Remove c1, Add c2.
            # This is equivalent to Add c2, Remove c1.
            # My kick does Add c2, then Remove c1 (if c1 becomes removable).
            # So it covers swaps.
            
            # If we are stuck, it means for all c2 not in C, adding c2 does not make any c1 in C removable.
            # This is a strong local minimum.
            # We might need to break it by allowing a "bad" move?
            # E.g., remove a codeword even if it creates holes, then fix?
            # That's a different strategy.
            
            # Let's just keep trying random kicks.
            # We can also vary the random seed or just rely on the loop.
            
            # To speed up, if we are stuck for many iterations, we can try a "greedy add" of a word that covers the most "fragile" words?
            # No, all words are covered.
            
            # Let's just run the loop.
            
    # Convert best_C back to list of lists
    result = []
    for w in best_C:
        result.append(int_to_digits(w, n))
    return result
