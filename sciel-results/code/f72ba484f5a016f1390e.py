import itertools
import time
import random
import sys

def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: use all words (worst case)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=7, radius=1
    # Precompute neighbors for all possible words in the space
    total_words = q ** n
    
    def word_to_int(word):
        result = 0
        for w in word:
            result = result * q + w
        return result

    def int_to_word(val):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word

    # Precompute neighbors for every integer in the space
    # neighbors[int_val] = list of int_vals within Hamming distance 1
    neighbors = [[] for _ in range(total_words)]
    for w_int in range(total_words):
        w = int_to_word(w_int)
        neigh = [w_int]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val != orig:
                    w[i] = new_val
                    neigh.append(word_to_int(w))
            w[i] = orig
        neighbors[w_int] = neigh

    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    code = list(initial_code)
    code_ints = [word_to_int(c) for c in code]
    code_set = set(code_ints)
    
    # Coverage counts
    covered_count = [0] * total_words
    for c_int in code_ints:
        for nb in neighbors[c_int]:
            covered_count[nb] += 1
            
    # Helper to check if code is valid
    def is_valid():
        # Check if all words are covered
        # min(covered_count) >= 1
        return min(covered_count) >= 1

    # Helper to find redundant codewords
    # A codeword c is redundant if for all w in N(c), covered_count[w] >= 2
    def find_redundant():
        redundant_indices = []
        for idx, c_int in enumerate(code_ints):
            is_red = True
            for nb in neighbors[c_int]:
                if covered_count[nb] <= 1:
                    is_red = False
                    break
            if is_red:
                redundant_indices.append(idx)
        return redundant_indices

    # Main optimization loop
    # Strategy:
    # 1. Remove redundant codewords.
    # 2. If stuck, try "swap and remove" moves.
    #    For each codeword c, and each neighbor c':
    #    If c' not in code:
    #      Simulate adding c', removing c.
    #      If valid, check if any other codeword becomes redundant.
    #      If yes, perform move (size - 1).
    #      If no, calculate "potential" (sum of min(0, count-1) or similar) and swap if potential increases.
    
    best_size = len(code)
    
    while time.time() < deadline:
        # Phase 1: Remove all redundant codewords
        removed_any = True
        while removed_any:
            removed_any = False
            red_indices = find_redundant()
            for idx in red_indices:
                c_int = code_ints[idx]
                for nb in neighbors[c_int]:
                    covered_count[nb] -= 1
                code.pop(idx)
                code_ints.pop(idx)
                code_set.remove(c_int)
                removed_any = True
                if len(code) < best_size:
                    best_size = len(code)
            
        if len(code) <= 156: # Lower bound
            break
            
        # Phase 2: Local Search for improvements
        improved = False
        # Shuffle codewords to break symmetry
        random.shuffle(code)
        random.shuffle(code_ints)
        
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
                
            c_int = code_ints[i]
            c_neighbors = neighbors[c_int]
            
            # Try all neighbors of c_int
            # We only consider neighbors that are not already in the code
            for c_prime_int in c_neighbors:
                if c_prime_int == c_int:
                    continue
                if c_prime_int in code_set:
                    continue
                
                # Simulate removing c_int and adding c_prime_int
                # Delta coverage
                # We need to check if the resulting coverage is valid (min count >= 1)
                # And if it allows removing another codeword.
                
                # Efficient check:
                # Words that become uncovered if we remove c_int:
                # These are words w in N(c_int) such that covered_count[w] == 1.
                # If c_prime_int does not cover any of these, the move is invalid (unless we don't remove c_int, but we are swapping).
                # Actually, we are replacing c_int with c_prime_int.
                # So for w in N(c_int) \ N(c_prime_int), count decreases by 1.
                # For w in N(c_prime_int) \ N(c_int), count increases by 1.
                # For w in N(c_int) \cap N(c_prime_int), count stays same.
                
                # Let's just do the full simulation for simplicity, as n=7 is small.
                # But we can shortcut:
                # Check if removing c_int leaves any hole.
                # A hole is a word w in N(c_int) with count 1, such that w is NOT in N(c_prime_int).
                
                valid_swap = True
                # Check words only covered by c_int
                for w in c_neighbors:
                    if covered_count[w] == 1:
                        # w is critical for c_int
                        # c_prime_int must cover w
                        if w not in neighbors[c_prime_int]:
                            valid_swap = False
                            break
                            
                if not valid_swap:
                    continue
                
                # Now the swap is valid (coverage remains >= 1).
                # Check if this swap makes any OTHER codeword redundant.
                # We need to compute the new coverage counts for neighbors of c_int and c_prime_int.
                # Let's compute the change in "redundancy potential".
                
                # Potential = sum over all w of max(0, count[w] - 1)
                # If we can increase potential, we are closer to removing more codewords.
                
                # However, checking if ANY codeword becomes redundant is more direct.
                # A codeword d becomes redundant if for all w in N(d), new_count[w] >= 2.
                
                # To avoid O(N^2) for each swap, we can just perform the swap if it increases the number of "redundant" codewords.
                # Or simply: if the swap allows removing at least one other codeword, do it.
                
                # Let's check if any d in code (d != i) is redundant after swap.
                # This requires knowing new counts.
                
                # Optimization: Only check codewords d that are "close" to c_int or c_prime_int?
                # Actually, the change in counts only affects words in N(c_int) U N(c_prime_int).
                # So only codewords d that have neighbors in this set can change their redundancy status.
                
                affected_words = set(c_neighbors) | set(neighbors[c_prime_int])
                
                # Identify candidate codewords to check for redundancy
                candidates = set()
                for d_idx, d_int in enumerate(code_ints):
                    if d_idx == i:
                        continue
                    # If d has any neighbor in affected_words, it might be affected
                    # Actually, we just need to check if d is redundant in the NEW state.
                    # Let's just check all d? That's O(N^2). N is ~200. 200*200 = 40k.
                    # With 20 neighbors per c, that's 800k checks per outer loop iteration?
                    # Too slow.
                    
                    # Better: Just check if the swap increases the global "redundancy score".
                    # Score = sum(covered_count[w] - 1 for w in range(total_words) if covered_count[w] > 1)
                    # If Score increases, it's likely a good move.
                    
                    pass # Skip this complex check for now, try simpler heuristic
                
                # Simpler Heuristic:
                # Perform the swap.
                # Then run a quick "remove redundant" pass.
                # If size decreases, keep it.
                # If size doesn't decrease, but "potential" (sum of counts-1) increases, keep it.
                # Otherwise, revert.
                
                # Apply Swap
                for w in c_neighbors:
                    covered_count[w] -= 1
                for w in neighbors[c_prime_int]:
                    covered_count[w] += 1
                    
                old_potential = sum(max(0, covered_count[w] - 1) for w in range(total_words))
                
                # Try to remove redundant codewords
                red_indices = find_redundant()
                
                if red_indices:
                    # We can remove at least one codeword
                    # Remove one (or all)
                    for idx in red_indices:
                        # Note: indices might shift if we remove multiple, but find_redundant returns current indices
                        # Since we are in a simulation, let's just remove one to see if size drops
                        d_int = code_ints[idx]
                        for w in neighbors[d_int]:
                            covered_count[w] -= 1
                        code.pop(idx)
                        code_ints.pop(idx)
                        code_set.remove(d_int)
                        improved = True
                        break # Remove one, then restart outer loop? Or continue?
                    # If we removed one, we should restart the main loop to clean up
                    # But we are in the middle of a swap simulation.
                    # Let's commit the swap and the removal.
                    # We need to update code_set for c_prime_int
                    code_set.remove(c_int)
                    code_set.add(c_prime_int)
                    code[i] = int_to_word(c_prime_int) # Update word representation? Not strictly needed for logic, but good for return
                    code_ints[i] = c_prime_int
                    
                    if len(code) < best_size:
                        best_size = len(code)
                    
                    # Restart main loop
                    break 
                else:
                    # No immediate removal. Check potential.
                    new_potential = sum(max(0, covered_count[w] - 1) for w in range(total_words))
                    
                    if new_potential > old_potential:
                        # Accept swap
                        code_set.remove(c_int)
                        code_set.add(c_prime_int)
                        code[i] = int_to_word(c_prime_int)
                        code_ints[i] = c_prime_int
                        improved = True
                    else:
                        # Revert
                        for w in neighbors[c_prime_int]:
                            covered_count[w] -= 1
                        for w in c_neighbors:
                            covered_count[w] += 1
                        
            if improved:
                break # Restart outer loop
        
        if not improved:
            # Try random restarts or perturbations if stuck
            # Pick a random codeword, remove it, add a random word that covers the most holes
            if len(code) > 0:
                idx = random.randint(0, len(code) - 1)
                c_int = code_ints[idx]
                
                # Remove c_int
                for w in c_neighbors:
                    covered_count[w] -= 1
                code.pop(idx)
                code_ints.pop(idx)
                code_set.remove(c_int)
                
                # Find best word to add
                best_gain = 0
                best_word = None
                # Sample random words
                for _ in range(100):
                    rand_int = random.randint(0, total_words - 1)
                    if rand_int in code_set:
                        continue
                    gain = 0
                    for w in neighbors[rand_int]:
                        if covered_count[w] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_word = rand_int
                        
                if best_word is not None:
                    code.append(int_to_word(best_word))
                    code_ints.append(best_word)
                    code_set.add(best_word)
                    for w in neighbors[best_word]:
                        covered_count[w] += 1
                else:
                    # Revert
                    code.append(int_to_word(c_int))
                    code_ints.append(c_int)
                    code_set.add(c_int)
                    for w in c_neighbors:
                        covered_count[w] += 1

    return code
