import itertools
import random
import time
import math

def construct(q, n, radius):
    # Constants for the specific task instance
    # q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other instances, though task implies specific
        # Just return a trivial valid code (all words) if not the target case
        # This is inefficient but valid
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Precompute neighbors for all 729 words
    # Index: v[0]*3^5 + v[1]*3^4 + ... + v[5]
    # Convert index to tuple
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls
    all_indices = list(range(q**n))
    neighbors = [[] for _ in range(q**n)]
    
    # Generate all words
    words = [tuple(t) for t in itertools.product(range(q), repeat=n)]
    
    for idx, w in enumerate(words):
        neighbors[idx].append(idx)
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        # Remove duplicates just in case, though logic above is unique
        neighbors[idx] = list(set(neighbors[idx]))
    
    # Convert neighbors to sets for faster membership testing
    neighbor_sets = [set(nb) for nb in neighbors]

    start_time = time.time()
    deadline = 145.0 # Safety margin (150s limit, 5s buffer)

    # Greedy Set-Cover Algorithm
    # Start with empty code, iteratively add codeword that covers most uncovered words
    uncovered = set(range(q**n))
    code = []
    
    # Precompute for each index, the list of codewords that cover it
    # Actually, we need to track which codewords cover each uncovered word
    
    # For efficiency, maintain a count of how many times each word is covered
    # and a set of currently uncovered words
    
    while uncovered and time.time() < deadline:
        # Find codeword that covers most uncovered words
        best_idx = -1
        best_count = 0
        
        # Sample a subset of candidates to save time
        # Try all 729 words, but we can be smart about it
        for c_idx in range(q**n):
            if time.time() > deadline:
                break
            # Count how many uncovered words this codeword covers
            count = 0
            for nb in neighbors[c_idx]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = c_idx
                if best_count == len(uncovered):
                    break  # Perfect coverage, can't do better
        
        if best_idx == -1 or best_count == 0:
            break  # No progress possible
        
        # Add this codeword
        code.append(idx_to_tuple(best_idx))
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    # Local improvement: try to remove or swap codewords
    if not code:
        # Fallback: use tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                code.append(tuple(h + list(tail)))
    
    # Second greedy pass: try to find a better solution with different randomization
    if time.time() < deadline - 30:
        # Try a randomized greedy approach
        uncovered2 = set(range(q**n))
        code2 = []
        
        while uncovered2 and time.time() < deadline - 30:
            best_idx = -1
            best_count = 0
            
            # Sample candidates with some randomness
            candidate_indices = list(range(q**n))
            random.shuffle(candidate_indices)
            
            for c_idx in candidate_indices:
                if time.time() > deadline - 30:
                    break
                count = 0
                for nb in neighbors[c_idx]:
                    if nb in uncovered2:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = c_idx
                    if best_count == len(uncovered2):
                        break
            
            if best_idx == -1 or best_count == 0:
                break
            
            code2.append(idx_to_tuple(best_idx))
            for nb in neighbors[best_idx]:
                uncovered2.discard(nb)
        
        # Use the better solution
        if len(code2) < len(code):
            code = code2
    
    # Initialize cover counts
    cover_count = [0] * (q**n)
    for c in code:
        c_idx = tuple_to_idx(c)
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1
    
    # Local Search
    def try_remove(c_tuple):
        c_idx = tuple_to_idx(c_tuple)
        uncovered = []
        for nb in neighbors[c_idx]:
            cover_count[nb] -= 1
            if cover_count[nb] == 0:
                uncovered.append(nb)
        
        if not uncovered:
            return True, [] # Successfully removed, no uncovered
        else:
            # Restore
            for nb in neighbors[c_idx]:
                cover_count[nb] += 1
            return False, uncovered

    def try_add(c_tuple):
        c_idx = tuple_to_idx(c_tuple)
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1

    def find_swap_target(uncovered_indices):
        # Find a word w such that all indices in uncovered_indices are in neighbors[w]
        # Heuristic: Pick a random point from uncovered, check its neighbors
        if not uncovered_indices:
            return None
        
        # Try multiple uncovered points to increase chance of finding a good swap
        candidates_to_try = uncovered_indices[:min(5, len(uncovered_indices))]
        random.shuffle(candidates_to_try)
        
        for u0 in candidates_to_try:
            candidates = list(neighbors[u0])
            random.shuffle(candidates)
            
            for c_idx in candidates:
                # Check if c_idx covers all uncovered
                c_nb_set = neighbor_sets[c_idx]
                if all(u in c_nb_set for u in uncovered_indices):
                    return idx_to_tuple(c_idx)
        
        # If no single neighbor covers all, try to find a codeword that covers
        # at least the most uncovered words (for partial improvement)
        if len(uncovered_indices) > 1:
            # Find the codeword that covers the most uncovered words
            best_c_idx = -1
            best_covered = 0
            u0 = uncovered_indices[0]
            candidates = list(neighbors[u0])
            random.shuffle(candidates)
            
            for c_idx in candidates:
                c_nb_set = neighbor_sets[c_idx]
                covered = sum(1 for u in uncovered_indices if u in c_nb_set)
                if covered > best_covered:
                    best_covered = covered
                    best_c_idx = c_idx
            
            if best_c_idx != -1 and best_covered > len(uncovered_indices) // 2:
                return idx_to_tuple(best_c_idx)
        
        return None

    # Simulated Annealing for local optimization
    def simulated_annealing(code_list, cover_counts, start_temp, end_temp, cooling_rate, max_iters):
        """
        Perform simulated annealing on the code to minimize its size.
        Moves: remove a codeword if possible, swap a codeword with another.
        """
        current_code = list(code_list)
        current_cover = list(cover_counts)
        current_size = len(current_code)
        best_code = list(current_code)
        best_size = current_size
        
        temp = start_temp
        n_words = q ** n
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3 and current_size > 0:
                # Try to remove a random codeword
                idx = random.randint(0, current_size - 1)
                c = current_code[idx]
                c_idx = tuple_to_idx(c)
                
                # Check if removal is valid
                can_remove = True
                uncovered = []
                for nb in neighbors[c_idx]:
                    if current_cover[nb] == 1:
                        can_remove = False
                        uncovered.append(nb)
                
                if can_remove:
                    # Accept removal
                    for nb in neighbors[c_idx]:
                        current_cover[nb] -= 1
                    current_code.pop(idx)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_code = list(current_code)
                else:
                    # Reject removal (cost is infinite for invalid)
                    pass
                    
            elif move_type < 0.7 and current_size > 0:
                # Try to swap a codeword
                idx = random.randint(0, current_size - 1)
                old_c = current_code[idx]
                old_c_idx = tuple_to_idx(old_c)
                
                # Remove old codeword temporarily
                for nb in neighbors[old_c_idx]:
                    current_cover[nb] -= 1
                
                # Find a new codeword to add
                # Pick a random word from the space, or one that helps cover uncovered areas
                # First, identify uncovered words after removal
                uncovered_after_removal = [nb for nb in neighbors[old_c_idx] if current_cover[nb] == 0]
                
                if not uncovered_after_removal:
                    # Old codeword was redundant, just add a random one or keep size same
                    # Actually, if it's redundant, we should have removed it. 
                    # Let's pick a random new codeword to maintain size but change composition
                    new_c_idx = random.randint(0, n_words - 1)
                    new_c = idx_to_tuple(new_c_idx)
                    
                    # Check if new_c is already in code
                    if new_c not in current_code:
                        for nb in neighbors[new_c_idx]:
                            current_cover[nb] += 1
                        current_code[idx] = new_c
                    else:
                        # Restore old
                        for nb in neighbors[old_c_idx]:
                            current_cover[nb] += 1
                else:
                    # Need to cover uncovered_after_removal
                    # Pick a random candidate that covers at least one uncovered word
                    candidates = []
                    # Sample a few candidates from neighbors of uncovered words
                    for u in uncovered_after_removal[:3]:
                        for nb in neighbors[u]:
                            if nb not in candidates:
                                candidates.append(nb)
                    
                    if candidates:
                        new_c_idx = random.choice(candidates)
                        new_c = idx_to_tuple(new_c_idx)
                        
                        # Check validity after adding new_c
                        # We need to ensure all words are covered
                        # Uncovered words were those with count 0 after removing old_c.
                        # After adding new_c, they might be covered.
                        # But other words might have been covered only by old_c? No, we subtracted old_c.
                        # So any word with count 0 now must be covered by new_c.
                        
                        # Check if new_c covers all currently uncovered words
                        valid_swap = True
                        for u in uncovered_after_removal:
                            if current_cover[u] == 0 and u not in neighbor_sets[new_c_idx]:
                                valid_swap = False
                                break
                        
                        if valid_swap:
                            # Also ensure no word goes negative (shouldn't happen)
                            # and that new_c is not already in the code
                            if new_c not in current_code:
                                for nb in neighbors[new_c_idx]:
                                    current_cover[nb] += 1
                                current_code[idx] = new_c
                            else:
                                # Restore old
                                for nb in neighbors[old_c_idx]:
                                    current_cover[nb] += 1
                        else:
                            # Invalid swap, restore old
                            for nb in neighbors[old_c_idx]:
                                current_cover[nb] += 1
                    else:
                        # No candidates, restore old
                        for nb in neighbors[old_c_idx]:
                            current_cover[nb] += 1

            elif current_size < 100:
                # Try to add a codeword (if we are far from optimal, this might help escape local minima)
                # Only accept if it allows a subsequent removal? No, SA usually fixes size.
                # Let's stick to fixed size or decreasing size.
                # Actually, adding increases cost. We only accept if it leads to a better state later?
                # Standard SA for minimization: cost = size. Adding increases cost.
                # We only accept if delta < 0 or with prob exp(-delta/T).
                # Delta = +1.
                if random.random() < math.exp(-1.0 / temp):
                    # Find a good codeword to add
                    # Pick one that covers the most "critical" words (low cover count)
                    best_add_idx = -1
                    best_gain = 0
                    for c_idx in range(n_words):
                        if idx_to_tuple(c_idx) in current_code:
                            continue
                        gain = 0
                        for nb in neighbors[c_idx]:
                            if current_cover[nb] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_add_idx = c_idx
                    
                    if best_add_idx != -1:
                        new_c = idx_to_tuple(best_add_idx)
                        for nb in neighbors[best_add_idx]:
                            current_cover[nb] += 1
                        current_code.append(new_c)
                        current_size += 1
                    # If no gain, don't add
            else:
                # No move taken
                pass
            
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp
                
        return best_code, best_size

    # Run Simulated Annealing
    if time.time() < deadline - 10:
        initial_size = len(code)
        # Tune parameters
        start_temp = 1.0
        end_temp = 0.01
        cooling_rate = 0.99
        # Estimate max iterations based on time remaining
        max_iters = 50000 # Adjust based on speed
        
        sa_code, sa_size = simulated_annealing(code, cover_count, start_temp, end_temp, cooling_rate, max_iters)
        
        if sa_size < initial_size:
            code = sa_code
            # Update cover_count if needed? 
            # The function returns best_code, but we need to recompute cover_count for further passes if we continue.
            # For now, we just update code.
            cover_count = [0] * (q**n)
            for c in code:
                c_idx = tuple_to_idx(c)
                for nb in neighbors[c_idx]:
                    cover_count[nb] += 1

    # Final cleanup: ensure no duplicate codewords
    seen = set()
    unique_code = []
    for c in code:
        c_key = tuple(c)
        if c_key not in seen:
            seen.add(c_key)
            unique_code.append(c)
    code = unique_code
    
    # Convert back to lists
    return [list(c) for c in code]
