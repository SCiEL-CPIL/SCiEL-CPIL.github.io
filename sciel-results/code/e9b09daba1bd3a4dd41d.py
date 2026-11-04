import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Helper to convert index to word and back
    # We work with indices 0..total_words-1
    
    # Pre-compute spheres for all words in the space
    # sphere[i] = list of indices j such that HammingDistance(i, j) <= radius
    # For efficiency, we compute neighbors on the fly or pre-compute if space is small.
    # For n=8, q=3, total=6561. Sphere size for R=1 is 17.
    # Pre-computation cost: 6561 * 17 ops. Very fast.
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Pre-compute spheres
    # Using a list of lists
    spheres = [[] for _ in range(total_words)]
    
    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    for i in range(total_words):
        w_i = all_words[i]
        # Add self
        spheres[i].append(i)
        # Add neighbors within radius
        # For radius 1, just flip one position
        # For general radius, we need BFS or combinatorial generation
        # Given the constraint, radius is likely 1 or small.
        # We'll implement a generic BFS up to radius for correctness, 
        # but optimize for R=1 which is the likely case.
        
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if w_i[pos] != val:
                        new_w = w_i.copy()
                        new_w[pos] = val
                        spheres[i].append(word_to_idx(new_w))
        else:
            # Generic BFS
            visited = {i}
            queue = [(w_i, 0)]
            while queue:
                curr_w, dist = queue.pop(0)
                if dist == radius:
                    continue
                curr_idx = word_to_idx(curr_w)
                spheres[i].append(curr_idx)
                for pos in range(n):
                    for val in range(q):
                        if curr_w[pos] != val:
                            new_w = curr_w.copy()
                            new_w[pos] = val
                            new_idx = word_to_idx(new_w)
                            if new_idx not in visited:
                                visited.add(new_idx)
                                queue.append((new_w, dist + 1))
            # Note: The above BFS adds to spheres[i] but we already added i.
            # This generic path is slow. We assume R=1 for speed, 
            # but fallback to this if R>1.
            # To keep it fast, we rely on R=1. If R>1, we might timeout.
            # But the task implies R=1 (covering code K_q(n,1)).

    # State
    cover_count = np.zeros(total_words, dtype=np.int32)
    uncovered = set(range(total_words))
    code = []
    code_set = set()

    def add_codeword(c):
        if c in code_set:
            return
        code_set.add(c)
        code.append(c)
        for w in spheres[c]:
            cover_count[w] += 1
            if cover_count[w] == 1:
                uncovered.discard(w)

    def remove_codeword(c):
        if c not in code_set:
            return
        code_set.remove(c)
        code.remove(c)
        for w in spheres[c]:
            cover_count[w] -= 1
            if cover_count[w] == 0:
                uncovered.add(w)

    def can_remove(c):
        # Check if any word in spheres[c] has cover_count == 1
        # and that count is from c.
        # Since we maintain cover_count, if cover_count[w] == 1 for any w in spheres[c],
        # it implies c is the ONLY one covering w (because if another codeword covered w,
        # count would be >= 2).
        # So we just check if min(cover_count[w] for w in spheres[c]) > 1
        # Actually, if cover_count[w] == 1, it MUST be c, because c is in the code.
        for w in spheres[c]:
            if cover_count[w] == 1:
                return False
        return True

    # Phase 1: Greedy Construction
    # We start with an empty code and fill it.
    # To speed up, we can seed with a known good structure if available,
    # but greedy is robust.
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered word
        w = random.choice(list(uncovered))
        
        # Candidates are all c such that w in spheres[c]
        # Which is equivalent to c in spheres[w] (symmetry of Hamming distance)
        candidates = spheres[w]
        
        best_c = None
        best_gain = -1
        
        # Evaluate candidates
        # Gain = number of uncovered words covered by c
        for c in candidates:
            # Quick check: if c is already in code, gain is 0 (or we handle duplicates)
            # Actually, we should only consider c NOT in code_set to avoid redundancy,
            # but adding a duplicate doesn't help.
            if c in code_set:
                continue
                
            gain = 0
            for u in spheres[c]:
                if u in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c is not None:
            add_codeword(best_c)
        else:
            # Fallback: pick any candidate
            best_c = candidates[0]
            add_codeword(best_c)

    # Phase 2: Local Search Optimization
    # Try to remove redundant codewords
    # Sort code by "uniqueness" (number of words it covers exclusively)
    # We approximate uniqueness by the number of words with cover_count == 1 in its sphere.
    
    def get_uniqueness(c):
        count = 0
        for w in spheres[c]:
            if cover_count[w] == 1:
                count += 1
        return count

    # Iterative removal pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort by uniqueness ascending (most redundant first)
        # This is O(M log M) where M is code size.
        # M is likely small (< 1000).
        code_sorted = sorted(code, key=get_uniqueness)
        
        for c in code_sorted:
            if time.time() > deadline:
                break
            if c in code_set and can_remove(c):
                remove_codeword(c)
                improved = True

    # Phase 3: 1-Swap Local Search
    # Try to swap a codeword with a neighbor to enable further removals
    # or simply to find a better configuration.
    
    swap_attempts = 0
    while time.time() < deadline and swap_attempts < 1000:
        swap_attempts += 1
        
        if not code:
            break
            
        # Pick random codeword
        c = random.choice(code)
        
        # Pick random neighbor
        neighbors = spheres[c]
        c_prime = random.choice(neighbors)
        
        if c_prime == c:
            continue
            
        # If c_prime is already in code, skip (or try to remove c if redundant)
        if c_prime in code_set:
            # Check if we can remove c
            if can_remove(c):
                remove_codeword(c)
            continue
        
        # Check if swap {c -> c_prime} is valid
        # Condition: All words uniquely covered by c must be covered by c_prime
        # Words uniquely covered by c: { w in spheres[c] | cover_count[w] == 1 }
        # If we remove c, these words become uncovered.
        # If we add c_prime, these words must be in spheres[c_prime].
        
        unique_words = []
        for w in spheres[c]:
            if cover_count[w] == 1:
                unique_words.append(w)
        
        valid_swap = True
        for w in unique_words:
            # w must be in spheres[c_prime]
            # Since spheres are lists, checking membership is O(len(sphere)).
            # For R=1, len=17. Fast.
            if w not in spheres[c_prime]:
                valid_swap = False
                break
        
        if valid_swap:
            # Perform swap
            remove_codeword(c)
            add_codeword(c_prime)
            # After swap, try to remove more
            # Quick removal pass
            for c_remove in code:
                if c_remove in code_set and can_remove(c_remove):
                    remove_codeword(c_remove)

    # Final cleanup: ensure no duplicates (shouldn't happen)
    # Return list of codewords as lists of integers
    result = [all_words[c] for c in code]
    
    return result
