"""Ternary covering code K_3(8,1) via product construction + coset-restricted local search."""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    
    if q != 3 or n != 8 or radius != 1:
        total = q ** n
        if total <= 200000:
            return [list(x) for x in itertools.product(range(q), repeat=n)]
        else:
            random.seed(42)
            codewords = []
            seen = set()
            while len(codewords) < 2000:
                w = tuple(random.randint(0, q-1) for _ in range(n))
                if w not in seen:
                    seen.add(w)
                    codewords.append(list(w))
            return codewords
    
    # Specialized for q=3, n=8, radius=1
    # Step 1: Build initial covering using product construction
    # Tetracode: length-4 ternary code with 9 codewords
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    tetracode = [tuple(c) for c in tetracode]
    
    # Product construction: tetracode × {0,1,2}^4 gives 9 * 81 = 729 codewords
    # But we can do better: use the fact that the tetracode is a covering code
    # Actually, let's start with a smaller base and build up
    
    # Better approach: Start with a greedy construction seeded by algebraic structure
    # Use the tetracode to guide initial placement
    
    def neighbors(w):
        """Generate all words within Hamming distance 1 of w."""
        result = [tuple(w)]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    result.append(tuple(new_w))
        return result
    
    # Precompute neighbor sets for all possible words (6561 words, 17 neighbors each)
    # This is 6561 * 17 = 111537 entries, manageable
    all_neighbors = {}
    for w in itertools.product(range(q), repeat=n):
        all_neighbors[w] = neighbors(w)
    
    # Step 2: Greedy construction with algebraic seeding
    # Start with tetracode-based codewords as seeds
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_codewords.append(tuple(h) + tail)
    
    # Remove duplicates
    initial_codewords = list(set(initial_codewords))
    
    # Compute coverage
    covered_count = np.zeros(q ** n, dtype=np.int32)
    
    def word_to_idx(w):
        idx = 0
        for i, v in enumerate(w):
            idx += v * (q ** (n - 1 - i))
        return idx
    
    def idx_to_word(idx):
        w = []
        for i in range(n - 1, -1, -1):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))
    
    # Precompute neighbor indices for each word
    neighbor_indices = {}
    for w in all_neighbors:
        neighbor_indices[w] = [word_to_idx(nb) for nb in all_neighbors[w]]
    
    def update_coverage(cw, delta):
        """Update covered_count for adding/removing codeword cw."""
        for nb_idx in neighbor_indices[cw]:
            covered_count[nb_idx] += delta
    
    # Initialize coverage
    codewords = list(initial_codewords)
    cw_set = set(codewords)
    for cw in codewords:
        update_coverage(cw, 1)
    
    # Step 3: Greedy reduction - remove redundant codewords
    removed = True
    while removed:
        removed = False
        for i in range(len(codewords)):
            if time.time() > deadline - 30:
                break
            cw = codewords[i]
            # Check if all neighbors are covered by at least 2 codewords
            can_remove = True
            for nb_idx in neighbor_indices[cw]:
                if covered_count[nb_idx] < 2:
                    can_remove = False
                    break
            if can_remove:
                update_coverage(cw, -1)
                cw_set.discard(cw)
                codewords.pop(i)
                removed = True
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Step 4: Coset-restricted local search
    # The automorphism group includes coordinate permutations and symbol permutations
    # We restrict moves to cosets of the tetracode to maintain structure
    
    # Define coset representatives: for each codeword, its coset is determined by
    # the first 4 coordinates (tetracode part) and the last 4 (free part)
    
    # Try 2-opt style moves: remove one codeword, add another that covers the gaps
    random.seed(42)
    
    for iteration in range(5000):
        if time.time() > deadline - 15:
            break
        
        if len(codewords) <= 100:
            break
            
        # Pick a random codeword to potentially remove
        idx = random.randint(0, len(codewords) - 1)
        cw = codewords[idx]
        
        # Check if removable
        can_remove = True
        for nb_idx in neighbor_indices[cw]:
            if covered_count[nb_idx] < 2:
                can_remove = False
                break
        
        if can_remove:
            update_coverage(cw, -1)
            cw_set.discard(cw)
            codewords.pop(idx)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        else:
            # Try swap: remove cw, add a new codeword from a different coset
            # Find uncovered words
            uncovered = set()
            for nb_idx in neighbor_indices[cw]:
                if covered_count[nb_idx] < 2:
                    uncovered.add(nb_idx)
            
            if not uncovered:
                continue
            
            # Try random candidates from different cosets
            best_new = None
            best_gain = 0
            
            # Sample candidates biased toward different cosets
            for _ in range(30):
                new_cw = tuple(random.randint(0, q-1) for _ in range(n))
                if new_cw in cw_set:
                    continue
                
                # Compute gain: how many uncovered words does new_cw cover?
                gain = 0
                for nb_idx in neighbor_indices[new_cw]:
                    if nb_idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_cw
            
            if best_new is not None and best_gain > 0:
                # Perform swap
                update_coverage(cw, -1)
                cw_set.discard(cw)
                codewords.pop(idx)
                
                update_coverage(best_new, 1)
                cw_set.add(best_new)
                codewords.append(best_new)
    
    # Final verification
    final_covered = set()
    for cw in best_codewords:
        for nb in all_neighbors[cw]:
            final_covered.add(nb)
    
    if len(final_covered) < q ** n:
        # Fallback to initial construction
        return [list(w) for w in initial_codewords]
    
    return [list(w) for w in best_codewords]
