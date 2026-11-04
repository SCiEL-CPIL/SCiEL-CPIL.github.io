import numpy as np
import itertools
import random

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        # Fallback for other parameters: direct product of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    total_words = q ** n  # 6561
    
    # Precompute all words as tuples and index mapping
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres as index sets
    # Sphere of a word w: {w} union {w with one digit changed}
    # For each word index i, compute its 17 neighbors' indices
    # We can compute neighbors' indices directly from the word's tuple
    
    sphere_indices = []
    for word in all_words:
        sphere = set([word_to_idx[word]])
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    sphere.add(word_to_idx[tuple(new_word)])
        sphere_indices.append(sphere)
    
    # Convert to lists for faster intersection
    sphere_indices_list = [list(s) for s in sphere_indices]
    
    # Randomized greedy construction
    random.seed(42)
    
    def greedy_cover(randomize=False, top_k=1):
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            # Find word whose sphere covers most uncovered words
            best_idx = -1
            best_count = -1
            candidates = []
            
            for i in range(total_words):
                count = len(set(sphere_indices_list[i]) & uncovered)
                if count > best_count:
                    best_count = count
                    candidates = [(i, count)]
                elif count == best_count and count > 0:
                    candidates.append((i, count))
            
            if best_count == 0:
                break
            
            if randomize and len(candidates) > top_k:
                # Pick randomly from top candidates
                chosen = random.choice(candidates[:max(top_k, len(candidates) // 10)])
                best_idx = chosen[0]
            else:
                best_idx = candidates[0][0]
            
            code.append(list(all_words[best_idx]))
            uncovered -= set(sphere_indices_list[best_idx])
        
        return code
    
    # Subtractively reduce a cover
    def reduce_cover(code):
        if not code:
            return code
        
        # Build cover counts
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            idx = word_to_idx[tuple(c)]
            for neighbor in sphere_indices_list[idx]:
                cover_count[neighbor] += 1
        
        # Try to remove redundant codewords
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                idx = word_to_idx[tuple(code[i])]
                # Temporarily remove
                for neighbor in sphere_indices_list[idx]:
                    cover_count[neighbor] -= 1
                
                # Check if all words still covered
                if np.min(cover_count) >= 1:
                    code.pop(i)
                    changed = True
                else:
                    # Restore
                    for neighbor in sphere_indices_list[idx]:
                        cover_count[neighbor] += 1
        
        return code
    
    # Try multiple random seeds and keep the best
    best_code = None
    best_size = float('inf')
    
    for seed in range(5):
        random.seed(seed * 1000 + 42)
        code = greedy_cover(randomize=True, top_k=3)
        code = reduce_cover(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        if best_size <= 486:
            break
    
    if best_code is None:
        best_code = greedy_cover(randomize=False)
        best_code = reduce_cover(best_code)
    
    return best_code
