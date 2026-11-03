import itertools
import random

def construct(q, n, radius):
    # For the specific case q=3, n=6, radius=1
    # We will use a Greedy Set Cover followed by Local Search (Pruning)
    
    if q == 3 and n == 6 and radius == 1:
        # Total words
        total_words = 3 ** n
        # Generate all words as tuples of ints
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        
        # Map word to index for fast lookup
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute coverage: for each word c, which words are within distance <= radius
        # Coverage is a list of indices
        coverage = [set() for _ in range(total_words)]
        
        # To speed up distance calculation, we can precompute neighbors
        # But 729 * 25 is small.
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        for i, c in enumerate(all_words):
            cov = set()
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    cov.add(j)
            coverage[i] = cov
            
        # Greedy Set Cover
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_cover_count = -1
            for i in range(total_words):
                # Optimization: if i is already in code, skip? 
                # Actually, we just pick the one that covers most uncovered.
                # We don't need to check if i is in code, just pick the best candidate.
                # But we should probably avoid picking a center that covers very few.
                new_cover = len(coverage[i] & uncovered)
                if new_cover > best_cover_count:
                    best_cover_count = new_cover
                    best_idx = i
                    if best_cover_count == len(uncovered):
                        break
            
            if best_idx == -1:
                break # Should not happen if space is finite and radius >= 0
                
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        # Local Search: Pruning
        # Check if we can remove any codeword
        improved = True
        while improved:
            improved = False
            current_code = set(code_indices)
            # Verify validity first? Greedy ensures validity.
            # Try removing each
            for c in list(code_indices):
                temp_code = current_code - {c}
                # Check if temp_code covers all words
                # A word w is covered if any c' in temp_code has w in coverage[c']
                # We can check if union of coverage[c'] for c' in temp_code is all words
                
                # Efficient check: 
                # If we remove c, the only words that might become uncovered are those 
                # that were ONLY covered by c.
                # But simpler: just check full coverage if code is small.
                # Code size ~73. Total words 729.
                # 73 * 25 checks is ~1800 operations. Very fast.
                
                is_valid = True
                # Optimization: Only check words that were covered by c?
                # No, just check all. It's fast enough.
                # Actually, we can maintain a count of how many codewords cover each word.
                
                # Let's use the count method for pruning efficiency
                pass
        
        # Let's implement the count method properly for pruning
        
        # Re-initialize for pruning phase
        # We have code_indices from greedy.
        # Calculate coverage counts
        cover_count = [0] * total_words
        for c in code_indices:
            for w in coverage[c]:
                cover_count[w] += 1
        
        improved = True
        while improved:
            improved = False
            # Try to remove each codeword
            for i in range(len(code_indices)):
                c = code_indices[i]
                # If we remove c, words with count 1 that are in coverage[c] will become 0
                # Check if any word in coverage[c] has count 1
                can_remove = True
                for w in coverage[c]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove c
                    for w in coverage[c]:
                        cover_count[w] -= 1
                    code_indices.pop(i)
                    improved = True
                    break # Restart loop with new index list
        
        # Convert indices back to words
        return [list(all_words[i]) for i in code_indices]
    
    else:
        # Fallback for other parameters: Simple Greedy
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        coverage = [set() for _ in range(total_words)]
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
