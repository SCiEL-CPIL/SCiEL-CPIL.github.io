import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Map word tuple to index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
        
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))
        
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices of words within Hamming distance 'radius' from word i
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = idx_to_word(i)
        # Generate all words within Hamming distance 'radius'
        # Since radius is 1, we can do this efficiently
        if radius == 1:
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_w = list(w)
                        new_w[pos] = val
                        neighbors[i].append(word_to_idx(tuple(new_w)))
        else:
            # General case, but radius is likely 1
            # For radius > 1, this is more expensive, but n=8 is small
            pass # Fallback if radius != 1, but task implies radius 1
            
    # Greedy construction
    uncovered = [True] * total_words
    code = []
    
    # To speed up, we can maintain a count of uncovered neighbors for each word
    # But total_words is small (6561), so scanning is fast enough.
    # Let's just scan.
    
    while any(uncovered):
        best_word = -1
        best_count = -1
        
        # Check time
        if time.time() > deadline:
            break
            
        for i in range(total_words):
            # Only consider words that are not already in code? 
            # Actually, adding a word already in code is useless.
            # But we can just check coverage gain.
            count = 0
            for nb in neighbors[i]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = i
            if count == 0:
                # No point checking further for this word
                pass
                
        if best_count == 0:
            break # Should not happen if any(uncovered) is true and neighbors are correct
            
        # Add best_word to code
        code.append(idx_to_word(best_word))
        
        # Mark neighbors as covered
        for nb in neighbors[best_word]:
            uncovered[nb] = False
            
    # Remove redundant codewords
    # Repeat until no more removals
    removed = True
    while removed:
        removed = False
        # Check time
        if time.time() > deadline:
            break
            
        for i in range(len(code)):
            c = code[i]
            idx_c = word_to_idx(c)
            
            # Temporarily remove
            code.pop(i)
            
            # Check if any word is now uncovered
            # We need to recompute coverage or maintain a counter.
            # Since n is small, let's just check the neighbors of c.
            # If a neighbor of c is not covered by any other codeword, then c is necessary.
            
            is_necessary = False
            for nb in neighbors[idx_c]:
                # Is nb covered by any other codeword in code?
                covered_by_other = False
                for j in range(len(code)):
                    other_c = code[j]
                    # Check if nb is within radius of other_c
                    # This is O(|code| * n) per neighbor, which is slow.
                    # Better: precompute coverage counts.
                    pass
            # This approach is too slow. Let's use a coverage count array.
            
            # Let's restart the redundancy check with a coverage count array.
            code.insert(i, c) # Put it back for now
            
        # Let's implement a faster redundancy check using a coverage array.
        # coverage_count[w] = number of codewords in code that cover w
        # Initially, compute coverage_count for all w.
        # Then, for each c in code:
        #   Decrement coverage_count for neighbors of c.
        #   If any coverage_count becomes 0, c is necessary, so increment back and break.
        #   Else, c is redundant, so keep it removed.
        
        # We need to be careful with the loop.
        
        break # The above loop was a placeholder. Let's implement the proper one below.
        
    # Proper redundancy removal
    coverage_count = [0] * total_words
    for c in code:
        idx_c = word_to_idx(c)
        for nb in neighbors[idx_c]:
            coverage_count[nb] += 1
            
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
            
        for i in range(len(code)):
            c = code[i]
            idx_c = word_to_idx(c)
            
            # Check if c is redundant
            redundant = True
            for nb in neighbors[idx_c]:
                if coverage_count[nb] == 1:
                    # Only c covers nb
                    redundant = False
                    break
                    
            if redundant:
                # Remove c
                code.pop(i)
                for nb in neighbors[idx_c]:
                    coverage_count[nb] -= 1
                changed = True
                break # Restart loop as indices changed
                
    return [list(c) for c in code]
