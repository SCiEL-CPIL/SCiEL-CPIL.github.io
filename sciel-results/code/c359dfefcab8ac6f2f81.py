"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Greedy set-cover algorithm to find a small covering code.
    For each candidate codeword, compute the set of words within Hamming distance <= radius.
    Iteratively pick the codeword that covers the most currently uncovered words.
    """
    if n <= 0:
        return [[]]
    
    # For small instances, we can afford a more exhaustive approach
    total_words = q ** n
    
    # If the space is small enough, we can use a bitmask representation
    # But for q=3, n=6, total_words=729, which is manageable
    
    if total_words <= 2**20:  # Use bitmask for small spaces
        return _construct_greedy_bitmask(q, n, radius)
    else:
        return _construct_greedy_general(q, n, radius)


def _construct_greedy_bitmask(q, n, radius):
    """Greedy set-cover using bitmasks for small spaces."""
    total_words = q ** n
    
    # Map each word to an index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute the ball of radius `radius` around each word as a bitmask
    # ball_mask[idx] is a bitmask of all words within distance <= radius of idx
    ball_masks = [0] * total_words
    
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 0
        # Enumerate all words within distance <= radius
        # For radius=1, this is 1 + n*(q-1) words
        # General approach: iterate over all words and check distance (slow for large n)
        # Better: generate neighbors directly
        
        if radius == 0:
            mask = 1 << idx
        elif radius == 1:
            mask = 1 << idx
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        else:
            # For larger radius, enumerate all words within distance
            # This can be expensive, so we use a recursive generator
            def generate_neighbors(center, remaining_radius, pos=0):
                if pos == n:
                    yield list(center)
                    return
                # Option 1: keep same value
                if remaining_radius >= 0:
                    yield from generate_neighbors(center, remaining_radius, pos + 1)
                # Option 2: change to other values (uses 1 from radius)
                if remaining_radius > 0:
                    for val in range(q):
                        if val != center[pos]:
                            old_val = center[pos]
                            center[pos] = val
                            yield from generate_neighbors(center, remaining_radius - 1, pos + 1)
                            center[pos] = old_val
            
            for neighbor in generate_neighbors(center, radius):
                nidx = word_to_idx(neighbor)
                mask |= 1 << nidx
        
        ball_masks[idx] = mask
    
    # Greedy set cover
    uncovered = (1 << total_words) - 1
    codewords = []
    
    while uncovered != 0:
        # Find the codeword that covers the most uncovered words
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            if uncovered & (1 << idx):
                # This word is uncovered, so it could be a candidate center
                # But actually, we should consider ALL possible centers, not just uncovered ones
                pass
        # Actually, we should consider all possible centers (all words in the space)
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = bin(covered).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
        
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    
    return codewords


def _construct_greedy_general(q, n, radius):
    """Greedy set-cover for larger spaces using a different representation."""
    # For larger spaces, we use a set-based approach
    # But this will be slow. Let's try to be smarter.
    
    # Actually, let's fall back to the bitmask approach but with numpy for efficiency
    # Or use a different strategy
    
    # For now, use the same bitmask approach but with early termination
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute ball masks
    ball_masks = [0] * total_words
    
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 0
        if radius == 0:
            mask = 1 << idx
        elif radius == 1:
            mask = 1 << idx
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        else:
            def generate_neighbors(center, remaining_radius, pos=0):
                if pos == n:
                    yield list(center)
                    return
                if remaining_radius >= 0:
                    yield from generate_neighbors(center, remaining_radius, pos + 1)
                if remaining_radius > 0:
                    for val in range(q):
                        if val != center[pos]:
                            old_val = center[pos]
                            center[pos] = val
                            yield from generate_neighbors(center, remaining_radius - 1, pos + 1)
                            center[pos] = old_val
            
            for neighbor in generate_neighbors(center, radius):
                nidx = word_to_idx(neighbor)
                mask |= 1 << nidx
        
        ball_masks[idx] = mask
    
    uncovered = (1 << total_words) - 1
    codewords = []
    
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = bin(covered).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
        
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    
    return codewords
# EVOLVE-BLOCK-END
