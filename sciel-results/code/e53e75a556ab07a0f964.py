import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    MAX_SPACE = 1 << N
    
    # Precompute all masks with bit_count < 6 (i.e., <= 5)
    # These are the vectors within distance 5 of a codeword
    # If a new word is within distance < 6 of any existing codeword, it's invalid
    masks = []
    for m in range(MAX_SPACE):
        if m.bit_count() < MIN_DIST:
            masks.append(m)
    
    # forbidden array: forbidden[i] is True if i is within distance < 6 of some codeword
    forbidden = bytearray(MAX_SPACE)
    words = []
    
    def add_word(w):
        if forbidden[w]:
            return False
        # Mark w and all its neighbors (within distance 5) as forbidden
        for m in masks:
            idx = w ^ m
            forbidden[idx] = 1
        words.append(w)
        return True
    
    # Initialize with a good seed code
    # Use the [18,9,6] extended quadratic-residue code shortened by one coordinate
    # This gives 256 words. But we can't easily construct it from scratch.
    # Instead, let's use a greedy approach starting from a small seed.
    
    # Start with all-zero word
    add_word(0)
    
    # Greedy add: try to add words in a structured order
    # First, try words with low weight (more likely to be far apart)
    # Actually, let's use a randomized greedy approach with multiple restarts
    
    # Phase 1: Fast greedy with structured ordering
    # Try words in order of increasing weight, then lexicographic
    # This is slow for 2^17, so let's use random sampling
    
    # Better approach: use the fact that we have a forbidden array
    # We can quickly check if a word is valid
    
    # Let's do multiple random greedy passes
    random.seed(42)
    
    # First, let's try to build a large code using a smarter initialization
    # Use the first-order Reed-Muller code RM(1,4) embedded in 17 bits
    # RM(1,4) has 32 codewords of length 16, distance 8
    # Embed in 17 bits by adding a parity bit or just zero-padding
    
    # Actually, let's just use pure random greedy with the forbidden array
    # The forbidden array makes each check O(1) after the O(|masks|) insertion
    
    # |masks| = sum_{i=0}^{5} C(17,i) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402
    # So each insertion takes ~9400 operations
    
    # Let's try to add as many as possible
    # We'll use a random permutation of all words and try to add them
    
    # Generate a random permutation of all 2^17 words
    # This is 131072 words, which is manageable
    
    all_words = list(range(MAX_SPACE))
    random.shuffle(all_words)
    
    # Try to add words in this random order
    for w in all_words:
        if time.time() > deadline - 5:
            break
        if forbidden[w]:
            continue
        add_word(w)
    
    # Now, try to improve by swapping
    # Actually, let's try a different strategy:
    # Use a local search / hill climbing approach
    
    # But first, let's see how many we got
    # If we got a lot, we can try to improve
    
    # Let's try a second phase: try to add words that are far from the current code
    # But the forbidden array already handles this
    
    # Actually, the greedy approach with random ordering should be decent.
    # Let's try multiple random orderings to see if we can do better
    
    best_words = list(words)
    best_forbidden = bytearray(forbidden)
    
    # Try a few more random orderings
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        
        # Reset
        words.clear()
        forbidden = bytearray(MAX_SPACE)
        
        all_words = list(range(MAX_SPACE))
        random.shuffle(all_words)
        
        for w in all_words:
            if time.time() > deadline - 5:
                break
            if forbidden[w]:
                continue
            add_word(w)
        
        if len(words) > len(best_words):
            best_words = list(words)
            best_forbidden = bytearray(forbidden)
    
    return best_words
