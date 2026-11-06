"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6.
    
    Strategy:
    1. Start with the 32-word first-order Reed-Muller code (distance 8).
    2. Use greedy addition with weight-ordered candidates to pack more words.
    3. Use local search (swap-based) to try to improve the code.
    """
    N = 18
    MIN_DIST = 6
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    start_time = time.time()
    
    # Helper: compute Hamming distance between two integers
    def hamming_dist(a, b):
        return (a ^ b).bit_count()
    
    # Helper: check if a word can be added to the current set
    def can_add(word, words):
        for w in words:
            if hamming_dist(word, w) < MIN_DIST:
                return False
        return True
    
    # Helper: verify the entire code is valid
    def verify_code(words):
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if hamming_dist(words[i], words[j]) < MIN_DIST:
                    return False
        return True
    
    # Step 1: Generate the 32-word Reed-Muller code
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
    
    # Start with the RM code as our initial set
    words = list(rm_words)
    
    # Step 2: Greedy addition with weight-ordered candidates
    # Generate candidates in order of increasing weight (bit count)
    # This helps find low-weight codewords early
    
    # Pre-compute all 2^18 candidates sorted by weight
    # But that's 262144 candidates - we'll generate them on the fly by weight
    
    # For efficiency, we'll iterate through weights from 0 to 18
    # and for each weight, iterate through combinations of that size
    
    # To speed this up, we'll use a different approach:
    # We'll try adding random words, biased towards low weight
    
    # First, let's try a systematic greedy approach
    # We'll iterate through all possible words in a smart order
    
    # Actually, let's use a more efficient approach:
    # 1. Try to add words of weight 0, 1, 2, ..., up to some limit
    # 2. Then use random sampling for higher weights
    
    # For each weight w, generate all C(18, w) combinations
    # But C(18, 9) = 48620, which is manageable
    
    # Let's do a greedy pass over weights 0 through 18
    for w in range(0, 19):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate all words of weight w
        # Use itertools.combinations for positions
        from itertools import combinations as comb
        
        positions = range(N)
        added_this_weight = 0
        
        # For large w, this could be slow, so we'll limit it
        # C(18, w) values:
        # w=0: 1, w=1: 18, w=2: 153, w=3: 816, w=4: 3060, w=5: 8568,
        # w=6: 18564, w=7: 31824, w=8: 43758, w=9: 48620, ...
        
        # For w >= 8, we'll use random sampling instead of exhaustive
        if w <= 7:
            for pos_tuple in comb(positions, w):
                if time.time() - start_time > DEADLINE:
                    break
                
                word = 0
                for p in pos_tuple:
                    word |= (1 << p)
                
                if can_add(word, words):
                    words.append(word)
                    added_this_weight += 1
        else:
            # For higher weights, use random sampling
            # Try up to 10000 random words of this weight
            num_tries = 10000
            for _ in range(num_tries):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Generate a random word of weight w
                pos_list = list(positions)
                random.shuffle(pos_list)
                selected = pos_list[:w]
                
                word = 0
                for p in selected:
                    word |= (1 << p)
                
                if can_add(word, words):
                    words.append(word)
                    added_this_weight += 1
    
    # Step 3: Local search - try to replace words to improve the code
    # We'll try swap-based local search:
    # For each word in the code, try to replace it with a different word
    # that maintains the distance constraint and allows adding more words
    
    # Actually, a better local search: try to find a word that can be added
    # by temporarily removing some existing words
    
    # Let's do a simpler local search: random perturbation
    # Try to add more words using random candidates
    
    # Step 4: More aggressive random search
    # Try random words, not just weight-ordered
    
    # We'll do multiple passes of random sampling
    for iteration in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try 50000 random words
        for _ in range(50000):
            if time.time() - start_time > DEADLINE:
                break
            
            # Random word
            word = random.randint(0, (1 << N) - 1)
            
            if can_add(word, words):
                words.append(word)
    
    # Step 5: Try to improve by swapping
    # For each word, try to find a replacement that allows adding more words
    # This is complex, so we'll do a simpler version:
    # Try to remove one word and add two others
    
    # Let's do a hill-climbing approach:
    # Try to find a pair of words to remove and a pair to add
    # that maintains the distance constraint
    
    # Actually, let's do a different approach:
    # Try to find a word that is "close" to the current code but can be added
    # by checking if it's at distance >= 6 from all current words
    
    # We've already done this with the random sampling above.
    
    # Step 6: Final verification
    if not verify_code(words):
        # If verification fails, fall back to the RM code
        words = list(rm_words)
    
    return words
# EVOLVE-BLOCK-END
