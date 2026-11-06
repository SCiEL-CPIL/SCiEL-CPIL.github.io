"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a binary error-correcting code of length 18 with minimum distance 6.
    Uses a systematic search with bit-parallel conflict masks to find a large code.
    """
    N = 18
    MIN_DIST = 6
    
    # Precompute popcount for all 18-bit numbers? No, that's 2^18 = 262144, which is fine.
    # But we need Hamming distance between pairs.
    # Let's use the approach described: conflict masks.
    # However, the directive suggests a specific approach. Let's try to implement a greedy algorithm
    # with efficient distance checking using numpy if possible, or just optimized Python.
    
    # Actually, for N=18, 2^18 is small enough that we can potentially use a bitset or array of size 2^18.
    # Let's try a greedy construction:
    # 1. Start with 0.
    # 2. Iterate through all possible words in a specific order (e.g., Gray code or just 0..2^18-1).
    # 3. For each word, check if it has distance >= 6 from all existing words.
    # 4. If yes, add it.
    
    # To speed up the check, we can maintain a list of codewords and check against them.
    # With up to ~500 codewords, checking against 500 words for each candidate is 500 * 262144 = 131M ops.
    # In Python, this might be too slow for 60s.
    
    # Better approach: Use the "conflict mask" idea.
    # For each existing codeword c, we know all words within distance < 6 are forbidden.
    # The number of words within distance 5 from c is sum_{i=0}^{5} C(18, i) = 1 + 18 + 153 + 816 + 3060 + 8568 = 12616.
    # So for each codeword, we mark ~12k words as forbidden.
    # If we have ~500 codewords, we mark ~6M words.
    # We can use a boolean array of size 2^18 (262144) to track forbidden words.
    # Wait, 2^18 is 262144. 6M marks on an array of 262144 is fine, but we need to update it efficiently.
    
    # Let's try this:
    # 1. Create a boolean array `forbidden` of size 2^18, initially all False.
    # 2. Start with code = [0]. Mark all words within distance < 6 from 0 as forbidden.
    # 3. Iterate through candidate words in a good order (e.g., Gray code).
    # 4. If candidate is not forbidden, add it to code, and mark all words within distance < 6 from it as forbidden.
    
    # How to mark efficiently?
    # For a given word c, the set of words within distance d is the Hamming ball.
    # We can precompute the list of offsets for each distance?
    # Or, we can iterate through all masks of weight < 6 and XOR with c.
    # Number of masks of weight < 6 is 12616.
    # For each new codeword, we do 12616 XORs and array updates.
    # If we have 500 codewords, that's 500 * 12616 = 6.3M operations. This is very fast.
    
    # Let's implement this.
    
    size = 1 << N
    forbidden = np.zeros(size, dtype=bool)
    
    # Precompute all masks of weight < MIN_DIST
    # Weight < 6 means weight 0 to 5.
    masks = []
    for w in range(MIN_DIST):
        # Generate all masks of weight w
        # We can use itertools.combinations or a recursive generator.
        # For efficiency, let's precompute them once.
        pass
        
    # Let's generate masks of weight 0 to 5.
    # There are 12616 such masks.
    # We can generate them by iterating i from 0 to size-1 and checking bit_count < MIN_DIST.
    # This takes 262144 iterations, which is fast.
    masks = [i for i in range(size) if i.bit_count() < MIN_DIST]
    masks = np.array(masks, dtype=np.uint32) # Use uint32 for fast XOR? Or just int.
    
    # Function to mark forbidden neighbors
    def mark_forbidden(c):
        # c is an integer
        # neighbors = (c XOR masks) % size
        # forbidden[neighbors] = True
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = True
        
    # Start with 0
    code = [0]
    mark_forbidden(0)
    
    # Iterate through candidates in Gray code order to ensure locality?
    # Or just 0 to size-1.
    # Gray code might help if we were doing something else, but here we're just checking a boolean array.
    # The order doesn't matter for correctness, but might affect the final code size if we stop early?
    # No, we go through all candidates.
    # Actually, the greedy choice depends on the order. A "better" order might yield a larger code.
    # Let's try a few orders or just the standard order first.
    # Standard order: 0, 1, 2, ...
    
    for c in range(1, size):
        if not forbidden[c]:
            code.append(c)
            mark_forbidden(c)
            
    return code

# EVOLVE-BLOCK-END
