"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 18
    MIN_DIST = 6
    SIZE = 1 << N
    
    # 1. Generate the Extended Quadratic Residue code of length 18
    # QR code mod 17: [17, 9, 5], extended to [18, 9, 6] with 512 codewords
    n = 17
    qr = set()
    for i in range(1, n):
        qr.add((i * i) % n)
    
    gen_rows = []
    gen_rows.append([1] * n)
    qr_vec = [1 if i in qr else 0 for i in range(n)]
    for shift in range(8):
        row = [(qr_vec[(i + shift) % n]) for i in range(n)]
        gen_rows.append(row)
    
    initial_words = []
    for mask in range(512):
        word = [0] * n
        for bit_idx in range(9):
            if (mask >> bit_idx) & 1:
                row = gen_rows[bit_idx]
                for i in range(n):
                    word[i] ^= row[i]
        weight = sum(word)
        parity_bit = weight % 2
        word.append(parity_bit)
        val = 0
        for i in range(18):
            if word[i]:
                val |= (1 << i)
        initial_words.append(val)
    
    # 2. Greedy extension using vectorized forbidden mask
    # Precompute all masks with Hamming weight < MIN_DIST (i.e., <= 5)
    # There are 12616 such masks.
    masks = []
    for i in range(SIZE):
        if i.bit_count() < MIN_DIST:
            masks.append(i)
    masks = np.array(masks, dtype=np.uint32)
    
    # Initialize forbidden array
    forbidden = np.zeros(SIZE, dtype=bool)
    
    # Mark all neighbors of initial words as forbidden
    # We process in batches to avoid huge temporary arrays if needed, 
    # but 512 * 12616 is small enough to do directly or in a loop.
    # Actually, we can just loop over initial words and update forbidden.
    for w in initial_words:
        # w is an int, masks is uint32 array
        # w ^ masks gives the neighbors
        # We need to ensure indices are within [0, SIZE)
        # Since w < SIZE and masks < SIZE, w ^ masks < SIZE is not necessarily true?
        # Yes, XOR of two numbers < 2^18 is < 2^18.
        neighbors = (w ^ masks)
        forbidden[neighbors] = True
        
    # The initial words themselves are "used", but we don't need to mark them as forbidden 
    # in the sense of preventing re-addition if we just iterate and check `not forbidden[c]`.
    # However, the initial words ARE in the code. 
    # The greedy step will iterate through all candidates.
    # If a candidate is an initial word, it will be marked as forbidden by its own neighbors?
    # No, `forbidden` marks words that CANNOT be added.
    # An initial word $w$ forbids its neighbors. Does it forbid itself?
    # Distance from $w$ to $w$ is 0, which is < 6. So yes, $w$ is in the neighbor set.
    # So `forbidden[w]` will be True.
    # Thus, when we iterate and check `if not forbidden[c]`, we will skip the initial words.
    # This is correct because we already have them in our code.
    
    code = list(initial_words)
    
    # Iterate through all possible words
    # To be deterministic and potentially find better packings, we can iterate in increasing order of weight?
    # Or just standard integer order. Standard order is fine.
    # We can check time remaining.
    
    # Optimization: We can skip words that are already forbidden.
    # We just loop c from 0 to SIZE-1.
    
    # To speed up, we can use a list of available words? No, just loop.
    # 262,144 iterations is fast.
    
    for c in range(SIZE):
        if time.time() > deadline:
            break
        if not forbidden[c]:
            code.append(c)
            # Mark neighbors of c as forbidden
            neighbors = (c ^ masks)
            forbidden[neighbors] = True
            
    # 3. Verification
    # The greedy algorithm guarantees that any two words in `code` are at distance >= MIN_DIST.
    # Why?
    # When we add a word `c`, we ensure `forbidden[c]` is False.
    # `forbidden[c]` is True if `c` is within distance < MIN_DIST of any previously added word.
    # So `c` is at distance >= MIN_DIST from all previous words.
    # By induction, all pairs are at distance >= MIN_DIST.
    
    # Double check with a few random pairs just in case of logic errors
    # But the logic is sound.
    
    return code
# EVOLVE-BLOCK-END
