import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Build the extended quadratic-residue code of length 18
    # QR(17) is a [17, 7, 5] code. Its extension is [18, 8, 6] with 256 words.
    
    # The QR code of length 17 (p=17, p≡1 mod 4)
    # Generator polynomial: g(x) = 1 + x^4 + x^7 + x^10 + x^14
    # This is the binary irreducible polynomial of degree 17
    
    # Actually, let me use a known construction.
    # The extended QR code of length 18 has 256 codewords and minimum distance 6.
    
    # Let me construct it properly.
    # The quadratic residues mod 17 are: {1, 2, 4, 8, 9, 13, 15, 16}
    # The QR code has generator polynomial related to the minimal polynomial.
    
    # For practical purposes, let me use a different approach.
    # I'll use the fact that the extended QR(17) code is equivalent to the [18, 8, 6] code.
    
    # Let me try to construct it using the BCH code approach.
    # Actually, let me just use a greedy approach starting from a good seed.
    
    # Better approach: Use the Reed-Muller RM(1,4) code extended to 18 bits
    # RM(1,4) has 32 words of length 16 with minimum distance 8.
    # We can extend this to 18 bits by adding 2 more coordinates.
    
    # Let me use a different strategy: start with a known good code and extend greedily.
    
    # Construct the [18, 8, 6] code using the following approach:
    # 1. Start with RM(1,4) on 16 coordinates (32 words, distance 8)
    # 2. Add 2 more coordinates to make it 18 bits
    # 3. Greedily extend
    
    # Actually, let me use a more direct approach.
    # The best known A_2(18,6) is 256.
    # Let me try to construct a 256-word code.
    
    # Use the extended QR code construction
    # The QR code of length 17 has generator polynomial g(x)
    # For p=17, the QR code is a [17, 7, 5] code
    
    # Let me use a simpler approach: build a good initial code and extend greedily
    
    # Start with RM(1,4) on 16 bits
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
    
    # Extend to 18 bits by adding 2 more coordinates
    # For each word, we can try different extensions
    # The simplest: add 2 zero bits
    initial_words = [w for w in rm_words]  # 16-bit words, treat as 18-bit
    
    # Better: use the extended QR code directly
    # Let me try to construct it
    
    # The extended QR(17) code can be constructed as follows:
    # Take the QR code of length 17 and add a parity check bit
    
    # For now, let me use a greedy approach with a good initial set
    
    # Use vectorized NumPy for fast compatibility checking
    
    def vectorized_can_add_batch(candidates, current_words):
        """Check if any candidates can be added to current_words"""
        if len(current_words) == 0:
            return np.ones(len(candidates), dtype=bool)
        
        # Convert to numpy arrays
        cand_arr = np.array(candidates, dtype=np.uint32)
        curr_arr = np.array(current_words, dtype=np.uint32)
        
        # For each candidate, check against all current words
        # Use broadcasting: cand_arr[:, None] ^ curr_arr[None, :]
        # This gives an (n_cand, n_curr) array
        
        # But this might be too large. Let's do it in chunks.
        results = np.zeros(len(candidates), dtype=bool)
        
        chunk_size = 1024
        for i in range(0, len(candidates), chunk_size):
            chunk = cand_arr[i:i+chunk_size]
            # (chunk_size, n_curr)
            xor_matrix = chunk[:, None] ^ curr_arr[None, :]
            # Count bits in each XOR result
            # Use a lookup table or bit manipulation
            
            # For each element in xor_matrix, count set bits
            # This is expensive. Let me use a different approach.
            
            # Actually, let me just do it in Python for now
            pass
        
        return results
    
    # Let me use a simpler but effective approach
    # Start with a good seed and greedily extend
    
    # Use the RM(1,4) code as seed
    current_words = list(rm_words)
    current_set = set(current_words)
    
    # Greedily extend by trying all 2^18 words
    # Use a priority queue or just iterate
    
    # For efficiency, let me use a different strategy:
    # 1. Try to add words in order of increasing weight
    # 2. Use early termination
    
    best_words = list(current_words)
    best_size = len(current_words)
    
    # Generate candidates in a smart order
    # Start with low-weight words, then higher weight
    
    # Actually, let me just iterate through all words and greedily add
    
    added_any = True
    while added_any and time.time() < deadline - 5.0:
        added_any = False
        for candidate in range(1 << N):
            if candidate in current_set:
                continue
            # Check if candidate can be added
            can_add = True
            for w in current_words:
                if (candidate ^ w).bit_count() < MIN_DIST:
                    can_add = False
                    break
            if can_add:
                current_words.append(candidate)
                current_set.add(candidate)
                added_any = True
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(current_words)
    
    # Try random restarts if time permits
    rng = random.Random(42)
    if time.time() < deadline - 3.0:
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            # Try a different initial set
            rng_words = [rng.randint(0, (1 << N) - 1) for _ in range(16)]
            rng_words = list(set(rng_words))
            if len(rng_words) < 2:
                continue
            
            # Verify the seed is valid
            valid = True
            for i in range(len(rng_words)):
                for j in range(i+1, len(rng_words)):
                    if (rng_words[i] ^ rng_words[j]).bit_count() < MIN_DIST:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Now greedily add from all candidates
            greedy_words = list(rng_words)
            greedy_set = set(rng_words)
            
            for candidate in range(1 << N):
                if candidate in greedy_set:
                    continue
                can_add = True
                for w in greedy_words:
                    if (candidate ^ w).bit_count() < MIN_DIST:
                        can_add = False
                        break
                if can_add:
                    greedy_words.append(candidate)
                    greedy_set.add(candidate)
            
            if len(greedy_words) > best_size:
                best_words = list(greedy_words)
                best_size = len(best_words)
    
    return best_words
