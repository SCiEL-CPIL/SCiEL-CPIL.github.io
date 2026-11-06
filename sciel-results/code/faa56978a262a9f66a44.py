import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 18
    MIN_DIST = 6
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # Construct the extended QR(17) code deterministically
    p = 17
    
    # Quadratic residues modulo 17: {1, 2, 4, 8, 9, 13, 15, 16}
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Create the base vector v of length 17
    v = 0
    for i in range(p):
        if i in residues:
            v |= (1 << i)
    
    # Generate the 8 cyclic shifts of v
    basis = []
    current = v
    mask_p = (1 << p) - 1
    for _ in range((p - 1) // 2):  # 8 shifts
        basis.append(current)
        current = ((current << 1) | (current >> (p - 1))) & mask_p
    
    # Add the all-ones vector (9th basis vector)
    basis.append((1 << p) - 1)
    
    num_basis = len(basis)  # 9
    
    # Vectorized generation of all 512 codewords using NumPy
    coeffs = np.arange(1 << num_basis, dtype=np.uint16)
    code_words = np.zeros(1 << num_basis, dtype=np.uint32)
    
    for i in range(num_basis):
        mask = (coeffs & (1 << i)) != 0
        code_words[mask] ^= basis[i]
    
    # Compute parity bits and form 18-bit words
    weights = np.array([w.bit_count() for w in code_words], dtype=np.uint8)
    parities = weights % 2
    full_words = (code_words.astype(np.uint32) << 1) | parities.astype(np.uint32)
    
    code = full_words.tolist()
    
    # Verify no duplicates
    if len(set(code)) != len(code):
        code = list(set(code))
    
    best_code = list(code)
    best_size = len(best_code)
    
    # Convert to numpy array for vectorized distance checks
    code_arr = np.array(best_code, dtype=np.uint32)
    
    # Phase 1: Deterministic structured search for additional words
    # Enumerate all 18-bit words by weight 6 to 12
    # Use vectorized checks against the current code
    
    # Pre-generate all candidates by weight for efficient iteration
    # Total candidates: sum(C(18,w) for w in 6..12)
    # C(18,6)=18564, C(18,7)=31824, C(18,8)=43758, C(18,9)=48620,
    # C(18,10)=43758, C(18,11)=31824, C(18,12)=18564 => ~236,912 total
    
    def generate_words_of_weight(n, w):
        """Generate all n-bit words with exactly w bits set."""
        if w == 0:
            yield 0
            return
        if w == n:
            yield (1 << n) - 1
            return
        # Use combinations of positions
        from itertools import combinations as comb
        for positions in comb(range(n), w):
            word = 0
            for pos in positions:
                word |= (1 << pos)
            yield word
    
    # Try to add words, starting from lower weights (more constrained)
    # Use vectorized distance check
    def can_add_vectorized(w, code_arr, min_dist):
        """Check if word w can be added to code_arr with min distance min_dist."""
        xor_vals = code_arr ^ w
        # Count bits in each xor value
        # Use a fast method: convert to bytes and use lookup or bit manipulation
        # For 512 words, this is fast enough
        dists = np.array([v.bit_count() for v in xor_vals], dtype=np.uint8)
        return np.min(dists) >= min_dist
    
    # Phase 1: Try adding words by weight
    # Process weights in order, trying to add as many as possible
    # Use a priority: lower weight first (more constrained positions)
    
    # To speed up, we'll batch process: generate candidates, filter by vectorized check
    # But 236K candidates * 512 checks each = ~120M operations, might be too slow
    
    # Instead, use a smarter approach: for each weight, generate candidates,
    # check them in batches using NumPy
    
    # Actually, let's try a more targeted approach:
    # The QR code is highly structured. Let's try specific families of candidates.
    
    # Alternative: Use a greedy approach with random sampling but much smarter
    # Focus on words that are "far" from the existing code structure
    
    # Let's try a hybrid: random sampling with vectorized checks, but more attempts
    # and with a better candidate distribution
    
    random.seed(42)
    attempts = 0
    max_attempts = 500000
    
    # For speed, use vectorized check in batches
    batch_size = 1000
    
    while time.time() < deadline - 10.0 and attempts < max_attempts:
        # Generate a batch of random candidates
        batch_end = min(attempts + batch_size, max_attempts)
        candidates = []
        
        for _ in range(batch_end - attempts):
            w = random.randint(0, (1 << N) - 1)
            wt = w.bit_count()
            if 6 <= wt <= 12:
                candidates.append(w)
        
        if not candidates:
            attempts = batch_end
            continue
        
        # Vectorized check: for each candidate, compute min distance to code
        # This is O(len(candidates) * len(code_arr)) which is 1000 * 512 = 512K ops per batch
        # That's fast with NumPy if we vectorize properly
        
        cand_arr = np.array(candidates, dtype=np.uint32)
        
        # Compute all pairwise distances: (batch_size, code_size)
        # xor_vals[i,j] = cand_arr[i] ^ code_arr[j]
        # This creates a large array: 1000 x 512 = 512K uint32 values = 2MB, fine
        
        # But doing this for each batch might be slow. Let's do it more efficiently.
        # For each candidate, we need min distance. We can do:
        
        valid_indices = []
        for idx, w in enumerate(candidates):
            if time.time() > deadline - 10.0:
                break
            xor_vals = code_arr ^ w
            # Fast bit count using NumPy's pack/unpack trick or just list comprehension
            # For 512 values, list comprehension is fast enough
            dists = [v.bit_count() for v in xor_vals]
            if min(dists) >= MIN_DIST:
                valid_indices.append(idx)
        
        # Add all valid candidates
        for idx in valid_indices:
            w = candidates[idx]
            if w not in best_code:
                best_code.append(w)
                best_size = len(best_code)
                code_arr = np.array(best_code, dtype=np.uint32)
        
        attempts = batch_end
    
    # Phase 2: Swap-based local search (if time remains)
    # Try to swap pairs of codewords to find a larger code
    # This is more expensive but might find improvements
    
    if time.time() < deadline - 5.0 and len(best_code) < 600:
        # Try limited swap search
        random.seed(123)
        swap_attempts = 0
        max_swap_attempts = 50000
        
        while time.time() < deadline - 5.0 and swap_attempts < max_swap_attempts:
            swap_attempts += 1
            
            # Pick two random codewords to potentially replace
            if len(best_code) < 2:
                break
            
            i, j = random.sample(range(len(best_code)), 2)
            
            # Generate random replacement for codeword i
            w_new = random.randint(0, (1 << N) - 1)
            wt_new = w_new.bit_count()
            if wt_new < 6 or wt_new > 12:
                continue
            
            # Check if w_new can replace best_code[i]
            # Remove best_code[i], check if w_new is valid against remaining
            remaining = [c for k, c in enumerate(best_code) if k != i]
            
            valid = True
            for c in remaining:
                if (w_new ^ c).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid and w_new not in remaining:
                # Also check distance to best_code[j] specifically (already in remaining)
                # Replace
                best_code[i] = w_new
                # Don't increase size, just swap
    
    return best_code
