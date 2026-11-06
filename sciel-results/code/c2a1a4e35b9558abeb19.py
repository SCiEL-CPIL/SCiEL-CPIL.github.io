"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

MINIMUM_DISTANCE = 6
N = 18

def _seed_words():
    # First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    # Every affine Boolean function on F_2^4 is a word: 32 words, distance 8.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _min_distance_to_code(word, code):
    # Bit-parallel minimum Hamming distance to any codeword.
    # Early exit as soon as a pair with distance < MINIMUM_DISTANCE is found.
    for c in code:
        d = (word ^ c).bit_count()
        if d < MINIMUM_DISTANCE:
            return d
    return MINIMUM_DISTANCE

def _try_add_word(word, code):
    if _min_distance_to_code(word, code) >= MINIMUM_DISTANCE:
        code.append(word)
        return True
    return False

def _greedy_construct(deadline):
    code = _seed_words()
    n = len(code)
    # Deterministic candidate order: 0..2**N-1.
    # We scan candidates in order and add them if feasible.
    # To make progress faster, we can also try to add words that are far from existing ones,
    # but a simple sequential scan is deterministic and predictable.
    # Optimization: Precompute nothing, just scan.
    # However, scanning 2**18 = 262144 candidates, each against up to M codewords.
    # M will grow. Let's estimate cost.
    # If M ~ 100, cost per candidate ~ 100 XOR+popcount. 262k * 100 = 26M ops. Fast.
    # If M ~ 1000, cost ~ 262M ops. Might be slow in Python.
    # We need to be careful with the deadline.
    
    # A better strategy:
    # 1. Start with seed.
    # 2. Loop through candidates.
    # 3. If time is running out, break.
    
    # To improve efficiency, we can maintain a list of 'active' constraints?
    # No, simple loop is fine for small N.
    
    # Let's use a while loop over candidates to allow early termination.
    # But we want to maximize count.
    # A common heuristic for packing: 
    #   - Generate random candidates? No, deterministic is preferred for reproducibility unless seeded.
    #   - The task allows linear/nonlinear. Greedy gives nonlinear.
    
    # Let's try a smarter candidate generation:
    #   - Start with all words.
    #   - Check distance to current code.
    #   - If OK, add.
    
    # To speed up the inner loop, we can use NumPy?
    # "Python standard library, NumPy and SciPy are available."
    # Using NumPy for vectorized XOR and popcount could be much faster.
    
    # Let's implement a NumPy-based version for the inner check.
    # However, converting Python ints to NumPy arrays repeatedly might have overhead.
    # But if we keep the code as a NumPy array of uint16 or uint32, we can vectorize.
    
    # Let's stick to pure Python first but with a tight deadline check.
    # If pure Python is too slow, we might not finish scanning all 262k candidates.
    # But we don't need to scan ALL if we find a good packing early?
    # Actually, we want the LARGEST possible collection.
    # Greedy is order-dependent.
    
    # Alternative: 
    # Use the fact that we have 60s.
    # Let's try to use NumPy for the distance check to allow scanning more candidates or checking larger codes faster.
    
    # Revised Plan with NumPy:
    # 1. Initialize code as list of ints, then convert to np.array for checks.
    # 2. Candidate loop.
    # 3. For each candidate, compute distances to all existing codewords using vectorized XOR and popcount.
    # 4. Check min distance.
    
    # Let's implement this.
    
    import numpy as np
    
    # Convert seed to numpy array
    code_np = np.array(code, dtype=np.uint32)
    
    # We will iterate through candidates.
    # To make it deterministic and potentially better, we can shuffle? 
    # No, "deterministic" usually implies fixed seed or fixed order. 
    # Fixed order 0..2**N-1 is fine.
    
    # But wait, if we just scan 0..262143, we might find a local optimum quickly.
    # Is there a better order? 
    # Maybe random order with a fixed seed? 
    # The prompt doesn't forbid randomness, but deterministic is safer for "same result every time" if implied.
    # However, "largest possible" suggests optimization.
    # Let's stick to sequential for now, but use NumPy to check speed.
    
    # Actually, checking 262k candidates * M words.
    # If M=100, 26M ops. NumPy can do this in milliseconds.
    # So we can easily scan the entire space of 2**18 candidates.
    # If we scan the entire space, we will find the maximum set that can be packed by this greedy order.
    # Since we scan ALL candidates, the result depends on the order.
    # Scanning 0, 1, 2... might not be optimal.
    # But it's a valid heuristic.
    
    # Can we do better?
    # Maybe try multiple passes? 
    # Pass 1: Add words if distance >= 6.
    # Pass 2: Try to remove words to add more? (Too complex for 60s?)
    
    # Let's just do a single full scan with NumPy.
    
    # Deadline check
    start_time = time.time()
    
    # We scan all candidates
    for candidate in range(1 << N):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Check distance to existing code
        # If code is empty, add. (It's not empty)
        if len(code_np) == 0:
            code.append(candidate)
            code_np = np.array(code, dtype=np.uint32)
            continue
            
        # Vectorized XOR
        # code_np is shape (M,). candidate is scalar.
        # XOR: code_np ^ candidate
        # Popcount: np.bit_count is available in newer NumPy?
        # If not, use a lookup table or bit manipulation.
        # np.bit_count is in NumPy 1.24+. Let's assume it's available or use a fallback.
        # To be safe, let's use a standard bit count method if np.bit_count is missing, 
        # but given "modern" environment, it's likely there.
        # Fallback: 
        # def popcount(x): return (x & 1) + ((x >> 1) & 1) ... too slow.
        # Let's assume np.bit_count exists.
        
        try:
            dists = np.bit_count(code_np ^ candidate)
        except AttributeError:
            # Fallback for older NumPy
            # This is slow, but better than nothing
            dists = np.array([(c ^ candidate).bit_count() for c in code_np])
            
        if dists.min() >= MINIMUM_DISTANCE:
            code.append(candidate)
            # Update code_np
            # Resizing numpy array is expensive. 
            # We can pre-allocate or just convert at the end?
            # If we convert at the end, we lose the vectorization benefit for the *next* candidate?
            # No, we need code_np to be current for the next candidate.
            # Appending to a list and converting every time is O(M^2) overall.
            # M might be ~100-1000. M^2 = 100k-1M. That's fine.
            code_np = np.array(code, dtype=np.uint32)
            
    return code

def construct():
    # 90s wall-clock limit, keep safety margin.
    # Let's aim for 85s.
    deadline = time.time() + 85.0
    
    # We can also try to improve the initial seed or order.
    # But let's start with the simple full scan.
    
    # One optimization: 
    # If we scan 0..2**N-1, we might spend time on candidates that are close to existing ones.
    # But with NumPy, the check is fast.
    
    # Let's run the greedy scan.
    result = _greedy_construct(deadline)
    
    # Ensure we return a non-empty list of distinct integers
    # The greedy algorithm ensures distinctness and distance.
    return result
# EVOLVE-BLOCK-END
