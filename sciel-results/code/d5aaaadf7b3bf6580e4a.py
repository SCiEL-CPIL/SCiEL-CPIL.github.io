import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin to ensure we finish before 150s

    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Precompute masks for all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in k_subsets:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    n_blocks = len(k_subsets)
    
    # Convert masks to a NumPy array for faster operations
    # We will use uint64 for masks, but if num_t_subsets > 64, we need a different approach
    # For C(19, 4) = 3876, we need multiple 64-bit words
    num_words = (num_t_subsets + 63) // 64
    
    # Create a 2D array of masks: shape (n_blocks, num_words)
    masks_array = np.zeros((n_blocks, num_words), dtype=np.uint64)
    for i, mask in enumerate(block_masks):
        for w in range(num_words):
            masks_array[i, w] = (mask >> (64 * w)) & ((1 << 64) - 1)
    
    # Function to compute marginal gains
    def compute_gains(uncovered_words):
        gains = np.zeros(n_blocks, dtype=np.int32)
        for w in range(num_words):
            # For each block, count the number of uncovered t-subsets it covers
            # We can use bit_count on each 64-bit word
            # Since NumPy doesn't have a direct bit_count for uint64 arrays, we'll use a trick
            # We can process in chunks or use a lookup table for 8-bit chunks
            # For simplicity, we'll use a loop over words and bit_count on individual elements
            # This might be slow, so let's try a different approach
            pass
        
        # Alternative: use a lookup table for 8-bit chunks
        # This is complex, so let's stick with a simpler approach for now
        # We'll compute gains in a loop over blocks, but vectorize where possible
        
        # For now, let's use a simpler method: compute gains for each block
        # This is O(n_blocks * num_words) which is 50388 * 61 = ~3 million operations
        # This should be fast enough
        for i in range(n_blocks):
            gain = 0
            for w in range(num_words):
                # Count bits in masks_array[i, w] & uncovered_words[w]
                val = masks_array[i, w] & uncovered_words[w]
                # Count bits using a simple method
                gain += bin(val).count('1')
            gains[i] = gain
        return gains
    
    best_result = None
    best_score = float('inf')
    
    # Run multiple randomized greedy searches
    num_restarts = 20  # Adjust based on time budget
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:  # Leave 5 seconds for final processing
            break
            
        # Initialize uncovered as all 1s
        uncovered_words = np.full(num_words, (1 << 64) - 1, dtype=np.uint64)
        # But we only have num_t_subsets bits, so the last word might have extra bits
        # We'll mask it properly
        last_word_bits = num_t_subsets % 64
        if last_word_bits == 0:
            last_word_bits = 64
        uncovered_words[-1] = (1 << last_word_bits) - 1
        
        selected_blocks = []
        
        while True:
            if time.time() > deadline - 2:  # Leave 2 seconds for safety
                break
            
            # Compute gains
            gains = compute_gains(uncovered_words)
            
            # Check if all covered
            if np.all(uncovered_words == 0):
                break
            
            # Find the best gain
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if a covering exists
                break
            
            # Randomized selection: pick a block with probability proportional to gain
            # Use a softmax-like approach
            # To avoid numerical issues, subtract max_gain
            adjusted_gains = gains - max_gain
            # Clip to avoid underflow
            adjusted_gains = np.clip(adjusted_gains, -100, 0)
            weights = np.exp(adjusted_gains)
            weights /= np.sum(weights)
            
            # Select a block
            selected_idx = np.random.choice(n_blocks, p=weights)
            
            # Update uncovered
            for w in range(num_words):
                uncovered_words[w] &= ~masks_array[selected_idx, w]
            
            selected_blocks.append(selected_idx)
            
            # Check if all covered
            if np.all(uncovered_words == 0):
                break
        
        # Convert selected indices to blocks
        result = [list(k_subsets[i]) for i in selected_blocks]
        score = len(result)
        
        if score < best_score:
            best_score = score
            best_result = result
    
    return best_result
