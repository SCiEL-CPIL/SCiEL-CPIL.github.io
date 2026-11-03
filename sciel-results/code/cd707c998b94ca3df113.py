import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_subset_index = {subset: i for i, subset in enumerate(t_subsets)}
    
    # NumPy masks
    # W = number of 64-bit words needed
    W = (n_t + 63) // 64
    all_mask_np = np.zeros(W, dtype=np.uint64)
    # Set all bits in the last word appropriately?
    # Actually, we just need to represent the set of all t-subsets.
    # Bit i corresponds to t_subset i.
    for i in range(n_t):
        word_idx = i // 64
        bit_idx = i % 64
        all_mask_np[word_idx] |= (1 << bit_idx)
        
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # masks_np shape (n_k, W)
    masks_np = np.zeros((n_k, W), dtype=np.uint64)
    
    for i, block in enumerate(k_subsets):
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            word_idx = idx // 64
            bit_idx = idx % 64
            masks_np[i, word_idx] |= (1 << bit_idx)
            
    # Greedy Construction with NumPy
    def greedy_construct():
        covered_np = np.zeros(W, dtype=np.uint64)
        blocks = []
        
        while not np.array_equal(covered_np, all_mask_np):
            remaining = all_mask_np ^ covered_np
            
            # Calculate gain for all blocks
            # gains[i] = sum(bitwise_and(remaining, masks_np[i]))
            # Using broadcasting
            # remaining shape (W,), masks_np shape (n_k, W)
            # np.bitwise_and(remaining[None, :], masks_np) -> (n_k, W)
            # .sum(axis=1) -> (n_k,)
            
            # Optimization: 
            # We only care about the max.
            # But we need to handle ties for randomization.
            
            ands = np.bitwise_and(remaining[None, :], masks_np)
            # Count bits. 
            # np.unpackbits(ands).sum(axis=1) is slow?
            # Better: use a lookup table or just sum if we convert to int?
            # Actually, for uint64, we can use a helper.
            # But numpy doesn't have a direct "popcount" for arrays of uint64 that is super fast?
            # We can use np.unpackbits(ands, axis=1).sum(axis=2)
            # This expands W to 64*W. 182 * 64 = 11648.
            # 75k * 11648 bytes = 800MB. Too much memory for intermediate array?
            # 75000 * 11648 * 1 byte (if unpacked to uint8) = ~870 MB. 
            # This might be too slow/memory heavy.
            
            # Alternative: Loop in Python over W words?
            # gains = np.zeros(n_k, dtype=np.int32)
            # for w in range(W):
            #    gains += np.bitwise_and(remaining[w], masks_np[:, w]) # No, need popcount
            
            # Let's stick to Python int bit_count for the greedy step if NumPy popcount is tricky.
            # Wait, the Python int approach was ~2s per construction.
            # If we can do 50 constructions, that's 100s.
            
            # Let's try a hybrid:
            # Use NumPy for the AND, then convert to Python ints for popcount?
            # No, that defeats the purpose.
            
            # Actually, `np.bitwise_and` is fast. The bottleneck is popcount.
            # We can precompute a popcount table for 16-bit chunks?
            # Or just use the Python loop but optimized?
            
            # Let's go back to Python loop but optimize the inner part.
            # The previous estimate of 2s might be pessimistic if we use local variables.
            
            pass 

    # Let's stick to the Python loop version but ensure it's as optimized as possible.
    # If it's too slow, we reduce the number of iterations.
    
    # Re-evaluating Python loop speed:
    # 22.5M iterations.
    # Simple Python loop: ~50M iterations/sec is optimistic, usually 5-10M/sec for simple ops.
    # 22.5M / 5M = 4.5 seconds.
    # 22.5M / 10M = 2.25 seconds.
    # So ~2-5 seconds per greedy run.
    # In 110s, we can do ~20-50 runs.
    # This is acceptable.
    
    # Let's implement the Python version carefully.
    
    def greedy_construct():
        covered = 0
        blocks = []
        while covered != all_mask:
            remaining = all_mask ^ covered
            best_gain = -1
            best_candidates = []
            
            # Local variables for speed
            masks = block_masks
            for i in range(n_k):
                gain = (remaining & masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
                
            chosen = random.choice(best_candidates)
            blocks.append(chosen)
            covered |= masks[chosen]
            
        return blocks

    def reduce_covering(blocks):
        changed = True
        while changed:
            changed = False
            n_blocks = len(blocks)
            if n_blocks == 0:
                break
            
            for i in range(n_blocks):
                cov = 0
                for j in range(n_blocks):
                    if j != i:
                        cov |= block_masks[blocks[j]]
                        
                if cov == all_mask:
                    blocks.pop(i)
                    changed = True
                    break
                    
        return blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run iterations
    while time.time() < deadline:
        blocks = greedy_construct()
        blocks = reduce_covering(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    if best_blocks is None:
        return []
        
    return [list(k_subsets[i]) for i in best_blocks]
