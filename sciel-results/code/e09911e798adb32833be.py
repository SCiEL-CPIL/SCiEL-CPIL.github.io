import numpy as np
import time

N = 16

def _verify_network(comparators, n=16):
    """
    Verifies if the network sorts all binary inputs.
    Uses NumPy for speed.
    """
    inputs = np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2**n)])
    for i, j in comparators:
        mask = inputs[:, i] > inputs[:, j]
        if np.any(mask):
            inputs[mask, i] = 1 - inputs[mask, i]
            inputs[mask, j] = 1 - inputs[mask, j]
    return np.array_equal(inputs, np.sort(inputs, axis=1))

def _batcher_oes(n):
    """
    Generates Batcher's Odd-Even Merge Sort network.
    Returns a list of (i, j) pairs.
    """
    if n == 1:
        return []
    
    # Helper: merge two sorted lists of length n/2
    # The lists are at indices offset + step*k for k=0..n-1
    # First list: even indices (offset, offset+step, ...)
    # Second list: odd indices (offset+step, offset+2*step, ...)
    def merge(n, offset=0, step=2):
        if n == 1:
            return []
        half = n // 2
        
        # Recursively merge odd-indexed elements
        # These are at offset + step, offset + 3*step, ...
        # They form a sequence of length half
        odd_comps = merge(half, offset + step, step * 2)
        # Remap: index k in the odd subproblem corresponds to offset + step + k*step*2
        # But we need to map back to original indices
        # In the subproblem, indices are 0..half-1
        # In the original, they are offset+step, offset+3*step, ...
        # So subproblem index k -> original index offset + step + k*(step*2)
        # Wait, the step in the recursive call is step*2
        # So subproblem index k -> offset + step + k * (step*2)
        # But we want to map subproblem (i,j) to original indices
        # Subproblem index k corresponds to original index offset + step + k * (step * 2)
        
        # Actually, let me think differently.
        # The merge function takes n elements at positions:
        # offset, offset+step, offset+2*step, ..., offset+(n-1)*step
        # The first half (even positions in this sequence) are at:
        # offset, offset+2*step, offset+4*step, ...
        # The second half (odd positions) are at:
        # offset+step, offset+3*step, offset+5*step, ...
        
        # So to merge the odd-indexed subsequence:
        # These are at offset+step, offset+3*step, ...
        # This is a sequence of length half with starting offset offset+step and step 2*step
        odd_comps = merge(half, offset + step, 2 * step)
        
        # To merge the even-indexed subsequence:
        # These are at offset, offset+2*step, ...
        # This is a sequence of length half with starting offset offset and step 2*step
        even_comps = merge(half, offset, 2 * step)
        
        # Final stage: compare adjacent pairs
        # Compare (offset + 2k*step, offset + (2k+1)*step) for k=0..half-1
        final_comps = []
        for k in range(half):
            i = offset + 2 * k * step
            j = offset + (2 * k + 1) * step
            final_comps.append((i, j))
        
        return odd_comps + even_comps + final_comps
    
    # Sorting network S(n):
    # 1. S(n/2) on even indices
    # 2. S(n/2) on odd indices
    # 3. Merge(n)
    
    half = n // 2
    
    # S(half) on even indices: 0, 2, 4, ...
    even_sort = _batcher_oes(half)
    # Remap: subproblem index k -> original index 2*k
    even_sort_remap = [(2 * i, 2 * j) for (i, j) in even_sort]
    
    # S(half) on odd indices: 1, 3, 5, ...
    odd_sort = _batcher_oes(half)
    # Remap: subproblem index k -> original index 2*k + 1
    odd_sort_remap = [(2 * i + 1, 2 * j + 1) for (i, j) in odd_sort]
    
    # Merge(n)
    merge_comps = merge(n, 0, 1)
    
    return even_sort_remap + odd_sort_remap + merge_comps

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit, leave 5s margin
    
    # Precompute all 2^16 binary inputs
    inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)])
    
    def verify_fast(comparators):
        state = inputs.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                state[mask, i] = 1 - state[mask, i]
                state[mask, j] = 1 - state[mask, j]
        return np.array_equal(state, np.sort(state, axis=1))
    
    # Generate Batcher's OES network
    network = _batcher_oes(N)
    
    # Verify it's correct
    if not verify_fast(network):
        # Fallback to OET if OES generation is wrong
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    current_network = network[:]
    
    # Greedy peeling: try to remove comparators from end to start
    # Multiple passes for better results
    for _pass in range(3):
        if time.time() > deadline - 5.0:
            break
        
        for k in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break
            
            trial_network = current_network[:k] + current_network[k+1:]
            if verify_fast(trial_network):
                current_network = trial_network
    
    return current_network
# EVOLVE-BLOCK-END
