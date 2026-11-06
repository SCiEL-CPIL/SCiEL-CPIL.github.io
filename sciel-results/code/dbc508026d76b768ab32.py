import time
import numpy as np

N = 18
DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Generate all 2^18 Boolean inputs as a (262144, 18) uint8 array
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1
        
    # State: each wire's value for each input
    state = all_inputs.copy()
    network = []
    
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def compute_suffix_zeros(s):
        """Compute suffix count of 0s for fast gain lookup.
        suffix_zeros[:, k] = number of 0s in s[:, k:N]
        """
        # 1 - s is 1 where s is 0
        zeros = 1 - s
        # Cumulative sum from right
        # np.cumsum doesn't do right-to-left easily without flip
        # flip, cumsum, flip back
        cumsum_zeros = np.cumsum(zeros[:, ::-1], axis=1)[:, ::-1]
        # We want suffix_zeros[:, k] to be sum of zeros from k to N-1
        # The above gives sum from k to N-1?
        # Let's check:
        # zeros = [z0, z1, ..., z17]
        # flipped = [z17, ..., z0]
        # cumsum = [z17, z17+z16, ...]
        # flip back = [sum(z0..z17), sum(z1..z17), ..., z17]
        # So cumsum_zeros[:, k] is sum of zeros from k to N-1.
        # This is exactly what we want for Z_k.
        # We need Z_{j+1} which is sum of zeros from j+1 to N-1.
        # So we just use cumsum_zeros[:, j+1].
        return cumsum_zeros

    def is_sorted(s):
        """Check if all inputs are sorted (non-decreasing across wires)."""
        # s[:, i] <= s[:, i+1] for all i
        # Equivalent to: no (1, 0) pair in adjacent wires
        # Or: max of s[:, i] - s[:, i+1] is 0?
        # Using np.any on a boolean mask is fast
        for i in range(N - 1):
            if np.any((s[:, i] == 1) & (s[:, i+1] == 0)):
                return False
        return True

    # Phase 1: Greedy construction
    while True:
        if time.time() > DEADLINE - 15:  # Leave 15s for refinement
            break
        
        if is_sorted(state):
            break
        
        # Compute suffix zeros for current state
        suffix_zeros = compute_suffix_zeros(state)
        
        best_score = -np.inf
        best_comp = None
        
        # Evaluate all comparators
        for i, j in all_comparators:
            # Only consider inputs where state[i]=1 and state[j]=0
            mask = (state[:, i] == 1) & (state[:, j] == 0)
            if not np.any(mask):
                continue
            
            # Score for each affected row: (j - i) - suffix_zeros[:, j+1]
            # We want to maximize the sum of these scores
            # Note: suffix_zeros[:, j+1] is the count of 0s after j
            # If j is N-1, j+1 is N, which is out of bounds for the array?
            # suffix_zeros has shape (2^N, N). Indices 0..N-1.
            # If j = N-1, we need sum of zeros from N to N-1, which is 0.
            # So we need to handle the boundary.
            
            if j + 1 < N:
                scores = (j - i) - suffix_zeros[mask, j+1]
            else:
                # suffix_zeros[:, N] would be 0
                scores = np.full(np.sum(mask), (j - i), dtype=np.int32)
            
            total_score = np.sum(scores)
            
            if total_score > best_score:
                best_score = total_score
                best_comp = (i, j)
        
        if best_comp is None or best_score <= 0:
            # No progress or stuck? 
            # If best_score <= 0, it means no comparator reduces inversions?
            # But if not sorted, there must be an inversion.
            # An inversion implies a 1 before a 0.
            # Swapping them should reduce inversions.
            # So best_score should be positive.
            # If it's not, we might be in a local minimum or error.
            # Let's just break to avoid infinite loop.
            break
        
        # Apply the best comparator
        i, j = best_comp
        # Swap values in state where mask is true?
        # No, compare-exchange: min to i, max to j.
        # Since we only apply to rows where state[i]=1, state[j]=0,
        # min is 0 (from j) and max is 1 (from i).
        # So we set state[:, i] = 0 and state[:, j] = 1 for those rows.
        
        mask = (state[:, i] == 1) & (state[:, j] == 0)
        state[:, i] = np.where(mask, 0, state[:, i])
        state[:, j] = np.where(mask, 1, state[:, j])
        
        network.append((i, j))
    
    # If we didn't finish sorting, use seed network
    if not is_sorted(state):
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Phase 2: Refinement - try to remove redundant comparators
    def replay_network(net, skip_idx=None):
        """Replay network, optionally skipping one comparator. Returns final state."""
        s = all_inputs.copy()
        for idx, (i, j) in enumerate(net):
            if idx == skip_idx:
                continue
            # Vectorized compare-exchange
            # min_val = np.minimum(s[:, i], s[:, j])
            # max_val = np.maximum(s[:, i], s[:, j])
            # s[:, i] = min_val
            # s[:, j] = max_val
            
            # Faster:
            # swap = (s[:, i] > s[:, j])
            # s[:, i] = np.where(swap, s[:, j], s[:, i])
            # s[:, j] = np.where(swap, s[:, i], s[:, j]) # Careful: s[:, i] is already updated?
            # No, np.where evaluates both branches fully.
            
            # Safest:
            min_vals = np.minimum(s[:, i], s[:, j])
            max_vals = np.maximum(s[:, i], s[:, j])
            s[:, i] = min_vals
            s[:, j] = max_vals
        return s

    refined_network = list(network)
    
    # Iteratively try to remove comparators, starting from the end
    changed = True
    while changed:
        changed = False
        if time.time() > DEADLINE:
            break
        
        for idx in range(len(refined_network) - 1, -1, -1):
            if time.time() > DEADLINE:
                break
            
            # Try removing comparator at idx
            test_state = replay_network(refined_network, skip_idx=idx)
            
            if is_sorted(test_state):
                # We can remove this comparator
                refined_network.pop(idx)
                changed = True
                break  # Restart from end after modification
    
    return refined_network
