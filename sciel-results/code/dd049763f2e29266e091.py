import time
import random
import numpy as np

N = 18
NUM_INPUTS = 1 << N

# Precompute initial masks and target masks
# Wire i has value 1 for input k iff (k >> i) & 1 == 1
INITIAL_MASKS = np.zeros(N, dtype=np.uint64)
for i in range(N):
    # For each k in 0..2^18-1, bit i is 1 if k & (1<<i)
    # We can build this pattern:
    # For i=0: 1,0,1,0,1,0,...
    # For i=1: 1,1,0,0,1,1,0,0,...
    # This is a repeating pattern
    period = 1 << (i + 1)
    half = 1 << i
    # Create pattern for one period: 0s then 1s
    # Actually, let's just use a simple loop or vectorized approach
    # For efficiency, we'll use a trick with repeated patterns
    # Pattern for bit i: (k & (1<<i)) != 0
    # This is equivalent to: (k >> i) & 1
    # Let's build it as: for k in range(NUM_INPUTS): (k >> i) & 1
    # But that's slow in pure Python. Let's use numpy.
    ks = np.arange(NUM_INPUTS, dtype=np.uint64)
    bits = (ks >> i) & 1
    # Pack into uint64: bit k of the result is bits[k]
    # We can do this by: sum(bits[k] << k for k in range(NUM_INPUTS))
    # But that's O(2^18). Let's use a different approach.
    # Actually, for verification, we can use a different method.
    # Let's use Python integers for the masks, which support arbitrary precision.
    pass

# Let's use Python integers for the masks
INITIAL_MASKS_INT = [0] * N
for i in range(N):
    mask = 0
    for k in range(NUM_INPUTS):
        if (k >> i) & 1:
            mask |= (1 << k)
    INITIAL_MASKS_INT[i] = mask

# Target masks: wire i is 1 iff popcount(k) >= N - i
TARGET_MASKS_INT = [0] * N
for i in range(N):
    threshold = N - i
    mask = 0
    for k in range(NUM_INPUTS):
        if bin(k).count('1') >= threshold:
            mask |= (1 << k)
    TARGET_MASKS_INT[i] = mask


def _verify_fast(comparators):
    """Verify if the network sorts all inputs using bitset representation."""
    V = list(INITIAL_MASKS_INT)
    for i, j in comparators:
        Vi = V[i]
        Vj = V[j]
        V[i] = Vi & Vj
        V[j] = Vi | Vj
    return all(V[i] == TARGET_MASKS_INT[i] for i in range(N))


def _verify_sample(comparators, samples):
    """Verify on a sample of inputs for faster feedback."""
    for input_val in samples:
        bits = [(input_val >> i) & 1 for i in range(N)]
        for i, j in comparators:
            if bits[i] > bits[j]:
                bits[i], bits[j] = bits[j], bits[i]
        expected = [0] * N
        ones = sum(bits)
        for i in range(N - ones, N):
            expected[i] = 1
        if bits != expected:
            return False
    return True


def _remove_comparator(comparators, idx):
    """Try to remove comparator at index idx and return True if still valid."""
    if len(comparators) <= 1:
        return False
    new_comparators = comparators[:idx] + comparators[idx+1:]
    if _verify_fast(new_comparators):
        return True
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin
    
    # Start with odd-even transposition network
    comparators = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    best_comparators = list(comparators)
    best_len = len(comparators)
    
    # Precompute samples for quick verification
    random.seed(42)
    samples = [random.randint(0, NUM_INPUTS - 1) for _ in range(200)]
    
    # Phase 1: Greedy removal
    for idx in range(len(comparators) - 1, -1, -1):
        if time.time() > deadline:
            break
        if _remove_comparator(comparators, idx):
            comparators = comparators[:idx] + comparators[idx+1:]
            best_comparators = list(comparators)
            best_len = len(comparators)
    
    # Phase 2: Local search with random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start from best known
        curr = list(best_comparators)
        curr_len = len(curr)
        
        # Try random removals
        for _ in range(500):
            if time.time() > deadline:
                break
            if curr_len <= 1:
                break
            
            # Try removing a random comparator
            idx = random.randint(0, len(curr) - 1)
            if _remove_comparator(curr, idx):
                curr = curr[:idx] + curr[idx+1:]
                curr_len = len(curr)
                if curr_len < best_len:
                    best_comparators = list(curr)
                    best_len = curr_len
        
        # Try swapping comparators to enable removal
        for _ in range(100):
            if time.time() > deadline:
                break
            if curr_len <= 2:
                break
            
            # Pick two random comparators and try to remove one after adding a new one
            idx1 = random.randint(0, len(curr) - 1)
            # Try removing idx1
            if _remove_comparator(curr, idx1):
                curr = curr[:idx1] + curr[idx1+1:]
                curr_len = len(curr)
                if curr_len < best_len:
                    best_comparators = list(curr)
                    best_len = curr_len
        
        # Add a random comparator and try to remove a different one
        for _ in range(100):
            if time.time() > deadline:
                break
            if curr_len <= 1:
                break
            
            i, j = sorted(random.sample(range(N), 2))
            # Check if this comparator already exists
            if (i, j) in curr:
                continue
            
            new_curr = list(curr) + [(i, j)]
            # Try removing a random comparator from new_curr
            if len(new_curr) > 1:
                idx = random.randint(0, len(new_curr) - 1)
                if _remove_comparator(new_curr, idx):
                    new_curr = new_curr[:idx] + new_curr[idx+1:]
                    if len(new_curr) < curr_len:
                        curr = new_curr
                        curr_len = len(curr)
                        if curr_len < best_len:
                            best_comparators = list(curr)
                            best_len = curr_len
    
    return best_comparators
