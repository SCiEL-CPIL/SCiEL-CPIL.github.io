import time
import random
import numpy as np

N = 18
NUM_INPUTS = 1 << N

# Precompute initial masks and target masks using Python integers
# Wire i has value 1 for input k iff (k >> i) & 1 == 1
INITIAL_MASKS_INT = [0] * N
for i in range(N):
    mask = 0
    # Pattern: for bit i, the pattern repeats with period 2^(i+1)
    # First 2^i values have bit i = 0, next 2^i have bit i = 1
    period = 1 << (i + 1)
    half = 1 << i
    for start in range(0, NUM_INPUTS, period):
        for k in range(start + half, start + period):
            if k < NUM_INPUTS:
                mask |= (1 << k)
    INITIAL_MASKS_INT[i] = mask

# Target masks: wire i is 1 iff popcount(k) >= N - i
TARGET_MASKS_INT = [0] * N
# Precompute popcount for all inputs
_popcount_cache = [0] * NUM_INPUTS
for k in range(1, NUM_INPUTS):
    _popcount_cache[k] = _popcount_cache[k >> 1] + (k & 1)

for i in range(N):
    threshold = N - i
    mask = 0
    for k in range(NUM_INPUTS):
        if _popcount_cache[k] >= threshold:
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


def _odd_even_transposition():
    """Generate odd-even transposition network."""
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _bubble_sort_network():
    """Generate bubble sort network."""
    comparators = []
    for i in range(N):
        for j in range(N - 1 - i):
            comparators.append((j, j + 1))
    return comparators


def _bitonic_sort_network():
    """Generate bitonic sort network."""
    comparators = []
    
    def bitonic_sort(start, length, direction):
        if length <= 1:
            return
        half = length // 2
        bitonic_sort(start, half, 1)
        bitonic_sort(start + half, half, 0)
        bitonic_merge(start, length, direction)
    
    def bitonic_merge(start, length, direction):
        if length <= 1:
            return
        half = length // 2
        for i in range(half):
            a = start + i
            b = start + half + i
            if direction == 0:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(start, half, direction)
        bitonic_merge(start + half, half, direction)
    
    bitonic_sort(0, N, 0)
    return comparators


def _random_network(num_comparators):
    """Generate a random network."""
    comparators = []
    for _ in range(num_comparators):
        i, j = sorted(random.sample(range(N), 2))
        comparators.append((i, j))
    return comparators


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s limit
    
    # Start with odd-even transposition network (153 comparators)
    comparators = _odd_even_transposition()
    
    best_comparators = list(comparators)
    best_len = len(comparators)
    
    # Precompute samples for quick verification
    random.seed(42)
    samples = [random.randint(0, NUM_INPUTS - 1) for _ in range(500)]
    
    # Phase 1: Greedy removal from the end
    for idx in range(len(comparators) - 1, -1, -1):
        if time.time() > deadline:
            break
        if _remove_comparator(comparators, idx):
            comparators = comparators[:idx] + comparators[idx+1:]
            best_comparators = list(comparators)
            best_len = len(comparators)
    
    # Phase 2: Multiple passes of greedy removal
    pass_count = 0
    while time.time() < deadline:
        pass_count += 1
        improved = False
        for idx in range(len(comparators) - 1, -1, -1):
            if time.time() > deadline:
                break
            if _remove_comparator(comparators, idx):
                comparators = comparators[:idx] + comparators[idx+1:]
                improved = True
        if not improved:
            break
        best_comparators = list(comparators)
        best_len = len(comparators)
    
    # Phase 3: Local search with random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from best known
        curr = list(best_comparators)
        curr_len = len(curr)
        
        # Try random removals
        for _ in range(1000):
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
        for _ in range(500):
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
        for _ in range(500):
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
    
    # Phase 4: Try different starting networks
    for net_gen in [_bubble_sort_network, _bitonic_sort_network]:
        if time.time() > deadline:
            break
        
        curr = net_gen()
        curr_len = len(curr)
        
        # Greedy removal
        for idx in range(len(curr) - 1, -1, -1):
            if time.time() > deadline:
                break
            if _remove_comparator(curr, idx):
                curr = curr[:idx] + curr[idx+1:]
                curr_len = len(curr)
        
        if curr_len < best_len:
            best_comparators = list(curr)
            best_len = curr_len
        
        # Local search from this starting point
        for _ in range(500):
            if time.time() > deadline:
                break
            if curr_len <= 1:
                break
            
            idx = random.randint(0, len(curr) - 1)
            if _remove_comparator(curr, idx):
                curr = curr[:idx] + curr[idx+1:]
                curr_len = len(curr)
                if curr_len < best_len:
                    best_comparators = list(curr)
                    best_len = curr_len
    
    # Final verification
    if not _verify_fast(best_comparators):
        # Fallback to a known good network
        best_comparators = _odd_even_transposition()
    
    return best_comparators
