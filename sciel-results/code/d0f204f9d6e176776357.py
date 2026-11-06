"""
Combines Big-Int bit-parallel verification (from Inspiration) with Simulated Annealing and Greedy Pruning (from Current).
Uses Python's arbitrary-precision integers for fast bitwise operations on the 2^18 input space.
"""
import time
import random
import sys

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE_MARGIN = 5.0  # Seconds to keep before hard limit
HARD_DEADLINE = 60.0   # Evaluator limit

# Global masks for fast verification
# masks[i] is a Python int where bit k is 1 if input k has bit i set.
# This allows processing all 2^18 inputs in parallel using bitwise ops.
def _init_masks():
    masks = []
    for i in range(N):
        # Pattern: 2^i zeros, 2^i ones.
        # Block length L = 2^(i+1)
        L = 1 << (i + 1)
        # Value of the block: (2^i - 1) shifted left by 2^i
        block_val = (1 << (1 << i)) - 1
        block_val <<= (1 << i)
        
        # Number of blocks
        num_blocks = TOTAL_INPUTS // L
        
        # Geometric series factor G = 1 + 2^L + 2^(2L) + ... + 2^((K-1)L)
        # G = (2^(K*L) - 1) / (2^L - 1)
        # K*L = TOTAL_INPUTS
        # So G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        
        # This creates a very large integer, but Python handles it.
        # (1 << TOTAL_INPUTS) is 2^262144, which is ~32KB.
        # The division is exact.
        
        # Optimization: Instead of huge division, we can build it iteratively or use the formula.
        # The formula is efficient enough for 18 masks.
        
        # However, calculating (1 << TOTAL_INPUTS) is expensive if done naively? No, it's just a shift.
        # The division of a 32KB integer by a smaller integer is fast.
        
        # Let's use the iterative multiplication to be safe and potentially faster for small L?
        # Actually, the geometric series formula is mathematically clean.
        # Let's stick to the formula but be careful with memory.
        
        # Alternative:
        # mask = 0
        # for k in range(num_blocks):
        #     mask |= block_val << (k * L)
        # For i=0, L=2, num_blocks=131072. This loop is too slow in Python.
        # So we MUST use the geometric series or similar bulk operation.
        
        # Let's use the geometric series.
        # G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        
        # To avoid creating the huge (1 << TOTAL_INPUTS) number every time if it's cached?
        # It's only done 18 times.
        
        denom = (1 << L) - 1
        numerator = (1 << TOTAL_INPUTS) - 1
        G = numerator // denom
        
        mask = block_val * G
        masks.append(mask)
    return masks

# Pre-compute masks at import time to avoid overhead in construct()
_MASKS = _init_masks()

def _verify_network(comparators):
    """
    Verify if the network sorts all Boolean inputs using Big-Int bit-parallelism.
    Returns True if valid.
    """
    # Copy the initial state (list of ints)
    wires = _MASKS[:]
    
    for i, j in comparators:
        wi = wires[i]
        wj = wires[j]
        # min(wi, wj) = wi & wj
        # max(wi, wj) = wi | wj
        wires[i] = wi & wj
        wires[j] = wi | wj
        
    # Check sorted: for all k, wires[k] <= wires[k+1]
    # This means no bit is 1 in wires[k] if it is 0 in wires[k+1].
    # So (wires[k] & ~wires[k+1]) must be 0.
    for k in range(N - 1):
        if (wires[k] & ~wires[k + 1]) != 0:
            return False
    return True

def _odd_even_transposition():
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _bubble_sort_network():
    comparators = []
    for i in range(N - 1):
        for j in range(N - 1 - i):
            comparators.append((j, j + 1))
    return comparators

def _random_dense_network(num_comparators=150, seed=None):
    if seed is not None:
        random.seed(seed)
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i != j:
            if i > j:
                i, j = j, i
            comparators.append((i, j))
    random.shuffle(comparators)
    return comparators

def _greedy_remove(comparators, time_limit):
    """
    Iteratively remove redundant comparators.
    Uses full passes if time permits, otherwise random sampling.
    """
    current = list(comparators)
    changed = True
    start_time = time.time()
    
    while changed:
        changed = False
        if time.time() > time_limit:
            break
        n = len(current)
        if n == 0:
            break
        
        # If network is small, check all. If large, sample.
        # With Big-Int verification, full passes are much cheaper.
        # Let's try full passes first.
        
        # To avoid bias towards index 0, we can shuffle indices or just iterate.
        # Iterating 0 to n-1 is fine, but if we remove one, we restart or continue?
        # Standard greedy: try remove 0, if fail try 1... if any succeeds, restart loop.
        
        # Optimization: Check indices in random order to escape local biases?
        # Or just standard order.
        
        # Let's do a full pass.
        for idx in range(n):
            if time.time() > time_limit:
                return current
            
            # Skip if index is out of bounds (shouldn't happen in this loop structure)
            test = current[:idx] + current[idx + 1:]
            if _verify_network(test):
                current = test
                changed = True
                break # Restart the while loop
        
    return current

def _simulated_annealing(initial_net, t_max=500.0, t_min=0.5, alpha=0.998, max_iter=10000, deadline=None):
    """
    Simulated Annealing to minimize comparator count.
    """
    current = list(initial_net)
    current_energy = len(current)
    best = list(current)
    best_energy = current_energy
    
    t = t_max
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        r = random.random()
        candidate = list(current)
        valid_mutation = True
        
        if r < 0.5:
            # Remove
            if len(candidate) == 0:
                valid_mutation = False
            else:
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
        elif r < 0.7:
            # Insert
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                valid_mutation = False
            else:
                if u > v:
                    u, v = v, u
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (u, v))
        elif r < 0.85:
            # Swap
            if len(candidate) < 2:
                valid_mutation = False
            else:
                i1, i2 = random.sample(range(len(candidate)), 2)
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        else:
            # Replace
            if len(candidate) == 0:
                valid_mutation = False
            else:
                idx = random.randint(0, len(candidate) - 1)
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    valid_mutation = False
                else:
                    if u > v:
                        u, v = v, u
                    candidate[idx] = (u, v)

        if not valid_mutation:
            continue

        if not _verify_network(candidate):
            continue # Invalid, reject

        new_energy = len(candidate)
        
        if new_energy < current_energy:
            # Always accept improvement
            current = candidate
            current_energy = new_energy
            if new_energy < best_energy:
                best = list(current)
                best_energy = new_energy
        else:
            # Accept with probability exp(-delta_e / t)
            delta = new_energy - current_energy
            if delta > 0:
                # Use math.exp for speed? random.random() is fast.
                # delta is small (1 or 2 usually).
                prob = 2.718281828 ** (-delta / t)
                if random.random() < prob:
                    current = candidate
                    current_energy = new_energy
            
        t *= alpha
        if t < t_min:
            break
        
    return best

def construct():
    start_time = time.time()
    deadline = start_time + HARD_DEADLINE - DEADLINE_MARGIN

    # Generate starting candidates
    candidates = []
    candidates.append(("odd_even", _odd_even_transposition()))
    candidates.append(("bubble", _bubble_sort_network()))
    
    # Add some random dense starts
    for i in range(3):
        candidates.append((f"random_{i}", _random_dense_network(150, seed=100 + i)))

    best_network = None
    best_count = float('inf')

    # Run SA on each candidate
    for name, net in candidates:
        if time.time() > deadline - 5.0:
            break
        
        # 1. Quick Greedy Prune
        pruned = _greedy_remove(net, time_limit=start_time + 5.0)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)
        
        # 2. Run SA from the pruned state
        # Allocate remaining time for SA, leaving some for final polish
        sa_deadline = deadline - 10.0
        sa_result = _simulated_annealing(
            pruned, 
            t_max=500, 
            t_min=0.5, 
            alpha=0.998, 
            max_iter=10000, 
            deadline=sa_deadline
        )
        
        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_network = list(sa_result)
            
        # 3. Final Greedy Remove on the SA result (using remaining time)
        final_pruned = _greedy_remove(best_network, time_limit=deadline - 2.0)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    if best_network is None:
        # Fallback
        best_network = _odd_even_transposition()
        best_network = _greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not _verify_network(best_network):
        # Should not happen, but safety
        fallback = _odd_even_transposition()
        fallback = _greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network

# Self-test on import to ensure masks are correct and basic network works
if __name__ == "__main__":
    # Simple check
    net = _odd_even_transposition()
    assert _verify_network(net), "Odd-even transposition should be valid"
    print("Masks and verification initialized successfully.")
    # Time a single verification to estimate budget
    start = time.time()
    for _ in range(100):
        _verify_network(net)
    elapsed = time.time() - start
    print(f"100 verifications took {elapsed:.4f}s. Avg: {elapsed/100*1000:.2f}ms")
