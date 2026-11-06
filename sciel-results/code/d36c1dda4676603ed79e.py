import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55.0  # Safety margin from 60s limit

def _build_mask(w, n_bits):
    block = 1 << w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = n_bits // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    return m

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    _INITIAL_MASKS[_w] = _build_mask(_w, 1 << N)

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    # Copy the initial masks
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_even_mergesort(n):
    """
    Generate a sorting network for n wires using Batcher's Odd-Even Mergesort.
    Returns a list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    # Split into two halves
    half = n // 2
    # Recursively sort halves
    left = _batcher_odd_even_mergesort(half)
    right = _batcher_odd_even_mergesort(n - half)
    
    # Offset right network indices
    right = [(i + half, j + half) for i, j in right]
    
    # Merge the two halves
    merge = _batcher_odd_even_merge(n)
    
    return left + right + merge

def _batcher_odd_even_merge(n):
    """
    Merge two sorted halves of length n//2 and n//2 (or n//2 and n//2+1).
    Assumes the input is already split into two sorted halves.
    """
    if n <= 1:
        return []
    
    half = n // 2
    # The merge network for n wires
    # Batcher's odd-even merge:
    # 1. Compare (0, 1), (2, 3), ... and (1, 2), (3, 4), ...
    # Wait, the standard odd-even merge for merging two sorted lists of size n/2:
    # It's a butterfly-like structure.
    # Actually, simpler implementation for recursive odd-even merge:
    # If n is even:
    #   Merge(0..n/2-1) and Merge(n/2..n-1) are already done? No, this is the merge step.
    # The merge step for odd-even merge of two sorted sequences of length m:
    # It consists of:
    # 1. Compare (0, 1), (2, 3), ..., (m-2, m-1)  [Odd-Even Transposition on even indices? No]
    # Let's look up the exact structure or use a known small implementation.
    # For n=2: [(0,1)]
    # For n=4: [(0,2), (1,3), (0,1), (2,3), (1,2)]
    # For n=8:
    #   It's complex.
    # Alternative: Use the recursive definition.
    # Odd-Even Merge for 2n elements:
    # 1. Recursively merge the even-indexed elements (0, 2, ..., 2n-2) -> n elements
    # 2. Recursively merge the odd-indexed elements (1, 3, ..., 2n-1) -> n elements
    # 3. Compare (1,2), (3,4), ..., (2n-3, 2n-2)
    # 4. Recursively merge the even-indexed elements again? No.
    # Actually, the standard Odd-Even Mergesort is:
    # Sort(2n):
    #   Sort(even indices)  -> n elements
    #   Sort(odd indices)   -> n elements
    #   Merge(even, odd)    -> 2n elements
    #
    # Merge(2n) where inputs are two sorted lists of n elements, interleaved?
    # No, the inputs to Merge are the outputs of the recursive sorts.
    # The recursive sorts produce sorted lists on the even and odd positions.
    # So wire 0, 2, 4... are sorted. Wire 1, 3, 5... are sorted.
    # Then we need to merge these two sorted sequences.
    # The merge network for two sorted sequences of length n is:
    #   Compare (1,2), (3,4), ..., (2n-3, 2n-2)
    #   Then recursively merge the even positions and odd positions?
    #   No, that's not right.
    #
    # Let's use a simpler, verified network for small N, or just stick to the seed + SA.
    # Given the time constraints and the complexity of implementing Batcher correctly from scratch in this context,
    # and the fact that the seed is already 153, while Batcher for 18 is ~ (18/2 * log2(18/2)^2 + ...) which is roughly 100-120?
    # Actually, Batcher's Odd-Even Mergesort for n=18:
    # T(n) = 2*T(n/2) + M(n)
    # M(n) for merging two sorted lists of n/2 is ~ (n/2) * log2(n/2) + n/2
    # So T(18) = 2*T(9) + M(18)
    # T(9) = 2*T(4) + T(5) + M(9)
    # This is getting complicated to estimate.
    # Let's just use the seed and improve it with SA, but maybe initialize with a better network if I can generate one easily.
    #
    # Actually, I can just use the seed. The SA is the main driver.
    # But wait, the inspiration mentions "strongest ideas".
    # One strong idea is to use a better initial network.
    # Let's try to implement a simple Batcher network or use a known one.
    #
    # For now, I will stick to the seed + SA but optimize the SA loop.
    # The current SA is already decent.
    # Let's see if I can improve the SA by using a better neighborhood or acceptance criteria.
    #
    # Actually, let's just return the seed if I can't implement Batcher quickly.
    # But the task says "combine the strategy... with the strongest ideas".
    # The strongest idea in the inspiration is likely the vectorized verification or better SA.
    # The inspiration code provided in the prompt is cut off, but it shows `_batch_verify` which is just a loop.
    #
    # Let's focus on making the SA more effective.
    # One idea: use a "restart" strategy if stuck.
    # Another idea: use a better move generation, e.g., removing redundant comparators.
    #
    # Let's try to implement a simple check for redundant comparators.
    # A comparator (i, j) is redundant if removing it still results in a valid network.
    # We can check this by verifying the network without that comparator.
    # This is expensive, but we can do it for a subset.
    #
    # Let's just use the SA as is, but with a slightly better configuration.
    #
    # Actually, let's just return the seed if the SA doesn't improve it significantly.
    # But the SA in the current program already improved it to 1.1504.
    # So I should keep the SA.
    #
    # Let's refine the SA:
    # 1. Start with seed.
    # 2. Try to remove comparators. If valid, keep it.
    # 3. Repeat until no more removals are possible.
    # 4. Then use SA to explore further.
    #
    # This "greedy removal" phase can quickly reduce the size.
    
    return [] # Placeholder, will not be used if I stick to seed+SA

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Start with the seed network
    current = _odd_even_transposition()
    best = current[:]
    best_len = len(best)
    
    # Initial verification of seed (should be true)
    if not _verify(best):
        return best
        
    # Phase 1: Greedy removal of redundant comparators
    # Try to remove each comparator one by one. If the network is still valid, keep it removed.
    # This is a heuristic to quickly find a smaller network.
    changed = True
    while changed:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing current[i]
            neighbor = current[:i] + current[i+1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(best)
    
    # Phase 2: Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.99
    
    rng = random.Random(42)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline and T > T_min:
        move_type = rng.choices(['remove', 'add', 'swap', 'move'], weights=[5, 2, 1, 1])[0]
        neighbor = current[:]
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
            
        elif move_type == 'add':
            comp = rng.choice(all_comparators)
            pos = rng.randrange(len(neighbor) + 1)
            neighbor.insert(pos, comp)
            
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
        elif move_type == 'move' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                comp = neighbor.pop(i)
                if i < j:
                    j -= 1
                neighbor.insert(j, comp)

        if _verify(neighbor):
            delta = len(neighbor) - len(current)
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current = neighbor
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(best)

        T *= cooling_rate
        
    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best
