import time
import numpy as np

N = 18
DEADLINE = 80.0  # Safety margin before 90s kill

def _bitonic_network(n):
    """Generate a bitonic sorting network for n wires."""
    comps = []
    def bitonic_sort(lo, hi, count):
        if count <= 1:
            return
        half = count // 2
        # Sort two halves
        bitonic_sort(lo, lo + half, half)
        bitonic_sort(lo + half, lo + count, half)
        # Merge
        k = half
        while k > 0:
            for i in range(lo, lo + half):
                comps.append((i, i + k))
            k //= 2

    # Bitonic sort requires n to be a power of 2 for standard recursive implementation.
    # For n=18, we can pad to 32 or use a generalized bitonic merge.
    # However, standard bitonic is for powers of 2. 
    # Let's use a simpler, known good network: Bubble sort is 153.
    # A better seed: Odd-Even Transposition is N^2/2 = 162? No, N(N-1)/2 is bubble.
    # Odd-even transposition for N wires takes N phases, each with N/2 comparators. Total N^2/2.
    # For N=18, 18*18/2 = 162.
    # Bubble sort is 153.
    # Let's start with Bubble sort (153) as it's smaller than OET (162).
    # Or better, use a known compact network if available? 
    # Let's stick to generating a valid network first, then optimize.
    # Actually, let's use a "brick wall" or similar?
    # Let's just use Bubble Sort as the seed, it's valid and 153.
    # We can also try to construct a network by sorting indices.
    
    # Wait, bitonic sort for non-power of 2 is tricky. 
    # Let's use a simple valid network: 
    # We can generate a network that sorts by simulating bubble sort.
    # This is the current seed. 
    # To make it different, let's try to start with a *different* valid network.
    # How about a network based on sorting a permutation of wires?
    # No, comparators must be (i, j) with i < j.
    
    # Let's use the "Odd-Even Merge" logic?
    # For now, let's just use Bubble Sort as the seed, but the *strategy* is the optimization.
    # The prompt asks for a strategy materially different from the current program's *strategy*.
    # Current program: "Bubble sort network".
    # My strategy: "Heuristic local search starting from a bitonic-like or other seed".
    # I will use a seed that is slightly different, e.g., a network that sorts by repeatedly applying comparators in a specific pattern that is not just bubble.
    # Actually, let's just use Bubble Sort as the seed because it's easy to generate and valid.
    # The *difference* is the optimization loop.
    
    for i in range(n - 1):
        for j in range(n - 1 - i):
            comps.append((j, j + 1))
    return comps

def _simulate(comps, x):
    """Simulate network on boolean input x (array of 0s and 1s)."""
    state = x.copy()
    for i, j in comps:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _is_valid(comps):
    """Check if network sorts all 2^N boolean inputs using 0-1 principle."""
    # There are 2^18 = 262144 inputs.
    # We can vectorize this with numpy.
    # Create a matrix of all possible boolean inputs.
    # 2^18 rows, 18 columns.
    # This might be memory heavy: 262144 * 18 * 8 bytes = ~37 MB. Acceptable.
    
    # Generate all binary vectors
    # Using np.arange and bit manipulation
    vecs = np.arange(1 << N, dtype=np.uint8)
    # Reshape to (2^N, 1)
    vecs = vecs.reshape(-1, 1)
    # Create bit mask
    bits = np.arange(N, dtype=np.uint8)
    # x = (vecs >> bits) & 1
    # vecs is (2^N, 1), bits is (N,). Broadcasting: (2^N, 1) >> (N,) -> (2^N, N)
    x = (vecs >> bits) & 1
    
    # Simulate network
    # We need to apply comparators to x.
    # x is (2^N, N)
    for i, j in comps:
        # Compare column i and j
        # If x[:, i] > x[:, j], swap
        # Since boolean, swap if x[:, i] == 1 and x[:, j] == 0
        # min goes to i, max to j.
        # So new_i = min(x_i, x_j), new_j = max(x_i, x_j)
        # min(a, b) for bools is a & b? No. min(0,1)=0, min(1,0)=0, min(0,0)=0, min(1,1)=1. So a & b.
        # max(a, b) for bools is a | b.
        # Wait, if a=1, b=0: min=0, max=1.
        # If a=0, b=1: min=0, max=1.
        # If a=1, b=1: min=1, max=1.
        # If a=0, b=0: min=0, max=0.
        # So yes, min is AND, max is OR for booleans.
        
        # However, we only need to swap if out of order.
        # If we just assign x[:, i] = x[:, i] & x[:, j] and x[:, j] = x[:, i] | x[:, j],
        # we must be careful with order of operations if we overwrite i before using it for j.
        # So we need temporary variables or simultaneous assignment.
        
        # Simultaneous assignment in numpy:
        # x[:, i], x[:, j] = x[:, i] & x[:, j], x[:, i] | x[:, j]
        # This works in Python tuple assignment, but does numpy handle it atomically?
        # Yes, right-hand side is evaluated first.
        
        x[:, i], x[:, j] = x[:, i] & x[:, j], x[:, i] | x[:, j]
        
    # Check if sorted: x[:, k] <= x[:, k+1] for all k
    # i.e., x[:, :-1] <= x[:, 1:]
    # For bools, this means NOT (x[:, :-1] == 1 and x[:, 1:] == 0)
    # Or simply: (x[:, :-1] & ~x[:, 1:]).any() should be False.
    
    if N == 1:
        return True
        
    # Check for any inversion
    inversions = (x[:, :-1] & ~x[:, 1:])
    if inversions.any():
        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Seed: Bubble Sort Network
    # This is 153 comparators.
    comps = []
    for i in range(N - 1):
        for j in range(N - 1 - i):
            comps.append((j, j + 1))
    
    # Local Search
    # Moves:
    # 1. Remove a comparator.
    # 2. Swap two comparators? (Order matters).
    # 3. Change a comparator (i, j) to (i', j')?
    
    # Let's try removal first. It's the most direct way to reduce size.
    # We can also try "shifting" comparators.
    
    # To speed up validity check, we can cache the state?
    # No, just re-run simulation. 262k inputs * 153 comparators is fast in numpy.
    # 262k * 153 operations is ~40M ops. NumPy does this in milliseconds.
    
    # Strategy:
    # Iterate through all possible removals? 
    # Or random removals?
    # Random removals are faster to implement and less likely to get stuck in a specific order.
    
    # Let's do a hill-climbing with random restarts or just random moves.
    # Since we have time budget, we can try many moves.
    
    # Pre-compute the full simulation matrix?
    # No, we modify comps, so we must re-simulate.
    
    # Optimization:
    # We can simulate the network on all 2^18 inputs at once.
    # Let's verify the cost.
    # N=18, 2^18 = 262144.
    # Matrix size: 262144 x 18.
    # Each comparator step: 2 column accesses, 1 AND, 1 OR, 2 column assignments.
    # This is very fast in NumPy.
    
    # Let's try to remove comparators one by one.
    # We will shuffle the list of comparators and try to remove them.
    # If removal keeps it valid, we remove it.
    # We repeat this process until no more can be removed or time runs out.
    
    # To avoid getting stuck, we can also try to *move* comparators.
    # But removal is the primary goal for score.
    
    # Let's implement a more robust search:
    # 1. Try to remove any comparator.
    # 2. If none can be removed, try to *swap* adjacent comparators? (Doesn't change count).
    # 3. Try to *replace* a comparator with a different one?
    
    # For now, let's just do random removal attempts.
    
    # To make it more effective, we can try removing comparators in random order.
    
    import random
    
    best_comps = comps
    best_len = len(comps)
    
    # We will iterate through random indices and try to remove.
    # If successful, we remove it and restart the scan or continue.
    
    # A better approach: 
    # While time < deadline:
    #   Pick a random comparator index k.
    #   Try to remove comps[k].
    #   If valid, remove it.
    #   Else, maybe try to swap it with another?
    
    # Let's add a "swap" move to escape local minima where removal isn't possible.
    # Swapping changes the order, which might allow subsequent removals.
    
    while time.time() < DEADLINE:
        # Randomly pick a move
        move_type = random.random()
        
        if len(comps) == 0:
            break
            
        if move_type < 0.7:
            # Try removal
            k = random.randint(0, len(comps) - 1)
            # Create candidate list without k
            candidate = comps[:k] + comps[k+1:]
            
            # Check validity
            if _is_valid(candidate):
                comps = candidate
                # If we removed one, we might want to restart the randomization?
                # Or just continue.
        else:
            # Try swap
            if len(comps) < 2:
                continue
            k = random.randint(0, len(comps) - 2)
            # Swap comps[k] and comps[k+1]
            candidate = comps.copy()
            candidate[k], candidate[k+1] = candidate[k+1], candidate[k]
            
            # Check validity
            if _is_valid(candidate):
                comps = candidate
                
    return comps
# EVOLVE-BLOCK-END
