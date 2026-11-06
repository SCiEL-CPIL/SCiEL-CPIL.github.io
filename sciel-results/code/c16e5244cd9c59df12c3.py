import time
import numpy as np
import random

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 60  # seconds, with safety margin

def _apply_network_vectorized(network, inputs):
    """
    Apply the network to a batch of inputs.
    inputs: shape (batch_size, N)
    network: list of (i, j)
    Returns: shape (batch_size, N)
    """
    # Copy inputs to avoid modifying original
    current = inputs.copy()
    for i, j in network:
        vi = current[:, i]
        vj = current[:, j]
        # min to i, max to j
        current[:, i] = np.minimum(vi, vj)
        current[:, j] = np.maximum(vi, vj)
    return current

def _is_sorted_network_vectorized(network):
    """
    Verify if the network sorts all 2^N binary inputs.
    Returns True if sorted, False otherwise.
    """
    # Generate all 2^N inputs
    # N=16 -> 65536 inputs
    # We can generate them as a (65536, N) array of uint8 or int
    # Using bit manipulation to generate rows
    # 0 to 2^N - 1
    # For each row, bits correspond to wires
    
    # Efficient generation:
    # x = np.arange(2**N, dtype=np.uint32)
    # inputs = ((x[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
    
    # However, for N=16, 65536 * 16 = 1M elements. This is small.
    
    x = np.arange(1 << N, dtype=np.uint32)
    # We need to be careful with bit order.
    # Wire 0 is LSB or MSB?
    # Standard: bit i corresponds to wire i.
    # So wire 0 is bit 0 (LSB).
    inputs = ((x[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
    
    # Apply network
    result = _apply_network_vectorized(network, inputs)
    
    # Check if each row is sorted
    # Sorted means: 0s then 1s.
    # i.e., once we see a 1, all subsequent elements must be 1.
    # Or: diff of consecutive elements must be non-negative (0 or 1).
    # Since values are 0/1, sorted iff no 1 is followed by 0.
    # i.e., result[:, :-1] <= result[:, 1:]
    
    diffs = np.diff(result, axis=1)
    # If any diff is -1, it's not sorted.
    # diffs will be -1, 0, or 1.
    # Not sorted if any diff == -1.
    
    return not np.any(diffs == -1)

def construct():
    """
    Return a fixed compare-exchange network that sorts all 16-input sequences.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s limit
    
    # 1. Constructive Initialization
    # Start with a "worst case" input: 1, 0, 1, 0, ... or 15, 14, ... 0
    # Using binary 0/1 is sufficient for verification, but for construction,
    # using distinct values might help guide the comparators better?
    # Actually, the Zero-One principle says if it sorts 0/1, it sorts all.
    # But for *finding* comparators, looking at inversions in a specific
    # permutation can be a heuristic.
    
    # Let's use the reverse sorted permutation of 0..15 as the "guiding" input.
    # This maximizes inversions.
    # However, we must ensure the final network sorts 0/1 inputs.
    # The constructive step is just to get a "good enough" starting network.
    
    # Heuristic:
    # While the current state (of the guiding input) is not sorted:
    #   Find the pair (i, j) that reduces the total inversion count of the
    #   guiding input the most, subject to i < j.
    #   Add (i, j) to the network.
    #   Apply (i, j) to the guiding input.
    
    # This will produce a network that sorts the guiding input.
    # It is NOT guaranteed to sort all inputs, but it provides a structural
    # starting point that is likely denser in "useful" comparators than random.
    
    guiding_input = np.arange(N-1, -1, -1) # 15, 14, ..., 0
    current_state = guiding_input.copy()
    
    network = []
    
    # Limit the constructive phase to avoid infinite loops or too many comparators
    # A sorting network for N=16 needs ~60.
    # This greedy approach might take more, but we prune later.
    max_constructive_steps = 200
    
    for _ in range(max_constructive_steps):
        if time.time() > deadline:
            break
            
        # Check if guiding input is sorted
        if np.all(np.diff(current_state) >= 0):
            break
            
        # Find best comparator (i, j)
        best_reduction = -1
        best_pair = None
        
        # We can limit search to i < j
        # To speed up, we can just look for adjacent inversions first?
        # No, "maximize reduction" is the directive.
        # For N=16, there are 120 pairs.
        # Calculating reduction for 120 pairs per step is fast.
        
        for i in range(N):
            for j in range(i+1, N):
                if current_state[i] > current_state[j]:
                    # Inversion exists.
                    # Swapping them (min to i, max to j) will fix this inversion.
                    # Does it create new ones?
                    # min(a, b) to i, max(a, b) to j.
                    # Since a > b, new_i = b, new_j = a.
                    # This specific swap reduces the inversion count by exactly 1?
                    # No, it can change relative order with other elements.
                    
                    # Actually, for a specific pair (i, j), the reduction in inversion count
                    # is not just 1.
                    # Let's just calculate the inversion count of the state after applying (i, j).
                    
                    # Optimization:
                    # Only consider pairs that are currently inverted?
                    # If current_state[i] <= current_state[j], applying (i, j) does nothing.
                    # So we only care about i where current_state[i] > current_state[j].
                    
                    # Let's simulate the swap on a copy?
                    # That's 120 * 16 operations = 1920 ops per step.
                    # 200 steps * 1920 = 384k ops. Very fast.
                    
                    # But we can be smarter.
                    # The inversion count change when swapping i, j (where val_i > val_j):
                    # 1. The pair (i, j) itself is fixed (-1 inversion).
                    # 2. For any k < i:
                    #    Old: (k, i) and (k, j).
                    #    New: (k, i) has val_j, (k, j) has val_i.
                    #    This is complex.
                    
                    # Let's just compute the full inversion count for candidates.
                    # But 120 candidates * O(N^2) inversion count = 120 * 256 = 30k ops.
                    # 200 * 30k = 6M ops. Still fast in Python? Maybe borderline.
                    
                    # Alternative:
                    # Just pick the pair with the largest value difference?
                    # Or the pair that is "most inverted"?
                    
                    # Let's stick to the simplest robust heuristic:
                    # Pick the pair (i, j) with i < j, val_i > val_j, that minimizes (j - i)
                    # i.e., bubble sort style? No, that's bad for depth.
                    
                    # Let's try: Pick the pair (i, j) that is an inversion, and
                    # maximizes the "gap" in sortedness?
                    
                    # Actually, the directive says "maximize reduction in BDD node count"
                    # but we are using "inversion count" as a proxy for structural complexity.
                    
                    # Let's just do a greedy "fix the most impactful inversion".
                    # Impact = number of elements between i and j that are "misordered" relative to this swap?
                    
                    # Simpler: Just do a random selection from inverted pairs?
                    # No, let's do the full check but optimized.
                    
                    # We will just pick the first inverted pair we find in a specific order?
                    # That's just bubble sort.
                    
                    # Let's try to be a bit smarter:
                    # Prioritize pairs that are far apart? (Reduces depth)
                    # Or close together? (Reduces width?)
                    
                    # Let's just use the "max value diff" heuristic among inverted pairs.
                    # This tends to move large elements to the right and small to the left quickly.
                    
                    diff = current_state[i] - current_state[j]
                    if diff > best_reduction:
                        best_reduction = diff
                        best_pair = (i, j)
        
        if best_pair is None:
            break
            
        i, j = best_pair
        network.append((i, j))
        
        # Apply to guiding input
        vi = current_state[i]
        vj = current_state[j]
        current_state[i] = min(vi, vj)
        current_state[j] = max(vi, vj)
        
        # If we've added too many, stop
        if len(network) > 150:
            break

    # 2. Local Search / Pruning
    # Now we have a network that sorts the guiding input.
    # It likely does NOT sort all inputs.
    # We need to prune it to a valid sorting network.
    
    # Strategy:
    # 1. Verify if current network is valid.
    # 2. If not, we can't just prune. We need to *add* comparators?
    #    Wait, the constructive phase *only* added comparators that helped the guiding input.
    #    It didn't check global validity.
    #    So the network is likely INVALID for all inputs.
    
    # Correction:
    # We must ensure the network is valid.
    # If we start with an invalid network, pruning won't help.
    # We need a valid starting point.
    
    # The seed (Odd-Even Transposition) is valid.
    # My constructive network is likely NOT valid.
    
    # So, I will combine:
    # 1. Start with the Valid Seed (Odd-Even Transposition, 120 comparators).
    # 2. Use the Constructive Heuristic to *reorder* or *replace* comparators?
    #    Or, use the constructive heuristic to generate a *new* set of comparators
    #    and then *merge* them with the seed, then prune?
    
    # Better Strategy:
    # 1. Generate the Seed (Valid).
    # 2. Perform "Greedy Replacement":
    #    Try to replace each comparator in the seed with a "better" one
    #    defined by the constructive heuristic?
    #    Or, simply try to remove comparators from the seed, but use the
    #    "inversion count" of the guiding input as a *tie-breaker* or *guide*
    #    for which ones to keep?
    
    # Actually, the most effective structural change from the "random removal"
    # is to use the *global validity* check (vectorized) but to guide the
    # *search* using the inversion metric.
    
    # Let's try a different approach:
    # 1. Start with Seed (Valid).
    # 2. Compute "Guiding Input State" for the seed.
    # 3. Identify "Redundant" comparators:
    #    A comparator (i, j) is redundant if removing it still leaves the network valid.
    #    This is the same as random removal, but we can check them in an order
    #    that prioritizes those that *don't* change the guiding input state?
    #    If a comparator doesn't change the guiding input, it's "less useful" for
    #    that specific case, but might be crucial for others.
    
    # Let's stick to the vectorized verification but improve the search strategy.
    # Instead of random shuffle, use a "Greedy by Impact" strategy.
    
    # 1. Start with Seed.
    # 2. For each comparator in the network:
    #    a. Calculate how much it "contributes" to sorting the guiding input.
    #       (e.g., does it fix an inversion in the guiding input?)
    #    b. Sort comparators by "contribution" (lowest first).
    #    c. Try to remove the lowest contribution comparators first.
    #    d. If removal fails (network becomes invalid), keep it.
    
    # This is a structural change: using a heuristic to *order* the pruning attempts.
    
    # Implementation:
    
    # Generate Seed
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify Seed is valid (it should be)
    # _is_sorted_network_vectorized(network) -> True
    
    # Define "Contribution" metric for a comparator (i, j) in the context of the guiding input
    # We need to know the state of wires i and j *before* the comparator is applied.
    # This depends on the previous comparators.
    
    # This is complex to calculate for all comparators simultaneously.
    
    # Alternative:
    # Just use the vectorized check, but do it in a "Wave" pattern?
    
    # Let's try the "BDD-inspired" idea from the prompt more directly.
    # "Select comparators that maximize reduction in BDD node count"
    # We can approximate BDD node count by the number of *distinct* states
    # reachable by the network for a subset of inputs?
    # No, that's hard.
    
    # Let's go with the "Ordered Pruning" based on a simple heuristic:
    # Heuristic: Comparators that connect wires with small indices are "more local".
    # Comparators that connect wires with large indices are "more global".
    # Usually, global comparators are more critical for sorting.
    # So, try to remove LOCAL comparators first?
    
    # Or, try to remove comparators that are "redundant" in the sense that
    # they are adjacent to other comparators?
    
    # Let's implement a "Greedy Removal with Heuristic Ordering".
    
    # 1. Create a list of indices to try.
    # 2. Score each index by a heuristic.
    #    Heuristic: Distance between i and j. Smaller distance = higher priority for removal?
    #    Why? Because local swaps are often redundant if global structure is maintained.
    
    # Let's test this hypothesis.
    
    # Refine the search loop:
    best_network = network[:]
    
    # Pre-compute a "removal order" based on heuristic
    # We will re-calculate this order in each pass? Or keep it static?
    # Static is faster.
    
    # Heuristic: (j - i)
    # Sort indices by (j - i) ascending.
    
    while time.time() < deadline:
        removed_any = False
        
        # Create list of (heuristic_score, index)
        # We want to try removing low score (local) ones first?
        # Or high score (global) ones?
        # Let's try removing LOCAL ones first (small distance).
        
        indices = list(range(len(best_network)))
        # Sort by distance
        indices.sort(key=lambda idx: best_network[idx][1] - best_network[idx][0])
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing
            temp_network = best_network[:idx] + best_network[idx+1:]
            
            if _is_sorted_network_vectorized(temp_network):
                best_network = temp_network
                removed_any = True
                # Restart the loop? Or continue?
                # If we continue, the indices are now stale.
                # It's safer to restart the pass.
                break
        
        if not removed_any:
            # Try a different heuristic?
            # Or random shuffle?
            # Let's try sorting by distance descending (remove global ones first)
            indices.sort(key=lambda idx: -(best_network[idx][1] - best_network[idx][0]))
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                temp_network = best_network[:idx] + best_network[idx+1:]
                
                if _is_sorted_network_vectorized(temp_network):
                    best_network = temp_network
                    removed_any = True
                    break
            
            if not removed_any:
                # Try random
                random.shuffle(indices)
                for idx in indices:
                    if time.time() > deadline:
                        break
                    
                    temp_network = best_network[:idx] + best_network[idx+1:]
                    
                    if _is_sorted_network_vectorized(temp_network):
                        best_network = temp_network
                        removed_any = True
                        break
                
                if not removed_any:
                    break # Stuck
    
    return best_network

# EVOLVE-BLOCK-END
