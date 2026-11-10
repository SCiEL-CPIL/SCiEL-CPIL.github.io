"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_max_discrepancy(signs):
    """Compute maximum absolute signed sum over all arithmetic progressions in [1, n]."""
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons have abs sum 1
    
    for step in range(1, n + 1):
        for r in range(step):
            if r >= n:
                break
            subseq = arr[r::step]
            if len(subseq) == 0:
                continue
            cumsums = np.cumsum(subseq)
            max_abs = np.max(np.abs(cumsums))
            if max_abs > max_disc:
                max_disc = max_abs
    return max_disc

def _build_progression_indices(n):
    """
    Pre-compute indices of all arithmetic progressions.
    Returns a list of arrays, where each array contains indices into 'signs' 
    that form a specific arithmetic progression.
    """
    progressions = []
    for step in range(1, n + 1):
        for r in range(step):
            if r >= n:
                continue
            # Generate indices: r, r+step, r+2*step, ... < n
            indices = np.arange(r, n, step, dtype=np.int32)
            if len(indices) > 0:
                progressions.append(indices)
    return progressions

def _compute_all_prefix_sums(signs, progressions):
    """
    Compute prefix sums for all progressions.
    Returns a list of arrays, each containing cumulative sums of the corresponding progression.
    """
    prefix_sums = []
    arr = np.array(signs, dtype=np.int64)
    for indices in progressions:
        subseq = arr[indices]
        cumsums = np.cumsum(subseq)
        prefix_sums.append(cumsums)
    return prefix_sums

def _compute_max_disc_from_prefix(prefix_sums):
    """Compute max discrepancy from pre-computed prefix sums."""
    max_disc = 1
    for ps in prefix_sums:
        if len(ps) > 0:
            max_abs = np.max(np.abs(ps))
            if max_abs > max_disc:
                max_disc = max_abs
    return max_disc

def _update_prefix_sums(prefix_sums, progressions, flip_idx, new_val, old_val):
    """
    Incrementally update prefix sums after flipping one sign.
    flip_idx: index in the main array (0-based)
    new_val: new sign value (+1 or -1)
    old_val: old sign value
    """
    delta = new_val - old_val
    if delta == 0:
        return
    
    # We need to find which progressions contain 'flip_idx' and at what position within them.
    # To do this efficiently, we pre-compute a mapping from index to list of (prog_id, pos_in_prog).
    # However, doing this on the fly is expensive. 
    # Alternative: Since n=256 is small, we can just check all progressions? 
    # No, that's O(P) per flip. P is roughly sum_{s=1}^{256} s = ~32k. 
    # 32k checks per flip is too slow for Python loops if we do many flips.
    
    # Better approach: Pre-compute the mapping.
    pass

def construct():
    n = 256
    deadline = time.time() + 50  # Safety margin
    
    # Pre-compute all arithmetic progression indices
    progressions = _build_progression_indices(n)
    num_progs = len(progressions)
    
    # Pre-compute mapping: for each index i in [0, n), which progressions contain it and at what position?
    # index_to_prog_pos[i] = list of (prog_id, pos_in_prog)
    index_to_prog_pos = [[] for _ in range(n)]
    for pid, indices in enumerate(progressions):
        for pos, idx in enumerate(indices):
            index_to_prog_pos[idx].append((pid, pos))
    
    # Simulated Annealing
    rng = random.Random(42)
    
    # Initial state: random signs
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    signs_arr = np.array(signs, dtype=np.int64)
    
    # Initial prefix sums
    prefix_sums = _compute_all_prefix_sums(signs_arr, progressions)
    current_disc = _compute_max_disc_from_prefix(prefix_sums)
    
    best_signs = list(signs)
    best_disc = current_disc
    
    # Annealing parameters
    T_start = 10.0
    T_end = 0.01
    # Number of iterations: we want to run as many as possible
    # Each iteration involves:
    # 1. Pick random index i
    # 2. Update prefix sums for all progressions containing i
    # 3. Compute new max disc (or update incrementally)
    # 4. Accept/Reject
    
    # To speed up max disc calculation, we can maintain a histogram of absolute prefix sums?
    # Or just recompute max over all progressions? 
    # Recomputing max over ~32k progressions, each up to length 256.
    # np.max on small arrays is fast, but 32k calls is slow in Python.
    # Let's try to vectorize the max calculation across all progressions?
    # They have different lengths. Padding is wasteful.
    
    # Alternative: Maintain a list of current max abs sum for each progression.
    # prog_max_abs[pid] = max(abs(prefix_sums[pid]))
    # When we flip, we update prefix_sums[pid] for affected pids.
    # Then we need to find the global max. 
    # If we store prog_max_abs, we can just take np.max(prog_max_abs).
    
    prog_max_abs = np.array([np.max(np.abs(ps)) if len(ps) > 0 else 0 for ps in prefix_sums], dtype=np.int64)
    current_disc = max(1, np.max(prog_max_abs))
    
    start_time = time.time()
    iterations = 0
    
    # Estimate time per iteration to set a target? 
    # Let's just run until deadline.
    
    while time.time() < deadline:
        T = T_start * (T_end / T_start) ** (iterations / 10000.0) # Decay over 10k iterations? 
        # Better: exponential decay based on time or fixed steps.
        # Let's use a fixed number of steps for temperature schedule, say 50,000 steps.
        # If we finish early, we can restart or continue.
        
        # Actually, let's just use a simple cooling schedule:
        # T = T_start * alpha^k
        # We don't know k ahead of time. 
        # Let's assume ~10,000 iterations per second? (optimistic)
        # Let's just use a time-based temperature or a fixed schedule that resets?
        
        # Simpler: Fixed schedule of 100,000 iterations.
        # If we run out of time, we stop.
        # If we finish 100k iterations, we restart with a new random state?
        
        # Let's try 20,000 iterations per "phase".
        
        # For now, let's just do a simple loop with a decaying T based on iterations.
        # We'll restart the annealing process if we hit a plateau or time runs out.
        
        i = rng.randint(0, n - 1)
        old_val = signs[i]
        new_val = -old_val
        delta = new_val - old_val
        
        # Update affected progressions
        affected = index_to_prog_pos[i]
        new_prog_max_abs = prog_max_abs.copy() # This is expensive: 32k copy per flip.
        # Better: update in place and track if global max changed.
        # But we need to know if the global max decreased.
        
        # Let's optimize the update:
        # 1. For each (pid, pos) in affected:
        #    - Update prefix_sums[pid][pos:] += delta
        #    - Recompute max_abs for this specific progression
        #    - Update prog_max_abs[pid]
        # 2. Compute new_global_max = max(1, np.max(prog_max_abs))
        
        # Is copying prog_max_abs (size ~32k) expensive? 
        # 32k * 8 bytes = 256KB. Copying 256KB per flip is ~microseconds? 
        # 10k flips/sec -> 2.5GB/s memory bandwidth. Feasible.
        
        # Let's try without copy first, just update and then take max.
        # But if we reject, we need to revert. 
        # Reverting prefix sums is also O(length).
        # It's easier to just compute the new state, and if rejected, revert.
        
        # Optimization: Only update prog_max_abs for affected pids.
        # Then find global max. 
        # If we keep a sorted structure or just scan the 32k array, it's O(P).
        # P ~ 32k. 32k operations per flip. 
        # If we do 1000 flips/sec, that's 32M ops/sec. Python is ~10-50M simple ops/sec. 
        # This might be too slow.
        
        # Let's check the size of P again.
        # n=256.
        # step=1: 1 prog (len 256)
        # step=2: 2 progs (len 128)
        # ...
        # step=128: 128 progs (len 2)
        # step=129: 129 progs (len 1) -> wait, range(step) is 0..128. 
        # For step > n/2, most progs have length 1.
        # Total progs = sum_{s=1}^{256} s = 256*257/2 = 32896.
        
        # Scanning 32896 elements in Python is ~0.3-0.5 ms?
        # 1000 flips/sec -> 300-500 ms per flip? No, that's 0.3ms per scan.
        # 1000 flips * 0.3ms = 300ms. So ~3000 flips/sec?
        # 50 seconds * 3000 flips = 150,000 flips.
        # This is reasonable.
        
        # Let's implement the flip logic.
        
        # Save old values for revert
        old_prefix_slices = []
        for pid, pos in affected:
            ps = prefix_sums[pid]
            # Save the slice we are about to modify? 
            # Or just re-calculate? 
            # Storing the old max is enough if we can re-calculate the max from the modified array?
            # No, we need to revert the array values.
            # Let's just store the old max for this pid, and if rejected, we need to revert the array.
            # Reverting the array means subtracting delta from ps[pos:].
            
            # Let's just do the update, store old max, and if rejected, undo update.
            old_max = prog_max_abs[pid]
            old_prefix_slices.append((pid, pos, old_max))
            
            # Update prefix sums
            ps[pos:] += delta
            
            # Recompute max for this progression
            # Since only suffix changed, and delta is +/- 2, the max might change.
            # We have to scan the whole progression to be safe? 
            # Or just scan the suffix? 
            # The prefix before 'pos' is unchanged.
            # So new_max = max(old_prefix_max, max(abs(ps[pos:])))
            # We don't have old_prefix_max stored separately.
            # Just scan the whole array. It's small (max 256).
            new_max = np.max(np.abs(ps))
            prog_max_abs[pid] = new_max
        
        new_disc = max(1, np.max(prog_max_abs))
        
        # Acceptance
        if new_disc <= current_disc:
            # Accept
            signs[i] = new_val
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = list(signs)
        else:
            # Reject with probability exp(-delta_disc / T)
            # Delta is positive (worse)
            delta_disc = new_disc - current_disc
            if rng.random() < np.exp(-delta_disc / T):
                signs[i] = new_val
                current_disc = new_disc
                # Update best? No, it's worse than current, but current might be better than best?
                # No, current_disc > best_disc always holds if we track best correctly.
                # Wait, if we accept a worse move, current_disc increases.
                # best_disc remains the minimum seen so far.
            else:
                # Revert
                for pid, pos, old_max in old_prefix_slices:
                    ps = prefix_sums[pid]
                    ps[pos:] -= delta
                    prog_max_abs[pid] = old_max
        
        iterations += 1
        # Simple temperature decay
        # Let's decay T every 1000 iterations
        if iterations % 1000 == 0:
             T *= 0.95
             if T < 0.01:
                 T = 0.01
        
    return best_signs
# EVOLVE-BLOCK-END
