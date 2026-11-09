import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin under 90s kill


def _build_seed_network():
    """Odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _verify(comps):
    """
    Verify sorting network using zero-one principle with bitmask simulation.
    Returns True if network sorts all 2^N boolean inputs.
    Uses vectorized NumPy operations for speed.
    """
    total = 1 << N
    masks = np.arange(total, dtype=np.uint32)
    
    for (i, j) in comps:
        bi = np.uint32(1 << i)
        bj = np.uint32(1 << j)
        # Swap condition: bit i is 1 AND bit j is 0
        # min goes to i, max goes to j.
        # If i has 1 and j has 0, we need to swap so i gets 0 and j gets 1.
        # Wait, let's re-verify the logic.
        # Input: i=1, j=0. min(1,0)=0 -> i, max(1,0)=1 -> j. Result: i=0, j=1.
        # So if (mask & (1<<i)) and not (mask & (1<<j)): swap.
        # mask ^= (1<<i) | (1<<j)
        
        cond = (masks & bi) & ~(masks & bj)
        if np.any(cond):
            swap_mask = np.uint32((1 << i) | (1 << j))
            masks = np.where(cond, masks ^ swap_mask, masks)
            
    # Check if all masks are sorted.
    # Sorted means all 1s are in lower positions.
    # mask = 0b...0111. mask + 1 = 0b...1000. mask & (mask+1) = 0.
    # This works for 0 (0 & 1 = 0) and 1 (1 & 2 = 0) and 3 (3 & 4 = 0) etc.
    return bool(np.all((masks & (masks + 1)) == 0))


def _depth_of_network(comps):
    """Compute the parallel depth (longest path) of the comparator network."""
    if not comps:
        return 0
    depth = [0] * N
    for (i, j) in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)


def _greedy_remove(comps, deadline):
    """
    Repeatedly try to remove comparators from the end of the list backwards.
    This is a fast way to clean up redundant comparators after mutations.
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        # Try removing from the end first, as it's often cheaper to verify
        # if we assume earlier comparators are more critical?
        # Actually, order doesn't matter for correctness, but for efficiency of 
        # finding removable ones, we can just scan.
        # Let's scan from end to start to potentially find "newer" redundant ones first.
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            new_comps = comps[:idx] + comps[idx + 1:]
            if _verify(new_comps):
                comps = new_comps
                improved = True
                break
    return comps


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    comps = _build_seed_network()
    if not _verify(comps):
        return comps

    best_comps = list(comps)
    best_len = len(comps)

    # Phase 1: Aggressive Greedy Reduction
    # Get a good baseline quickly
    comps = _greedy_remove(comps, deadline - 10)
    if len(comps) < best_len:
        best_len = len(comps)
        best_comps = list(comps)

    # Phase 2: Simulated Annealing
    # We use SA to escape local minima.
    # State: list of comparators
    # Energy: len(comps)
    # Moves:
    #   1. Remove a random comparator. If valid, accept (energy decreases).
    #   2. Insert a random comparator at random position. If valid, accept (energy increases).
    #      Acceptance probability: exp(-delta_E / T)
    #   3. Swap two comparators. If valid, accept.
    
    rng = np.random.default_rng(42)
    
    current_comps = list(best_comps)
    current_len = len(current_comps)
    
    # Temperature schedule
    # Start high to allow many insertions, cool down to favor removals
    T_start = 2.0
    T_end = 0.01
    steps = 2000  # Max steps, but bounded by time
    
    # We will interleave SA moves with occasional greedy cleanup
    last_cleanup_time = start_time
    
    for step in range(steps):
        if time.time() > deadline - 5:
            break
        
        # Periodically clean up with greedy removal to lock in gains
        if time.time() - last_cleanup_time > 1.0:
            cleaned = _greedy_remove(current_comps, deadline - 5)
            if len(cleaned) < current_len:
                current_comps = cleaned
                current_len = len(cleaned)
                if current_len < best_len:
                    best_len = current_len
                    best_comps = list(current_comps)
            last_cleanup_time = time.time()
            
            # Update temperature based on remaining time or step count
            # Let's use a simple exponential cooling based on step count
            progress = step / steps
            T = T_start * (T_end / T_start) ** progress
            
            # If we've been stuck for a while and T is low, maybe restart T?
            # For now, simple schedule.
            
        else:
            # Determine current temperature
            progress = step / steps
            T = T_start * (T_end / T_start) ** progress
            
        move_type = rng.random()
        
        if move_type < 0.6:
            # Try Removal
            if len(current_comps) <= 1:
                continue
            idx = rng.randint(0, len(current_comps))
            cand = current_comps[:idx] + current_comps[idx+1:]
            if _verify(cand):
                current_comps = cand
                current_len = len(cand)
                if current_len < best_len:
                    best_len = current_len
                    best_comps = list(current_comps)
        elif move_type < 0.9:
            # Try Insertion
            # Generate random comparator (i, j) with i < j
            i = rng.randint(0, N - 1)
            j = rng.randint(i + 1, N)
            pos = rng.randint(0, len(current_comps) + 1)
            cand = current_comps[:pos] + [(i, j)] + current_comps[pos:]
            
            # Check validity
            if _verify(cand):
                # Energy increases by 1
                delta_E = 1
                # Accept with probability exp(-delta_E / T)
                if rng.random() < np.exp(-delta_E / T):
                    current_comps = cand
                    current_len = len(cand)
        else:
            # Try Swap
            if len(current_comps) < 2:
                continue
            i = rng.randint(0, len(current_comps))
            j = rng.randint(0, len(current_comps))
            if i == j:
                continue
            cand = list(current_comps)
            cand[i], cand[j] = cand[j], cand[i]
            if _verify(cand):
                current_comps = cand
                # Energy unchanged
                # If we found a swap that allows further reduction later, we just accept
                # We don't update best here because len is same, but structure changed
                # However, if this swap enables a removal, the next removal step will catch it.

    # Final cleanup
    final_comps = _greedy_remove(current_comps, deadline)
    if len(final_comps) < best_len:
        best_comps = final_comps

    # Ensure we return the best found
    return best_comps
