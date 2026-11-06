N = 18

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_SEED = 42

def _verify_network(comps):
    """
    Verify a sorting network using the 0-1 principle.
    Tests all 2^18 = 262144 Boolean inputs.
    Returns True if the network sorts all inputs.
    """
    n = len(comps)
    if n == 0:
        return True
    # Precompute comparator indices
    ci = np.array([c[0] for c in comps], dtype=np.int32)
    cj = np.array([c[1] for c in comps], dtype=np.int32)
    
    for mask in range(1 << N):
        # Build input vector
        inp = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int32)
        for i in range(n):
            a, b = ci[i], cj[i]
            if inp[a] > inp[b]:
                inp[a], inp[b] = inp[b], inp[a]
        # Check sorted
        for i in range(N - 1):
            if inp[i] > inp[i + 1]:
                return False
    return True

def _verify_fast(comps, sample_size=None):
    """
    Fast verification using random sampling + full check only if needed.
    For the 0-1 principle, we need to check all 2^N inputs for N=18.
    But 2^18 = 262144 is feasible with numpy if we batch.
    """
    n = len(comps)
    if n == 0:
        return True
    ci = np.array([c[0] for c in comps], dtype=np.int32)
    cj = np.array([c[1] for c in comps], dtype=np.int32)
    
    # Batch process all 2^18 inputs
    total = 1 << N
    batch_size = 65536
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        # Generate all masks in this batch
        masks = np.arange(start, end, dtype=np.uint32)
        # Build input matrix: shape (batch, N)
        # For each mask, input[i] = (mask >> i) & 1
        shifts = np.arange(N, dtype=np.uint32)
        inp = ((masks[:, None] >> shifts[None, :]) & 1).astype(np.int32)
        
        # Apply comparators
        for i in range(n):
            a, b = ci[i], cj[i]
            # Where inp[a] > inp[b], swap
            swap_mask = inp[:, a] > inp[:, b]
            if np.any(swap_mask):
                inp[swap_mask, a] = inp[swap_mask, a] ^ inp[swap_mask, b] ^ inp[swap_mask, a]
                inp[swap_mask, b] = inp[swap_mask, b] ^ inp[swap_mask, a] ^ inp[swap_mask, b]
                # Simpler:
                tmp = inp[swap_mask, a].copy()
                inp[swap_mask, a] = inp[swap_mask, b]
                inp[swap_mask, b] = tmp
        
        # Check if sorted: inp[:, i] <= inp[:, i+1] for all i
        diffs = np.diff(inp, axis=1)
        if np.any(diffs < 0):
            return False
    return True

def _odd_even_transposition():
    """Generate the seed odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _sa_optimize(comps, deadline):
    """
    Simulated Annealing to minimize comparator count.
    Moves: remove, insert, swap adjacent, move element.
    """
    random.seed(_SEED)
    best = list(comps)
    best_count = len(best)
    current = list(comps)
    current_count = len(current)
    
    # Verify initial
    if not _verify_fast(current):
        raise ValueError("Initial network is not a valid sorting network")
    
    start_time = time.time()
    t = 10.0  # initial temperature
    cooling = 0.9995
    min_temp = 0.01
    max_comparators = 4096
    
    # Pre-verify cache
    verified_cache = {}
    
    def is_valid(net):
        key = tuple(net)
        if key in verified_cache:
            return verified_cache[key]
        result = _verify_fast(net)
        verified_cache[key] = result
        return result
    
    iteration = 0
    while t > min_temp:
        if time.time() > deadline:
            break
        iteration += 1
        
        # Choose a random move
        move_type = random.choice(['remove', 'insert', 'swap_adj', 'move_el', 'reorder_pair'])
        new_net = list(current)
        
        if move_type == 'remove' and len(new_net) > 1:
            idx = random.randint(0, len(new_net) - 1)
            new_net.pop(idx)
            
        elif move_type == 'insert':
            if len(new_net) >= max_comparators:
                new_net = list(current)
                continue
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(new_net))
            new_net.insert(pos, (i, j))
            
        elif move_type == 'swap_adj' and len(new_net) >= 2:
            idx = random.randint(0, len(new_net) - 2)
            new_net[idx], new_net[idx + 1] = new_net[idx + 1], new_net[idx]
            
        elif move_type == 'move_el' and len(new_net) >= 2:
            idx = random.randint(0, len(new_net) - 1)
            new_idx = random.randint(0, len(new_net) - 1)
            if idx != new_idx:
                elem = new_net.pop(idx)
                new_net.insert(new_idx, elem)
            else:
                new_net = list(current)
                
        elif move_type == 'reorder_pair':
            # Swap the endpoints of a random comparator
            if len(new_net) >= 1:
                idx = random.randint(0, len(new_net) - 1)
                a, b = new_net[idx]
                # Try swapping to (b, a) - but must maintain a < b
                # Actually, comparator (a,b) means min->a, max->b
                # We can't just swap; instead, pick a different random pair
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net[idx] = (i, j)
            else:
                new_net = list(current)
        else:
            new_net = list(current)
        
        new_count = len(new_net)
        
        # Check validity
        valid = is_valid(new_net)
        
        if valid:
            # Accept if better, or with probability exp(-delta/t) if worse
            delta = new_count - current_count  # positive means worse (more comps)
            if delta <= 0:
                current = new_net
                current_count = new_count
                if current_count < best_count:
                    best = list(current)
                    best_count = current_count
            else:
                # Accept worse with some probability
                if random.random() < np.exp(-delta / t):
                    current = new_net
                    current_count = new_count
        else:
            # Invalid: reject, but if it's a removal that made it invalid,
            # try to repair by re-adding
            if move_type == 'remove':
                # Re-add the removed element
                pass  # Just reject
        
        # Cool down
        t *= cooling
        
        # Periodic full verification of best
        if iteration % 500 == 0 and time.time() < deadline - 5:
            if not _verify_fast(best):
                # Should not happen, but safety
                best = _odd_even_transposition()
                best_count = len(best)
    
    return best

def _greedy_prune(comps, deadline):
    """
    Greedy removal: try removing each comparator, keep if still valid.
    Iterate until no more removals possible or deadline.
    """
    start_time = time.time()
    current = list(comps)
    
    while True:
        if time.time() > deadline - 10:
            break
        
        improved = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            trial = current[:idx] + current[idx + 1:]
            if _verify_fast(trial):
                current = trial
                improved = True
                break  # Restart scan after each removal
        
        if not improved:
            break
    
    return current

def construct():
    """
    Construct a sorting network for 18 wires using simulated annealing
    starting from the odd-even transposition network.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin for 90s limit
    
    # Step 1: Generate seed network
    comps = _odd_even_transposition()
    
    # Step 2: Verify seed
    if not _verify_fast(comps):
        # Should not happen
        return comps
    
    # Step 3: Greedy prune first (fast, deterministic improvement)
    prune_deadline = min(start_time + 20.0, deadline - 40.0)
    if prune_deadline > start_time:
        comps = _greedy_prune(comps, prune_deadline)
    
    # Step 4: Simulated Annealing for further optimization
    if time.time() < deadline - 5:
        comps = _sa_optimize(comps, deadline)
    
    # Final verification
    if not _verify_fast(comps):
        # Fallback to known valid network
        comps = _odd_even_transposition()
    
    return comps
# EVOLVE-BLOCK-END
