import numpy as np
import time
import random

N = 18
SIZE = 1 << N  # 262144
MAX_COMPARATORS = 4096

def _build_states():
    """Build initial states array (N, SIZE) where states[w][k] = bit w of k."""
    k_range = np.arange(SIZE, dtype=np.uint64)
    states = np.empty((N, SIZE), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)
    return states

def _is_sorted(states):
    """Check if all inputs produce sorted output (no 1 above 0 in any column)."""
    return not np.any(states[:-1] < states[1:])

def _fast_check(states, diff_buf):
    """Fast check if all inputs are sorted using pre-allocated buffer."""
    np.less(states[:-1], states[1:], out=diff_buf)
    return not np.any(diff_buf)

def _apply_comparator(states, i, j):
    """Apply comparator (i,j) in-place, return saved values."""
    si = states[i].copy()
    sj = states[j].copy()
    states[i] = si & sj
    states[j] = si | sj
    return si, sj

def _simulate_network(network, init_states):
    """Simulate network on all inputs, return final states."""
    s = init_states.copy()
    for i, j in network:
        si = s[i].copy()
        sj = s[j].copy()
        s[i] = si & sj
        s[j] = si | sj
    return s

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network for n wires."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _shell_sort_network(n):
    """Generate a Shell-sort-inspired network with multiple gap sequences."""
    network = []
    # Use several gap sequences for better coverage
    for gap in [n//2, n//4, n//8, n//16, 1]:
        if gap < 1:
            break
        for start in range(gap):
            for i in range(start, n - 1, gap):
                network.append((i, i + gap))
    # Add final adjacent passes to ensure correctness
    for _ in range(2):
        for i in range(0, n - 1, 2):
            network.append((i, i + 1))
        for i in range(1, n - 1, 2):
            network.append((i, i + 1))
    return network

def _batcher_odd_even_merge(n):
    """Generate Batcher's odd-even merge sort network."""
    def oe_merge(start, count):
        if count <= 1:
            return []
        comps = []
        half = count // 2
        # Compare elements at distance 'half'
        for i in range(start, start + half):
            comps.append((i, i + half))
        # Recursively merge even and odd subsequences
        # Even subsequence: start, start+2, start+4, ...
        # Odd subsequence: start+1, start+3, start+5, ...
        # This is complex for arbitrary n; use simpler approach
        # For correctness, fall back to bitonic merge structure
        left = oe_merge(start, half)
        right = oe_merge(start + half, count - half)
        return comps + left + right
    
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    mid = n // 2
    left = _batcher_odd_even_merge(mid)
    right = _batcher_odd_even_merge(n - mid)
    right = [(i + mid, j + mid) for i, j in right]
    merge = oe_merge(0, n)
    return left + right + merge

def _batcher_bitonic_merge(start, n):
    """Bitonic merge for n elements starting at wire 'start'."""
    if n <= 1:
        return []
    comparators = []
    half = n // 2
    for i in range(start, start + half):
        comparators.append((i, i + half))
    left = _batcher_bitonic_merge(start, half)
    right = _batcher_bitonic_merge(start + half, n - half)
    return comparators + left + right

def _batcher_bitonic_sort(n):
    """Generate Batcher bitonic sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    mid = n // 2
    left = _batcher_bitonic_sort(mid)
    right = _batcher_bitonic_sort(n - mid)
    right = [(i + mid, j + mid) for i, j in right]
    merge = _batcher_bitonic_merge(0, n)
    return left + right + merge

def _greedy_remove(network, init_states, diff_buf, deadline, max_rounds=50):
    """Aggressively try to remove comparators, starting from the end."""
    best_net = network[:]
    removed_count = 0
    
    for _ in range(max_rounds):
        if time.time() > deadline - 8:
            break
        if len(best_net) <= 1:
            break
        found_removal = False
        # Try from end to start (later comparators are more likely redundant)
        for idx in range(len(best_net) - 1, -1, -1):
            if time.time() > deadline - 8:
                break
            # Skip if this comparator might be critical (first half)
            if idx < len(best_net) // 4:
                continue
            trial = best_net[:idx] + best_net[idx+1:]
            states = _simulate_network(trial, init_states)
            if _fast_check(states, diff_buf):
                best_net = trial
                removed_count += 1
                found_removal = True
                break
        if not found_removal:
            # Try a few random positions
            for _ in range(10):
                if time.time() > deadline - 8:
                    break
                idx = random.randint(0, len(best_net) - 1)
                trial = best_net[:idx] + best_net[idx+1:]
                states = _simulate_network(trial, init_states)
                if _fast_check(states, diff_buf):
                    best_net = trial
                    removed_count += 1
                    found_removal = True
                    break
            if not found_removal:
                break
    
    return best_net, removed_count

def _try_swap_remove(network, init_states, diff_buf, deadline, max_attempts=200):
    """Try to swap and remove: replace two comparators with one, or move and remove."""
    best_net = network[:]
    
    for _ in range(max_attempts):
        if time.time() > deadline - 8:
            break
        if len(best_net) <= 2:
            break
        
        op = random.random()
        
        if op < 0.4:
            # Try removing one and adding a different one
            remove_idx = random.randint(0, len(best_net) - 1)
            i1, j1 = best_net[remove_idx]
            # Try inserting a random comparator elsewhere
            i2, j2 = random.sample(range(N), 2)
            if i2 > j2:
                i2, j2 = j2, i2
            insert_idx = random.randint(0, len(best_net))
            trial = best_net[:remove_idx] + best_net[remove_idx+1:]
            trial.insert(insert_idx, (i2, j2))
            if len(trial) < len(best_net):
                states = _simulate_network(trial, init_states)
                if _fast_check(states, diff_buf):
                    best_net = trial
                    break
        
        elif op < 0.7:
            # Try moving a comparator
            idx = random.randint(0, len(best_net) - 1)
            comp = best_net[idx]
            new_idx = random.randint(0, len(best_net) - 1)
            if idx == new_idx:
                continue
            trial = best_net[:]
            del trial[idx]
            trial.insert(new_idx, comp)
            states = _simulate_network(trial, init_states)
            if _fast_check(states, diff_buf):
                # Now try to remove something
                for ridx in range(len(trial) - 1, -1, -1):
                    if ridx == new_idx:
                        continue
                    trial2 = trial[:ridx] + trial[ridx+1:]
                    states2 = _simulate_network(trial2, init_states)
                    if _fast_check(states2, diff_buf):
                        best_net = trial2
                        break
        
        else:
            # Try removing two adjacent comparators
            if len(best_net) >= 3:
                idx = random.randint(0, len(best_net) - 2)
                trial = best_net[:idx] + best_net[idx+2:]
                states = _simulate_network(trial, init_states)
                if _fast_check(states, diff_buf):
                    best_net = trial
                    break
    
    return best_net

def _hill_climb(network, init_states, diff_buf, deadline, max_steps=500):
    """Local search: try single comparator modifications."""
    best_net = network[:]
    best_len = len(best_net)
    
    for step in range(max_steps):
        if time.time() > deadline - 8:
            break
        if len(best_net) <= 1:
            break
        
        improved = False
        # Try removing
        idx = random.randint(0, len(best_net) - 1)
        trial = best_net[:idx] + best_net[idx+1:]
        states = _simulate_network(trial, init_states)
        if _fast_check(states, diff_buf):
            best_net = trial
            best_len = len(trial)
            improved = True
        
        if not improved and len(best_net) > 1:
            # Try swapping two comparators
            i, j = random.sample(range(len(best_net)), 2)
            trial = best_net[:]
            trial[i], trial[j] = trial[j], trial[i]
            states = _simulate_network(trial, init_states)
            if _fast_check(states, diff_buf):
                best_net = trial
                improved = True
        
        if not improved:
            # Try changing a comparator's wires
            idx = random.randint(0, len(best_net) - 1)
            old_i, old_j = best_net[idx]
            new_i, new_j = random.sample(range(N), 2)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if (new_i, new_j) == (old_i, old_j):
                continue
            trial = best_net[:]
            trial[idx] = (new_i, new_j)
            states = _simulate_network(trial, init_states)
            if _fast_check(states, diff_buf):
                best_net = trial
                improved = True
        
        if not improved:
            # Stuck, try a bigger perturbation
            for _ in range(3):
                op = random.random()
                if op < 0.3 and len(best_net) > 2:
                    idx = random.randint(0, len(best_net) - 1)
                    best_net.pop(idx)
                    # Verify and restore if needed
                    states = _simulate_network(best_net, init_states)
                    if not _fast_check(states, diff_buf):
                        best_net.insert(idx, (old_i if 'old_i' in dir() else 0, old_j if 'old_j' in dir() else 1))
                elif op < 0.6 and len(best_net) > 1:
                    i, j = random.sample(range(len(best_net)), 2)
                    best_net[i], best_net[j] = best_net[j], best_net[i]
                    states = _simulate_network(best_net, init_states)
                    if not _fast_check(states, diff_buf):
                        best_net[i], best_net[j] = best_net[j], best_net[i]
                else:
                    idx = random.randint(0, len(best_net) - 1)
                    old_i, old_j = best_net[idx]
                    new_i, new_j = random.sample(range(N), 2)
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    best_net[idx] = (new_i, new_j)
                    states = _simulate_network(best_net, init_states)
                    if not _fast_check(states, diff_buf):
                        best_net[idx] = (old_i, old_j)
    
    return best_net

def _simulated_annealing(network, init_states, diff_buf, deadline):
    """Simulated annealing for network optimization."""
    current_net = network[:]
    best_net = network[:]
    best_len = len(network)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.98
    
    # Use a subset of inputs for fast energy calculation
    # Sample 8192 random inputs for speed
    sample_indices = np.random.choice(SIZE, 8192, replace=False)
    
    def energy(net):
        """Count unsorted positions over sampled inputs."""
        s = init_states[:, sample_indices].copy()
        for i, j in net:
            si = s[i].copy()
            sj = s[j].copy()
            s[i] = si & sj
            s[j] = si | sj
        # Count violations: positions where s[k][col] > s[k+1][col]
        violations = np.sum(s[:-1] > s[1:])
        return int(violations) + len(net) * 0.01  # Small penalty for size
    
    current_energy = energy(current_net)
    
    steps = 0
    max_steps = 3000
    
    while time.time() < deadline - 10 and steps < max_steps:
        steps += 1
        
        # Generate neighbor
        new_net = current_net[:]
        op = random.random()
        
        if op < 0.25 and len(new_net) > 1:
            idx = random.randint(0, len(new_net) - 1)
            new_net.pop(idx)
        elif op < 0.5:
            i, j = random.sample(range(len(new_net)), 2)
            new_net[i], new_net[j] = new_net[j], new_net[i]
        elif op < 0.75:
            idx = random.randint(0, len(new_net) - 1)
            new_i, new_j = random.sample(range(N), 2)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            new_net[idx] = (new_i, new_j)
        else:
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            idx = random.randint(0, len(new_net))
            new_net.insert(idx, (i, j))
        
        new_energy = energy(new_net)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
            current_net = new_net
            current_energy = new_energy
            
            if current_energy < energy(best_net) and len(new_net) < len(best_net):
                # Full verification before updating best
                states = _simulate_network(new_net, init_states)
                if _fast_check(states, diff_buf):
                    best_net = new_net[:]
                    best_len = len(new_net)
        
        T = max(T_min, T * cooling)
    
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Build initial states
    init_states = _build_states()
    diff_buf = np.empty((N - 1, SIZE), dtype=np.uint8)
    
    # Generate multiple candidate networks and pick the best starting point
    candidates = []
    
    # 1. Odd-even transposition (seed, 153 comparators)
    candidates.append(_odd_even_transposition(N))
    
    # 2. Shell sort inspired
    candidates.append(_shell_sort_network(N))
    
    # 3. Batcher odd-even merge
    try:
        candidates.append(_batcher_odd_even_merge(N))
    except Exception:
        pass
    
    # 4. Batcher bitonic sort
    try:
        candidates.append(_batcher_bitonic_sort(N))
    except Exception:
        pass
    
    # Verify each candidate and pick the shortest valid one
    best_net = None
    best_len = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        states = _simulate_network(cand, init_states)
        if _fast_check(states, diff_buf):
            if len(cand) < best_len:
                best_net = cand[:]
                best_len = len(cand)
    
    if best_net is None:
        # Fallback to seed
        best_net = _odd_even_transposition(N)
        best_len = len(best_net)
    
    # Phase 1: Aggressive greedy removal
    if time.time() < deadline - 30:
        best_net, _ = _greedy_remove(best_net, init_states, diff_buf, deadline, max_rounds=30)
        best_len = len(best_net)
    
    # Phase 2: Swap-remove optimization
    if time.time() < deadline - 25:
        improved_net = _try_swap_remove(best_net, init_states, diff_buf, deadline, max_attempts=100)
        if len(improved_net) < best_len:
            best_net = improved_net
            best_len = best_len
    
    # Phase 3: Hill climbing
    if time.time() < deadline - 20:
        improved_net = _hill_climb(best_net, init_states, diff_buf, deadline, max_steps=300)
        if len(improved_net) < best_len:
            best_net = improved_net
            best_len = len(best_net)
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 15:
        improved_net = _simulated_annealing(best_net, init_states, diff_buf, deadline)
        if len(improved_net) < best_len:
            best_net = improved_net
            best_len = len(best_net)
    
    # Phase 5: Final greedy cleanup
    if time.time() < deadline - 5:
        final_net, _ = _greedy_remove(best_net, init_states, diff_buf, deadline, max_rounds=20)
        if len(final_net) < best_len:
            best_net = final_net
            best_len = len(best_net)
    
    # Final verification
    states = _simulate_network(best_net, init_states)
    if not _fast_check(states, diff_buf):
        # Fallback to seed
        best_net = _odd_even_transposition(N)
    
    return best_net
