"""Sorting network search for N=18 wires using Batcher merge + removal + local search."""
import time
import numpy as np
import random

N = 18
DEADLINE = 82.0  # safety margin before 90s kill


def _batcher_odd_even_merge(n, offset=0):
    """Generate Batcher's odd-even merge network for n wires starting at offset."""
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    # Odd-even merge: recursively merge odd and even positions, then compare-merge
    # For n wires, split into two halves of size n//2
    half = n // 2
    comps = []
    # Recursively merge the two halves (shifted by offset)
    # The standard odd-even merge:
    # 1. Merge odd-indexed and even-indexed subsequences recursively
    # 2. Then do the merge comparisons
    
    # Actually, let's use the standard Batcher odd-even merge:
    # For n elements at positions 0..n-1 (relative):
    # If n == 1: done
    # If n == 2: compare (0,1)
    # Otherwise:
    #   - Recursively merge odd positions (1,3,5,...) and even positions (0,2,4,...)
    #   - Then for i in 0..n/2-1: compare (2i, 2i+1)
    #   - Then for i in 1..n/2-1: compare (2i-1, 2i+1)
    
    # Wait, let me use a cleaner formulation.
    # Batcher's odd-even merge for n wires:
    #   if n == 1: return []
    #   if n == 2: return [(0,1)]
    #   Let m = n//2
    #   Step 1: Recursively merge wires at even positions (0,2,4,...) -> m wires
    #           Recursively merge wires at odd positions (1,3,5,...) -> n-m wires
    #   Step 2: For i in range(m): compare wire (2i) with wire (2i+1)  [if 2i+1 < n]
    #   Step 3: For i in range(1, m): compare wire (2i-1) with wire (2i+1) [if 2i+1 < n]
    
    # Hmm, this is getting confusing. Let me just use a well-known implementation.
    
    # Standard odd-even merge network:
    # merge(n):
    #   if n == 1: return []
    #   if n == 2: return [(0,1)]
    #   m = n // 2
    #   # Recursively merge the first m and last n-m elements? No...
    #   # Odd-even merge: split into odd and even indexed elements
    #   # Recursively merge odds and evens
    #   # Then interleave with comparisons
    #   
    # Let me use a different, simpler approach: just use the known Batcher merge.
    
    # Actually, the simplest correct approach for Batcher's odd-even merge:
    def _merge(n_local):
        if n_local == 1:
            return []
        if n_local == 2:
            return [(0, 1)]
        m = n_local // 2
        # Recursively merge odd-indexed (positions 1,3,5,...) and even-indexed (0,2,4,...)
        # Odd positions: 1, 3, 5, ..., up to n_local-1 or n_local-2
        # Even positions: 0, 2, 4, ..., up to n_local-1 or n_local-2
        odd_count = (n_local + 1) // 2  # positions 1,3,5,...
        even_count = n_local // 2       # positions 0,2,4,...
        
        # This recursive structure is complex. Let me use a simpler known construction.
        # I'll just use a straightforward recursive merge.
        
        # Actually, let me just hardcode a known good Batcher sort for small N
        # and use a simpler merge strategy.
        pass
    
    # Let me use a completely different, simpler approach:
    # Use a known efficient sorting network construction.
    
    # For now, let's use a simple but effective construction:
    # Shell sort network (odd-even transposition variant) or Batcher.
    
    # Let me implement Batcher's odd-even merge properly:
    def _odd_even_merge(n_local, off=0):
        if n_local == 1:
            return []
        if n_local == 2:
            return [(off, off + 1)]
        m = n_local // 2
        # Recursively merge the odd-indexed elements and even-indexed elements
        # Odd indices in local: 1, 3, 5, ... (count = (n_local+1)//2)
        # Even indices in local: 0, 2, 4, ... (count = n_local//2)
        
        # For the recursive calls, we need to map local indices to contiguous ranges
        # This is tricky. Let me use a different well-known construction.
        
        # Simpler: use the "brick wall" pattern or just return a known good network.
        # For N=18, let me use a combination approach.
        
        # Actually, I recall that Batcher's merge can be implemented as:
        # merge(n):
        #   if n <= 1: return
        #   if n == 2: compare(0,1); return
        #   m = n//2
        #   merge(m) on elements [0, m)
        #   merge(n-m) on elements [m, n)
        #   for i in range(m):
        #       compare(i, m+i)
        #   for i in range(1, m):
        #       compare(i-1, m+i)  # if m+i < n
        
        # Wait, that's not quite right either. Let me just use a known good network.
        # For N=18, I'll use a layered construction.
        
        # Let me just use a simple but effective approach:
        # 1. Start with odd-even transposition (153 comparators)
        # 2. Apply removal and local search
        # This is what the current program does, but let me make it more efficient.
        
        return []  # placeholder, we'll use a different base network
    
    return []


def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comps.append((i, i + 1))
    return comps


def _batcher_sort(n):
    """Generate Batcher's bitonic sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Bitonic sort: recursively create bitonic sequences then merge
    # For non-power-of-2, pad or use a general approach.
    
    # Let me use a simpler approach: Batcher's odd-even merge sort
    # which works for any n.
    
    def _merge(n_local, off=0):
        if n_local == 1:
            return []
        if n_local == 2:
            return [(off, off + 1)]
        m = n_local // 2
        # Recursively merge first half and second half
        comps = []
        comps.extend(_merge(m, off))
        comps.extend(_merge(n_local - m, off + m))
        # Now do the merge comparisons
        # For i in 0..m-1: compare (off+i, off+m+i)
        for i in range(m):
            j = m + i
            if j < n_local:
                comps.append((off + i, off + j))
        # For i in 1..m-1: compare (off+i-1, off+m+i)
        for i in range(1, m):
            j = m + i
            if j < n_local:
                comps.append((off + i - 1, off + j))
        return comps
    
    def _sort(n_local, off=0):
        if n_local == 1:
            return []
        if n_local == 2:
            return [(off, off + 1)]
        m = n_local // 2
        comps = []
        # Sort first half ascending, second half descending
        comps.extend(_sort(m, off))
        # For descending, we can just reverse the output, but since we're
        # building a network, we need to be careful.
        # Actually, for bitonic sort, we sort one half ascending and the other
        # descending, then merge.
        # Let's just sort both ascending for simplicity (this gives a valid
        # sorting network, just not the most efficient).
        comps.extend(_sort(n_local - m, off + m))
        # Merge
        comps.extend(_merge(n_local, off))
        return comps
    
    return _sort(n, 0)


def _verify_network(comps, n=N):
    """Verify that a network sorts all 2^n Boolean inputs."""
    all_inputs = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        all_inputs[:, i] = (np.arange(1 << n) >> i) & 1
    
    state = all_inputs.copy()
    for (i, j) in comps:
        new_i = np.minimum(state[:, i], state[:, j])
        new_j = np.maximum(state[:, i], state[:, j])
        state[:, i] = new_i
        state[:, j] = new_j
    
    # Check if sorted: no 1 before a 0
    for i in range(n - 1):
        if np.any((state[:, i] == 1) & (state[:, i + 1] == 0)):
            return False
    return True


def _verify_fast(comps, n=N):
    """Faster verification using only a subset of inputs for quick checks."""
    # Use all inputs for final verification, but for intermediate checks
    # we can use a smaller set.
    all_inputs = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        all_inputs[:, i] = (np.arange(1 << n) >> i) & 1
    
    state = all_inputs.copy()
    for (i, j) in comps:
        new_i = np.minimum(state[:, i], state[:, j])
        new_j = np.maximum(state[:, i], state[:, j])
        state[:, i] = new_i
        state[:, j] = new_j
    
    for i in range(n - 1):
        if np.any((state[:, i] == 1) & (state[:, i + 1] == 0)):
            return False
    return True


def _count_errors(comps, n=N):
    """Count how many of the 2^n inputs are not sorted by this network."""
    all_inputs = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        all_inputs[:, i] = (np.arange(1 << n) >> i) & 1
    
    state = all_inputs.copy()
    for (i, j) in comps:
        new_i = np.minimum(state[:, i], state[:, j])
        new_j = np.maximum(state[:, i], state[:, j])
        state[:, i] = new_i
        state[:, j] = new_j
    
    # Count unsorted inputs
    unsorted = 0
    for i in range(n - 1):
        mask = (state[:, i] == 1) & (state[:, i + 1] == 0)
        unsorted = max(unsorted, np.sum(mask))
        # Actually, an input is unsorted if ANY adjacent pair is inverted
        # Let's compute it properly
    unsorted_mask = np.zeros(1 << n, dtype=bool)
    for i in range(n - 1):
        unsorted_mask |= ((state[:, i] == 1) & (state[:, i + 1] == 0))
    return np.sum(unsorted_mask)


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # Start with odd-even transposition network (153 comparators)
    network = _odd_even_transposition(N)
    
    # Verify it's correct
    if not _verify_network(network):
        # Shouldn't happen, but just in case
        return network
    
    best_network = network[:]
    best_count = len(network)
    
    # Phase 1: Greedy removal - try removing each comparator and check if still valid
    i = 0
    while i < len(network):
        if time.time() > DEADLINE:
            break
        
        # Try removing comparator at position i
        test_net = network[:i] + network[i + 1:]
        if _verify_network(test_net):
            network = test_net
            best_network = network[:]
            best_count = len(network)
            # Don't increment i, try removing again from same position
        else:
            i += 1
    
    # Phase 2: Local search - try swapping, inserting, and replacing comparators
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Try insertion: add a comparator at a random position
    for _ in range(200):
        if time.time() > DEADLINE:
            break
        
        # Try inserting a random comparator at a random position
        pos = random.randint(0, len(network))
        comp = random.choice(all_pairs)
        test_net = network[:pos] + [comp] + network[pos:]
        
        # Check if this reduces the error count (or keeps it at 0 and we can remove later)
        # Actually, for local search, we want to try to reduce the count.
        # Since we start from a valid network, let's try:
        # 1. Remove a comparator, then try to fix with a different one
        # 2. Swap two comparators
        # 3. Replace a comparator with a different one
        
        # Strategy: Try removing one and adding another
        if len(network) > 1:
            rm_idx = random.randint(0, len(network) - 1)
            test_net = network[:rm_idx] + network[rm_idx + 1:]
            # Now try to add a different comparator at a random position
            add_pos = random.randint(0, len(test_net))
            add_comp = random.choice(all_pairs)
            test_net2 = test_net[:add_pos] + [add_comp] + test_net[add_pos:]
            
            if len(test_net2) < len(network):
                # Only consider if shorter (shouldn't happen since we removed and added)
                pass
            
            if _verify_network(test_net2) and len(test_net2) < len(network):
                network = test_net2
                best_network = network[:]
                best_count = len(network)
    
    # Phase 3: More aggressive local search with simulated annealing
    # Try random perturbations and accept if valid or with some probability
    current_net = network[:]
    current_valid = _verify_network(current_net)
    
    for step in range(500):
        if time.time() > DEADLINE:
            break
        
        # Random perturbation: remove one, add one, or swap two
        op = random.randint(0, 2)
        
        if op == 0 and len(current_net) > 1:
            # Remove one
            idx = random.randint(0, len(current_net) - 1)
            test_net = current_net[:idx] + current_net[idx + 1:]
        elif op == 1:
            # Add one
            pos = random.randint(0, len(current_net))
            comp = random.choice(all_pairs)
            test_net = current_net[:pos] + [comp] + current_net[pos:]
        else:
            # Swap two adjacent comparators
            if len(current_net) < 2:
                continue
            idx = random.randint(0, len(current_net) - 2)
            test_net = current_net[:]
            test_net[idx], test_net[idx + 1] = test_net[idx + 1], test_net[idx]
        
        if _verify_network(test_net) and len(test_net) < len(current_net):
            current_net = test_net
            if len(current_net) < best_count:
                best_network = current_net[:]
                best_count = len(current_net)
    
    # Final phase: Another round of greedy removal on the best network found
    network = best_network[:]
    i = 0
    while i < len(network):
        if time.time() > DEADLINE:
            break
        
        test_net = network[:i] + network[i + 1:]
        if _verify_network(test_net):
            network = test_net
            best_network = network[:]
            best_count = len(network)
        else:
            i += 1
    
    # Ensure we return a valid network
    if not _verify_network(best_network):
        # Fallback to original
        return _odd_even_transposition(N)
    
    return best_network
