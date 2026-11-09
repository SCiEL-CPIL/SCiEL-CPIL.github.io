"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _batcher_sort_network(n):
    """Generate Batcher's bitonic sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Bitonic sort: recursively build bitonic sequences then merge
    # For simplicity, use the standard bitonic sort construction
    def _bitonic_merge(n, direction=1):
        """Generate comparators for bitonic merge of n elements."""
        if n <= 1:
            return []
        comparators = []
        # Compare elements that are n/2 apart, then recursively merge halves
        for i in range(n // 2):
            a = i
            b = i + n // 2
            if direction == 1:
                comparators.append((a, b))
            else:
                comparators.append((a, b))
        # Recursively merge the two halves
        left = _bitonic_merge(n // 2, direction)
        for i, j in left:
            comparators.append((i, j))
        right = _bitonic_merge(n // 2, direction)
        for i, j in right:
            comparators.append((i + n // 2, j + n // 2))
        return comparators
    
    def _build_bitonic(n, direction=1):
        """Generate comparators to build a bitonic sequence of length n."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comparators = []
        # Build two bitonic sequences of length n/2
        left = _build_bitonic(n // 2, direction)
        for i, j in left:
            comparators.append((i, j))
        right = _build_bitonic(n // 2, -direction)
        for i, j in right:
            comparators.append((i + n // 2, j + n // 2))
        # Merge the two bitonic sequences
        merge = _bitonic_merge(n, direction)
        comparators.extend(merge)
        return comparators
    
    return _build_bitonic(n, 1)

def _validate_network(net, n=18):
    """Vectorized validation using numpy. Returns True if network sorts all 2^n inputs."""
    if not net:
        return n <= 1
    
    # Generate all 2^n input patterns as a (2^n, n) array of 0s and 1s
    # Use bit manipulation: for mask in range(2^n), bit i of mask gives wire i
    all_masks = np.arange(1 << n, dtype=np.uint32)
    
    # Extract bits: wires[m, i] = (m >> i) & 1
    # This is expensive to do naively, so let's use a loop over bits
    wires = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        wires[:, i] = ((all_masks >> i) & 1).astype(np.uint8)
    
    # Apply each comparator
    for i, j in net:
        # Compare wires[:, i] and wires[:, j]
        # min goes to i, max goes to j
        wi = wires[:, i].copy()
        wj = wires[:, j].copy()
        # Where wi > wj, swap
        mask = wi > wj
        wires[mask, i] = wj[mask]
        wires[mask, j] = wi[mask]
    
    # Check if sorted: for each row, check if non-decreasing
    # A row is sorted if all adjacent pairs satisfy wires[k] <= wires[k+1]
    for k in range(n - 1):
        if np.any(wires[:, k] > wires[:, k + 1]):
            return False
    
    return True

def _validate_network_fast(net, n=18):
    """Faster validation using the zero-one principle with early termination."""
    if not net:
        return n <= 1
    
    # For each possible count of 1s (0 to n), check that the network produces sorted output
    # But we need to check all 2^n inputs, not just by count
    # Actually, the zero-one principle says we only need to check binary inputs
    # Let's use a smarter approach: check all 2^n inputs but with early termination
    
    # Use numpy for vectorized operations
    all_masks = np.arange(1 << n, dtype=np.uint32)
    
    # We'll simulate in chunks to avoid memory issues
    chunk_size = 65536
    for start in range(0, 1 << n, chunk_size):
        end = min(start + chunk_size, 1 << n)
        masks = all_masks[start:end]
        
        wires = np.zeros((len(masks), n), dtype=np.uint8)
        for i in range(n):
            wires[:, i] = ((masks >> i) & 1).astype(np.uint8)
        
        for i, j in net:
            wi = wires[:, i].copy()
            wj = wires[:, j].copy()
            mask = wi > wj
            wires[mask, i] = wj[mask]
            wires[mask, j] = wi[mask]
        
        # Check sorted
        for k in range(n - 1):
            if np.any(wires[:, k] > wires[:, k + 1]):
                return False
    
    return True

def _genetic_algorithm(n=18, pop_size=50, generations=100, timeout=50):
    """Evolutionary search for minimal sorting network."""
    start_time = time.time()
    
    # Initialize population with Batcher's bitonic sort and variations
    base_net = _batcher_sort_network(32)  # Use 32-wire network, then restrict to 18 wires
    # Actually, let's use a proper construction for 18 wires
    
    # Better: use odd-even merge sort for 18 wires
    def _odd_even_merge_sort(n):
        """Generate odd-even merge sort network for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comparators = []
        
        # Split into even and odd indexed elements
        even_indices = list(range(0, n, 2))
        odd_indices = list(range(1, n, 2))
        
        # Recursively sort even and odd subsequences
        even_net = _odd_even_merge_sort(len(even_indices))
        for i, j in even_net:
            comparators.append((even_indices[i], even_indices[j]))
        
        odd_net = _odd_even_merge_sort(len(odd_indices))
        for i, j in odd_net:
            comparators.append((odd_indices[i], odd_indices[j]))
        
        # Merge the two sorted sequences
        # Odd-even merge step
        def _odd_even_merge(n, offset=0):
            """Merge two sorted sequences of total length n."""
            if n <= 1:
                return []
            
            comparators = []
            # Compare (offset+0, offset+1), (offset+2, offset+3), ...
            for k in range(n // 2):
                a = offset + 2 * k
                b = offset + 2 * k + 1
                comparators.append((a, b))
            
            # Recursively merge the even and odd positions of the result
            # The even positions are offset+0, offset+2, ...
            # The odd positions are offset+1, offset+3, ...
            even_pos = list(range(offset, offset + n, 2))
            odd_pos = list(range(offset + 1, offset + n, 2))
            
            # This recursive merge is complex; let's use a simpler approach
            # Just compare adjacent pairs in a specific pattern
            # For odd-even merge, after the first compare-exchange phase,
            # we recursively merge the even and odd indexed elements
            
            if n > 2:
                # Recursively merge even-indexed elements
                even_merge = _odd_even_merge(len(even_pos), 0)
                for i, j in even_merge:
                    comparators.append((even_pos[i], even_pos[j]))
                
                # Recursively merge odd-indexed elements  
                odd_merge = _odd_even_merge(len(odd_pos), 0)
                for i, j in odd_merge:
                    comparators.append((odd_pos[i], odd_pos[j]))
            
            return comparators
        
        merge_comparators = _odd_even_merge(n)
        comparators.extend(merge_comparators)
        
        return comparators
    
    # Get initial network
    init_net = _odd_even_merge_sort(n)
    
    # If initial network is not valid, fall back to odd-even transposition
    if not _validate_network_fast(init_net, n):
        init_net = [(i, i + 1) for phase in range(n)
                    for i in range(phase % 2, n - 1, 2)]
    
    # Initialize population
    population = [list(init_net)]
    # Add some variations
    for _ in range(pop_size - 1):
        net = list(init_net)
        # Randomly add some extra comparators
        for _ in range(random.randint(0, 5)):
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            net.append((i, j))
        population.append(net)
    
    best_net = None
    best_len = float('inf')
    
    for gen in range(generations):
        if time.time() - start_time > timeout:
            break
        
        # Evaluate population
        evaluated = []
        for net in population:
            if len(net) < best_len and _validate_network_fast(net, n):
                evaluated.append((net, len(net), True))
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = list(net)
            else:
                evaluated.append((net, len(net), False))
        
        # Selection: keep top half
        evaluated.sort(key=lambda x: (not x[2], x[1]))
        survivors = [x[0] for x in evaluated[:pop_size // 2]]
        
        # Crossover and mutation
        new_population = list(survivors)
        while len(new_population) < pop_size:
            if len(survivors) < 2:
                break
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            
            # Crossover: take a prefix from parent1 and suffix from parent2
            if len(parent1) > 0 and len(parent2) > 0:
                point = random.randint(0, min(len(parent1), len(parent2)))
                child = list(parent1[:point]) + list(parent2[point:])
            else:
                child = list(parent1)
            
            # Mutation: randomly add, remove, or swap comparators
            if random.random() < 0.3 and len(child) > 0:
                # Remove a random comparator
                idx = random.randint(0, len(child) - 1)
                child.pop(idx)
            
            if random.random() < 0.3:
                # Add a random comparator
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                child.append((i, j))
            
            if random.random() < 0.2 and len(child) > 1:
                # Swap two adjacent comparators
                idx = random.randint(0, len(child) - 2)
                child[idx], child[idx + 1] = child[idx + 1], child[idx]
            
            new_population.append(child)
        
        population = new_population
    
    if best_net is not None:
        return best_net
    return init_net

def construct():
    """Construct a sorting network on 18 wires using genetic algorithm."""
    start_time = time.time()
    timeout = 50  # Leave 10s for safety
    
    # Try genetic algorithm
    net = _genetic_algorithm(N, pop_size=30, generations=80, timeout=timeout)
    
    # Verify the result
    if not _validate_network_fast(net, N):
        # Fall back to odd-even transposition
        net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    return net
# EVOLVE-BLOCK-END
