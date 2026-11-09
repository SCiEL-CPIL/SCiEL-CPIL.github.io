"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
# Precompute initial wire masks for fast verification using numpy
# Wire i outputs 1 for input pattern k iff bit i of k is 1
# Represent as numpy array of shape (16, 65536) with uint8

def _build_initial_masks_numpy():
    """Build initial numpy masks for 16 wires."""
    # Create all 65536 input patterns as a (16, 65536) uint8 array
    # where row i, col k = 1 if bit i of k is 1
    masks = np.zeros((16, 65536), dtype=np.uint8)
    for i in range(16):
        # Bit i is 1 for k where (k >> i) & 1 == 1
        # Pattern: 2^i zeros, 2^i ones, repeating
        block = 1 << i
        period = 1 << (i + 1)
        # Create pattern for one period
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[block:] = 1
        # Tile to fill 65536 entries
        num_periods = 65536 // period
        masks[i] = np.tile(pattern, num_periods)
    return masks

_INITIAL_MASKS_NUMPY = _build_initial_masks_numpy()

def _verify_network_numpy(comps):
    """Verify sorting network using zero-one principle with numpy."""
    if not comps:
        return True
    # Copy initial masks
    wires = _INITIAL_MASKS_NUMPY.copy()
    for (i, j) in comps:
        a = wires[i]
        b = wires[j]
        wires[i] = np.minimum(a, b)  # min: 1 only if both are 1
        wires[j] = np.maximum(a, b)  # max: 1 if either is 1
    
    # Check sorted: wire[i] <= wire[i+1] for all i (element-wise)
    for i in range(15):
        if not np.all(wires[i] <= wires[i+1]):
            return False
    return True

def _verify_network(comps):
    """Verify sorting network using zero-one principle with numpy (wrapper)."""
    return _verify_network_numpy(comps)

def _batcher_oe_merge_sort(n):
    """Build Batcher's odd-even merge sort network."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        # Compare (i, n+i) for i in 0..n-1
        comps = [(i, n + i) for i in range(n)]
        # Recursively merge even-indexed and odd-indexed subsequences
        # Even indices: 0, 2, 4, ... -> map to 0, 1, 2, ...
        # Odd indices: 1, 3, 5, ... -> map to 0, 1, 2, ...
        even_merge = oe_merge_network(n // 2)
        odd_merge = oe_merge_network(n // 2)
        # Apply even merge on even positions
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
        # Apply odd merge on odd positions
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
        return comps
    
    def sort_network(n):
        """Sorting network using odd-even merge sort."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 == 1:
            # Handle odd number: sort n-1, then insert the last element
            base = sort_network(n - 1)
            # Insert element n-1 into sorted network of n-1 elements
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        # Even: split into two halves
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        # Odd-even merge the two sorted halves
        merge = oe_merge_network(half)
        return left + right + merge
    
    return sort_network(n)

def _local_search(comps, max_iters=200000, deadline=None):
    """Local search to minimize comparator count while maintaining correctness."""
    best = list(comps)
    best_count = len(best)
    
    start_time = time.time()
    iters = 0
    no_improve = 0
    
    import random
    random.seed(42)
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random removal of a comparator
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                no_improve = 0
            else:
                no_improve += 1
        
        iters += 1
        
        if no_improve > 20000:
            break
    
    return best

def _layer_based_search(comps, deadline):
    """
    Layer-based genetic/local search:
    - Represent network as sequence of layers (sets of disjoint comparators).
    - Mutate by replacing a layer with a random permutation-based layer.
    - Also try removing redundant layers/comparators.
    - Fitness: comparator count (lower is better), subject to correctness.
    """
    import random
    random.seed(123)
    
    best = list(comps)
    best_count = len(best)
    start_time = time.time()
    no_improve = 0
    
    # Build initial layer structure from flat list
    # A layer is a set of comparators that are pairwise disjoint (no shared wire)
    def _to_layers(comps):
        """Group comparators into layers where each layer has disjoint wires."""
        layers = []
        used = set()
        for c in comps:
            i, j = c
            if i in used or j in used:
                layers.append([])
                used = set()
            layers[-1].append(c)
            used.add(i)
            used.add(j)
        return layers
    
    def _layers_to_comps(layers):
        """Flatten layers back to a list of comparators."""
        comps = []
        for layer in layers:
            comps.extend(layer)
        return comps
    
    def _random_layer():
        """Generate a random layer: a random matching of wires."""
        wires = list(range(16))
        random.shuffle(wires)
        layer = []
        for k in range(0, 14, 2):
            a, b = wires[k], wires[k+1]
            if a > b:
                a, b = b, a
            layer.append((a, b))
        return layer
    
    def _count_comparators(layers):
        return sum(len(layer) for layer in layers)
    
    # Start with layer representation
    layers = _to_layers(best)
    
    iters = 0
    while iters < 500000:
        if time.time() > deadline:
            break
        
        # Choose mutation type
        r = random.random()
        
        if r < 0.4:
            # Replace a random layer with a new random layer
            if not layers:
                iters += 1
                continue
            li = random.randint(0, len(layers) - 1)
            new_layers = layers[:li] + [_random_layer()] + layers[li+1:]
            trial = _layers_to_comps(new_layers)
            if _verify_network(trial):
                cnt = len(trial)
                if cnt < best_count:
                    best = trial
                    best_count = cnt
                    layers = new_layers
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif r < 0.7:
            # Remove a random comparator from a random layer
            if not layers:
                iters += 1
                continue
            li = random.randint(0, len(layers) - 1)
            if len(layers[li]) <= 1:
                iters += 1
                continue
            ci = random.randint(0, len(layers[li]) - 1)
            new_layers = [list(l) for l in layers]
            new_layers[li].pop(ci)
            if not new_layers[li]:
                new_layers.pop(li)
            trial = _layers_to_comps(new_layers)
            if _verify_network(trial):
                cnt = len(trial)
                if cnt < best_count:
                    best = trial
                    best_count = cnt
                    layers = new_layers
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif r < 0.85:
            # Swap two layers
            if len(layers) < 2:
                iters += 1
                continue
            li, lj = random.sample(range(len(layers)), 2)
            new_layers = layers[:]
            new_layers[li], new_layers[lj] = new_layers[lj], new_layers[li]
            trial = _layers_to_comps(new_layers)
            if _verify_network(trial):
                cnt = len(trial)
                if cnt < best_count:
                    best = trial
                    best_count = cnt
                    layers = new_layers
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            # Add a new random layer at a random position (exploration)
            pos = random.randint(0, len(layers))
            new_layers = layers[:pos] + [_random_layer()] + layers[pos:]
            trial = _layers_to_comps(new_layers)
            # Only accept if it somehow helps (unlikely to reduce count, but might enable later removals)
            # For now, skip adding layers as they increase count
            pass
        
        iters += 1
        
        if no_improve > 50000:
            break
    
    return best

def _final_polish(comps, deadline):
    """
    Final high-precision polish pass:
    - Systematically try removing each comparator and check if the network still sorts.
    - Use a greedy approach: iterate through all comparators, try to remove each one.
    - This is a local search with a tighter tolerance (exact verification).
    """
    import random
    random.seed(999)
    
    best = list(comps)
    best_count = len(best)
    
    # Multiple passes to converge
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing each comparator in order
        for idx in range(len(best)):
            if time.time() > deadline:
                break
            
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                improved = True
                break  # Restart the pass since the list changed
        
        if not improved:
            # No more improvements possible in this pass, try random order
            order = list(range(len(best)))
            random.shuffle(order)
            for idx in order:
                if time.time() > deadline:
                    break
                
                trial = best[:idx] + best[idx+1:]
                if _verify_network(trial):
                    best = trial
                    best_count = len(best)
                    improved = True
                    break
            
            if not improved:
                break
    
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    if not _verify_network(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to reduce comparators
    # Safety margin: assume we have ~85s left
    deadline = time.time() + 80.0
    
    # First pass: standard local search (random removal)
    best = _local_search(comps, max_iters=200000, deadline=deadline)
    
    # Second pass: layer-based search for further refinement
    remaining = deadline - time.time()
    if remaining > 5.0:
        best = _layer_based_search(best, deadline)
    
    # Third pass: final polish with systematic removal
    remaining = deadline - time.time()
    if remaining > 2.0:
        best = _final_polish(best, deadline)
    
    return best
# EVOLVE-BLOCK-END
