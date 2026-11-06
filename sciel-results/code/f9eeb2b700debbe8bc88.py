"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 16 wires.
    Strategy: Layer-based genetic algorithm.
    1. Start with Batcher's OOM (60 comparators).
    2. Convert to a list of layers (parallel stages).
    3. Evolve the layers: mutate by adding/removing/swapping comparators within layers.
    4. Fitness: valid network with minimum comparators.
    5. Local search: greedy removal.
    """
    
    def batcher_oom(n):
        """Batcher's odd-even mergesort network for n=2^k wires."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        k = n // 2
        # Recursively sort halves
        left = batcher_oom(k)
        right = [(i + k, j + k) for (i, j) in left]
        
        # Odd-even merge
        def merge(a, b, size):
            """Merge two sorted sequences of 'size' elements starting at positions a and b."""
            if size == 1:
                return [(a, b)]
            half = size // 2
            result = []
            result.extend(merge(a, b, half))
            result.extend(merge(a + half, b + half, half))
            # Compare-exchange between the two halves
            for i in range(half):
                result.append((a + i, b + i))
            return result
        
        merged = merge(0, k, k)
        return left + right + merged
    
    # Pre-compute all 2^16 inputs and their sorted targets
    inputs = np.arange(2**N, dtype=np.uint32)
    # Target for k ones: top k bits are 1, rest 0 (since min goes to lower wire)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) for k in popcounts], dtype=np.uint32)
    
    def simulate(net):
        """Simulate the network on all 65536 inputs, return final state array."""
        masks = inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            # Condition: wire i has 1, wire j has 0 -> swap
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return masks
    
    def verify(net):
        """Check if net sorts all inputs."""
        if not net:
            return False
        result = simulate(net)
        return np.array_equal(result, targets)
    
    def layers_to_net(layers):
        """Flatten layers into a list of comparators."""
        net = []
        for layer in layers:
            net.extend(layer)
        return net
    
    def net_to_layers(net):
        """
        Convert a linear network into layers.
        A layer is a set of comparators that can be executed in parallel.
        We greedily assign comparators to layers: a comparator (i, j) can be in the next layer
        if neither i nor j was used in the previous layer (simple constraint).
        Actually, for correctness of the *sequence*, the order matters.
        If we group into layers, we must ensure that within a layer, comparators are disjoint.
        If they are not disjoint, the "layer" is not truly parallel, but the *sequence* of comparators
        in the flattened list is what defines the network.
        So, we just need to partition the list into contiguous blocks where each block consists
        of pairwise disjoint comparators.
        """
        layers = []
        current_layer = []
        used_wires = set()
        
        for comp in net:
            i, j = comp
            if i in used_wires or j in used_wires:
                # Cannot add to current layer
                if current_layer:
                    layers.append(current_layer)
                current_layer = []
                used_wires = set()
            
            # Check if adding this comp keeps disjointness within current_layer
            # Since we just reset or are adding to empty, we just need to check against current_layer
            # Wait, the logic above is slightly flawed if we just reset.
            # Let's do it properly:
            # Try to add (i, j) to current_layer.
            # If it conflicts with any wire in current_layer, start a new layer.
            
            # Re-evaluate:
            # The logic above: if i or j is in used_wires (from current_layer), we close current_layer.
            # Then we add to new current_layer.
            # But what if the new layer also has conflicts?
            # Since we reset, current_layer is empty, so no conflict.
            
            # However, the check `if i in used_wires or j in used_wires` happens BEFORE adding.
            # If we close the layer, used_wires becomes empty.
            # Then we add (i, j) to the new empty layer.
            # This works.
            
            current_layer.append(comp)
            used_wires.add(i)
            used_wires.add(j)
        
        if current_layer:
            layers.append(current_layer)
            
        return layers
    
    # Generate Batcher's network for 16 wires
    batcher_net = batcher_oom(N)
    
    # Verify Batcher's network
    if not verify(batcher_net):
        # Fallback: use odd-even transposition (should not happen)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Initial layers
    current_layers = net_to_layers(batcher_net)
    best_net = list(batcher_net)
    best_len = len(best_net)
    
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Helper to get a random valid comparator (i, j) with i < j
    def random_comparator():
        i = random.randint(0, N - 1)
        j = random.randint(i + 1, N - 1)
        return (i, j)
    
    # Systematic greedy reduction
    def greedy_reduce_step(net):
        """Try to remove any comparator. Return (new_net, removed_flag)."""
        for idx in range(len(net) - 1, -1, -1):
            candidate = net[:idx] + net[idx+1:]
            if verify(candidate):
                return candidate, True
        return net, False
    
    def full_greedy_reduce(net, max_iterations=50):
        """Apply greedy reduction repeatedly until no more removals possible."""
        current = list(net)
        for _ in range(max_iterations):
            if time.time() - start_time > deadline - 5.0:
                break
            current, removed = greedy_reduce_step(current)
            if not removed:
                break
        return current
    
    # Apply initial greedy reduction
    net = full_greedy_reduce(batcher_net)
    if len(net) < best_len:
        best_net = list(net)
        best_len = len(net)
    
    # Main search loop with multiple strategies
    while time.time() - start_time < deadline - 3.0:
        strategy = random.random()
        
        if strategy < 0.35:
            # Add a random comparator, then reduce
            comp = random_comparator()
            pos = random.randint(0, len(net))
            candidate = net[:pos] + [comp] + net[pos:]
            reduced = full_greedy_reduce(candidate, max_iterations=15)
            if len(reduced) < len(net):
                net = reduced
                if len(net) < best_len:
                    best_net = list(net)
                    best_len = len(net)
                    
        elif strategy < 0.65:
            # Swap two comparators, then reduce
            if len(net) < 2:
                continue
            i1 = random.randint(0, len(net) - 1)
            i2 = random.randint(0, len(net) - 1)
            if i1 == i2:
                continue
            candidate = net.copy()
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            if verify(candidate):
                reduced = full_greedy_reduce(candidate, max_iterations=15)
                if len(reduced) < len(net):
                    net = reduced
                    if len(net) < best_len:
                        best_net = list(net)
                        best_len = len(net)
        else:
            # Move a comparator to a different position, then reduce
            if len(net) < 2:
                continue
            src = random.randint(0, len(net) - 1)
            dst = random.randint(0, len(net) - 1)
            if src == dst:
                continue
            candidate = net.copy()
            comp = candidate.pop(src)
            candidate.insert(dst, comp)
            if verify(candidate):
                reduced = full_greedy_reduce(candidate, max_iterations=15)
                if len(reduced) < len(net):
                    net = reduced
                    if len(net) < best_len:
                        best_net = list(net)
                        best_len = len(net)
    
    # Final polish
    final_net = full_greedy_reduce(best_net, max_iterations=30)
    if len(final_net) < best_len:
        best_net = final_net
        best_len = len(final_net)
    
    return best_net
# EVOLVE-BLOCK-END
