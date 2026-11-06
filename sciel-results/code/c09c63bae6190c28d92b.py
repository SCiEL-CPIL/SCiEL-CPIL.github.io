"""
Size-efficient sorting network search for N=17 wires.
Uses a depth-parallel layer representation and local search to minimize comparator count.
"""
import time
import random
import numpy as np

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072

# Precompute all boolean inputs as a bit matrix for fast verification
# Each row is one boolean input (17 bits)
# We'll use numpy for speed


def _build_bool_matrix():
    """Build numpy array of shape (131072, 17) with all boolean inputs."""
    # Use bit manipulation
    arr = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.int8)
    for bit in range(N):
        # For bit position 'bit', the pattern is: 0 for first 2^(N-1-bit), 1 for next 2^(N-1-bit)
        # Actually simpler: for each input index, bit b is (index >> b) & 1
        # But that's slow in a loop. Let's use broadcasting.
        # arr[:, bit] = (np.arange(NUM_BOOL_INPUTS) >> bit) & 1
        # This creates a large array each time. Let's do it differently.
        # Use the fact that bit b has period 2^(b+1), with first half 0, second half 1.
        period = 1 << (bit + 1)
        half = 1 << bit
        # Create pattern of length 'period' then tile
        pattern = np.zeros(period, dtype=np.int8)
        pattern[half:] = 1
        # Tile to fill NUM_BOOL_INPUTS
        num_tiles = NUM_BOOL_INPUTS // period
        arr[:, bit] = np.tile(pattern, num_tiles)
    return arr


def _apply_layer(state, layer):
    """Apply a layer of disjoint comparators to a state array.
    state: numpy array of shape (num_inputs, N)
    layer: list of (i, j) pairs, all disjoint
    Returns new state.
    """
    if not layer:
        return state
    new_state = state.copy()
    for i, j in layer:
        a = new_state[:, i]
        b = new_state[:, j]
        new_state[:, i] = np.minimum(a, b)
        new_state[:, j] = np.maximum(a, b)
    return new_state


def _verify_network(layers):
    """Verify that the network sorts all boolean inputs.
    layers: list of layers, each layer is a list of (i,j) pairs.
    Returns True if sorted, False otherwise.
    """
    if not layers:
        return False
    state = _build_bool_matrix()
    for layer in layers:
        state = _apply_layer(state, layer)
    # Check if each row is sorted (non-decreasing)
    # A row is sorted if all consecutive pairs are non-decreasing
    for i in range(N - 1):
        if np.any(state[:, i] > state[:, i + 1]):
            return False
    return True


def _count_comparators(layers):
    """Count total comparators in the layer representation."""
    return sum(len(layer) for layer in layers)


def _depth(layers):
    """Return the depth (number of layers)."""
    return len(layers)


def _make_seed_network():
    """Create the odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers


def _random_network():
    """Generate a random network with some comparators."""
    layers = []
    num_layers = random.randint(8, 20)
    for _ in range(num_layers):
        layer = []
        used = set()
        # Randomly add some disjoint comparators
        for i in range(N):
            if random.random() < 0.3 and i not in used:
                j = random.randint(i + 1, N - 1)
                if j not in used:
                    layer.append((i, j))
                    used.add(i)
                    used.add(j)
        layers.append(layer)
    return layers


def _mutate_network(layers):
    """Mutate a network by adding, removing, or moving comparators."""
    new_layers = [list(layer) for layer in layers]
    mutation = random.choice(['add', 'remove', 'move', 'swap'])
    
    if mutation == 'add':
        # Add a random comparator to a random layer
        if new_layers:
            layer_idx = random.randint(0, len(new_layers) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            # Check if i and j are already used in this layer
            used = set()
            for (a, b) in new_layers[layer_idx]:
                used.add(a)
                used.add(b)
            if i not in used and j not in used:
                new_layers[layer_idx].append((i, j))
    elif mutation == 'remove':
        # Remove a random comparator
        if new_layers:
            # Pick a random layer with comparators
            non_empty = [i for i, l in enumerate(new_layers) if l]
            if non_empty:
                layer_idx = random.choice(non_empty)
                comp_idx = random.randint(0, len(new_layers[layer_idx]) - 1)
                new_layers[layer_idx].pop(comp_idx)
    elif mutation == 'move':
        # Move a comparator from one layer to another
        if len(new_layers) > 1:
            non_empty = [i for i, l in enumerate(new_layers) if l]
            if non_empty:
                src_idx = random.choice(non_empty)
                comp = new_layers[src_idx].pop(random.randint(0, len(new_layers[src_idx]) - 1))
                dst_idx = random.randint(0, len(new_layers) - 1)
                # Check if comp can be added to dst layer
                used = set()
                for (a, b) in new_layers[dst_idx]:
                    used.add(a)
                    used.add(b)
                if comp[0] not in used and comp[1] not in used:
                    new_layers[dst_idx].append(comp)
    elif mutation == 'swap':
        # Swap two comparators in the same layer
        if new_layers:
            non_empty = [i for i, l in enumerate(new_layers) if len(l) >= 2]
            if non_empty:
                layer_idx = random.choice(non_empty)
                i1 = random.randint(0, len(new_layers[layer_idx]) - 1)
                i2 = random.randint(0, len(new_layers[layer_idx]) - 1)
                if i1 != i2:
                    new_layers[layer_idx][i1], new_layers[layer_idx][i2] = new_layers[layer_idx][i2], new_layers[layer_idx][i1]
    
    # Remove empty layers
    new_layers = [l for l in new_layers if l]
    return new_layers


def _optimize_network(layers, deadline):
    """Local search to optimize the network."""
    start_time = time.time()
    best_layers = [list(l) for l in layers]
    best_count = _count_comparators(best_layers)
    
    # Verify initial network
    if not _verify_network(best_layers):
        return _make_seed_network()
    
    current_layers = [list(l) for l in layers]
    current_count = _count_comparators(current_layers)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        mutated = _mutate_network(current_layers)
        mut_count = _count_comparators(mutated)
        
        # Only verify if the mutation could be an improvement or is close
        if mut_count < current_count or (mut_count == current_count and random.random() < 0.1):
            if _verify_network(mutated):
                if mut_count < current_count:
                    current_layers = [list(l) for l in mutated]
                    current_count = mut_count
                    no_improve_count = 0
                    if current_count < best_count:
                        best_layers = [list(l) for l in current_layers]
                        best_count = current_count
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Restart from best
            current_layers = [list(l) for l in best_layers]
            current_count = best_count
            no_improve_count = 0
    
    return best_layers


def construct():
    """Construct a sorting network for 17 wires."""
    deadline = time.time() + 75  # 75s safety margin before 90s kill
    
    # Start with seed network
    seed_layers = _make_seed_network()
    seed_count = _count_comparators(seed_layers)
    
    # Try local search from seed
    optimized = _optimize_network(seed_layers, deadline)
    opt_count = _count_comparators(optimized)
    
    # If optimization didn't help, try a few random restarts
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            rand_layers = _random_network()
            if _verify_network(rand_layers):
                rand_opt = _optimize_network(rand_layers, deadline)
                rand_count = _count_comparators(rand_opt)
                if rand_count < opt_count:
                    optimized = rand_opt
                    opt_count = rand_count
    
    # Convert layers to flat list of comparators
    result = []
    for layer in optimized:
        result.extend(layer)
    
    # Safety: if something went wrong, return seed
    if len(result) > seed_count or len(result) == 0:
        result = []
        for layer in seed_layers:
            result.extend(layer)
    
    return result


# Precompute bool matrix at module level for speed
_BOOL_MATRIX = _build_bool_matrix()


def _apply_layer(state, layer):
    """Apply a layer of disjoint comparators to a state array.
    state: numpy array of shape (num_inputs, N)
    layer: list of (i, j) pairs, all disjoint
    Returns new state.
    """
    if not layer:
        return state
    new_state = state.copy()
    for i, j in layer:
        a = new_state[:, i]
        b = new_state[:, j]
        new_state[:, i] = np.minimum(a, b)
        new_state[:, j] = np.maximum(a, b)
    return new_state


def _verify_network(layers):
    """Verify that the network sorts all boolean inputs."""
    if not layers:
        return False
    state = _BOOL_MATRIX.copy()
    for layer in layers:
        state = _apply_layer(state, layer)
    # Check if each row is sorted
    for i in range(N - 1):
        if np.any(state[:, i] > state[:, i + 1]):
            return False
    return True
