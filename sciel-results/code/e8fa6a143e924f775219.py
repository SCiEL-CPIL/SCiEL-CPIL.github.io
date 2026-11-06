"""
Strategy:
1. Exploit the structural properties of optimal sorting networks.
   Optimal networks for N=16 (and generally) are often "layered" or "bitonic-like",
   where comparators at depth d only connect wires i, j such that (j-i) is a power of 2.
   Specifically, for N=16, the maximum distance is 8 (2^3), then 4, 2, 1.
   However, optimal networks might mix these. But a strong heuristic is that
   most comparators are local or semi-local.
   
   Actually, a more robust structural insight: 
   The optimal 16-wire network is known to be 55 comparators.
   The current SA approach is slow because it does random moves.
   A DFS with a good heuristic ordering of comparators to try (e.g., by "usefulness" or "standard position")
   combined with early pruning (if partial network fails to sort a subset of inputs) can be faster.
   
   However, checking validity of a partial network is hard.
   
   Alternative Strategy: 
   Use the "Zero-One Principle" more efficiently.
   Instead of checking all 65536 inputs for every move, we can maintain the set of "critical" inputs
   that are currently unsorted by the network. If we remove a comparator, we only need to check
   if the critical inputs become sorted. If we add/replace, we need to check if the new comparator
   helps sort the critical inputs.
   
   But the most promising structural change is:
   **Layered Search**:
   Assume the network consists of L layers. In each layer, comparators are disjoint (parallel).
   This is a standard assumption for high-performance networks.
   We search for the sequence of layers.
   Each layer is a matching (set of disjoint pairs).
   The number of layers (depth) is typically 10-12 for 16 wires.
   Total comparators = sum of comparators in each layer.
   We want to minimize the sum.
   
   This reduces the search space significantly because:
   1. Comparators in the same layer cannot share wires.
   2. We can enumerate possible layers (matchings) and combine them.
   
   However, the optimal 55-comparator network for 16 wires might not be strictly layered in the sense of disjoint matchings at every depth? 
   Actually, standard sorting networks are defined by layers of disjoint comparators. 
   The "parallel depth" is the number of layers.
   The "size" is the total number of comparators.
   A network with 55 comparators and depth 12 is possible.
   
   Let's implement a search over layered networks.
   State: list of layers. Each layer is a set of disjoint (i, j) pairs.
   Move: Add a layer, or modify a layer.
   Cost: Total comparators.
   
   To make it efficient:
   1. Start with a known valid layered network (e.g., Batcher's).
   2. Try to remove comparators from layers. If a layer becomes empty, remove it.
   3. Try to replace a comparator in a layer with another disjoint comparator (within that layer's constraints? No, layers are just sets of disjoint pairs. We can change which pairs are in the layer).
   
   Actually, a simpler structural change:
   **Greedy Construction with Backtracking (DFS)**
   Build the network layer by layer.
   For each layer, try a set of disjoint comparators.
   Prune if the partial network cannot possibly sort the remaining inputs.
   Heuristic: Prioritize comparators that fix the most "disorder" in the current worst-case input.
   
   This is complex. Let's try a different structural idea:
   **Evolutionary Algorithm with a "Gene" representation of layers.**
   Each individual is a list of layers.
   Mutation: Change a layer by swapping a comparator, adding/removing a pair (maintaining disjointness).
   Crossover: Take layers from two parents.
   
   But the key is the **validity check**.
   We can speed up the validity check by using a "worst-case" input set.
   If a network sorts the "worst" inputs (e.g., those with maximum disorder), it likely sorts all.
   But for correctness, we must check all 2^16.
   
   Let's stick to the **Layered Representation** and use **Simulated Annealing on the layers**.
   This is different from the current program which uses a flat list.
   The constraint that layers are matchings reduces the search space.
   
   Implementation:
   1. Represent network as list of matchings (layers).
   2. Initial: Batcher's network (which is layered).
   3. Moves:
      a) Remove a comparator from a layer.
      b) Add a comparator to a layer (if it doesn't conflict with existing pairs in that layer).
      c) Move a comparator from one layer to another (if it fits).
      d) Swap two comparators in the same layer (if they remain disjoint? No, swapping pairs in a matching is just reordering, which doesn't change the network function, only depth order. Wait, depth order matters! Comparator (0,1) then (2,3) is different from (2,3) then (0,1) if they share wires? No, if they are disjoint, order doesn't matter. If they are in the same layer, they MUST be disjoint. So order within a layer doesn't matter.
      
      So the state is: List of Matchings.
      
   4. Validity check: Flatten layers, apply to all inputs.
   
   This structure ensures that any network generated is a valid "layered" sorting network.
   Does the optimal 55-comparator network fit this? Yes, all standard sorting networks are layered.
   
   Why is this better than the current SA?
   Current SA allows "illegal" intermediate states? No, it checks validity.
   But current SA can create networks with comparators that are not in a clean layered structure?
   Actually, any sequence of comparators can be viewed as layers (one comparator per layer is a valid layering, just depth = size).
   But the current SA doesn't enforce disjointness in "depth". It just has a flat list.
   The flat list allows comparators (0,1) and (1,2) to be "close" in the list, which is fine.
   
   The structural difference is that we **enforce disjointness within layers** and **search over the composition of layers**.
   This might help escape local minima where the flat list is stuck in a bad ordering, by allowing "re-shuffling" of comparators into different layers more naturally.
   
   Also, we can use a **greedy initialization** that builds a minimal layered network from scratch?
   No, start with Batcher's.
   
   Let's refine the moves for the Layered SA:
   - Remove a comparator from a layer.
   - Insert a comparator into a layer (if disjoint from others in that layer).
   - "Slide" a comparator to an adjacent layer (if it fits there).
   
   This allows changing the depth/parallelism structure, which the flat list SA does implicitly but less efficiently.
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _flat_to_layers(comps):
    """
    Convert a flat list of comparators into a list of layers (matchings).
    Greedy assignment: assign each comparator to the first layer where it fits (disjoint).
    If it doesn't fit, create a new layer.
    This is not optimal layering, but it's a valid representation.
    """
    layers = []
    for c in comps:
        placed = False
        for l_idx in range(len(layers)):
            # Check if c is disjoint from all pairs in layers[l_idx]
            ok = True
            for p in layers[l_idx]:
                if c[0] in p or c[1] in p:
                    ok = False
                    break
            if ok:
                layers[l_idx].append(c)
                placed = True
                break
        if not placed:
            layers.append([c])
    return layers

def _layers_to_flat(layers):
    """Flatten layers back to a list of comparators."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all 2^16 boolean inputs.
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate initial network (Batcher's)
    initial_comps = _generate_batcher_network()
    
    # Verify initial network
    if not _is_sorted_network_vectorized(initial_comps, bits):
        # Fallback: Odd-Even Transposition
        initial_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    initial_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    initial_comps.append((j, j + 1))
    
    best_comps = initial_comps[:]
    best_len = len(best_comps)
    
    # Convert to layered representation
    current_layers = _flat_to_layers(initial_comps)
    
    rng = np.random.default_rng(42)
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        if time.time() > deadline - 1.0:
            break
            
        move_type = rng.random()
        
        # 1. Remove a comparator from a random layer
        if move_type < 0.4 and any(layer for layer in current_layers):
            # Pick a random non-empty layer
            non_empty_layers = [i for i, l in enumerate(current_layers) if l]
            if not non_empty_layers:
                continue
            l_idx = rng.choice(non_empty_layers)
            c_idx = rng.integers(0, len(current_layers[l_idx]))
            
            # Remove
            removed_comp = current_layers[l_idx].pop(c_idx)
            if not current_layers[l_idx]:
                current_layers.pop(l_idx)
            
            candidate_comps = _layers_to_flat(current_layers)
            
            if _is_sorted_network_vectorized(candidate_comps, bits):
                # Accept
                best_comps = candidate_comps
                best_len = len(candidate_comps)
            else:
                # Revert
                # Re-insert at same position? Or just append to that layer?
                # Since order in layer doesn't matter, we can just append.
                # But we need to be careful if we popped the last element and removed the layer.
                if l_idx < len(current_layers):
                    current_layers[l_idx].append(removed_comp)
                else:
                    current_layers.append([removed_comp])
                    
        # 2. Insert a comparator into a random layer
        elif move_type < 0.7:
            # Pick a random layer (or create new)
            if current_layers and rng.random() < 0.8:
                l_idx = rng.integers(0, len(current_layers))
            else:
                l_idx = len(current_layers)
                current_layers.append([])
                
            # Pick a random comparator
            c = all_comps[rng.integers(0, len(all_comps))]
            
            # Check if it fits in the layer
            fits = True
            for p in current_layers[l_idx]:
                if c[0] in p or c[1] in p:
                    fits = False
                    break
            
            if fits:
                current_layers[l_idx].append(c)
                candidate_comps = _layers_to_flat(current_layers)
                
                # Only accept if size doesn't increase too much? 
                # Or always accept if valid? 
                # Insertion increases size. We accept it to explore, but we prefer removals.
                # To prevent drift, we only accept if it's valid.
                if _is_sorted_network_vectorized(candidate_comps, bits):
                    # We don't update best_len here because it's worse.
                    # But we keep it as current state to allow future removals.
                    pass
                else:
                    current_layers[l_idx].pop()
                    if not current_layers[l_idx] and l_idx == len(current_layers) - 1:
                        current_layers.pop()
                        
        # 3. Move a comparator from one layer to another
        elif move_type < 0.9:
            # Pick a comparator from a random layer
            non_empty_layers = [i for i, l in enumerate(current_layers) if l]
            if not non_empty_layers:
                continue
            src_l_idx = rng.choice(non_empty_layers)
            c_idx = rng.integers(0, len(current_layers[src_l_idx]))
            comp = current_layers[src_l_idx][c_idx]
            
            # Pick a destination layer
            dst_l_idx = rng.integers(0, len(current_layers) + 1)
            
            # Temporarily remove from source
            current_layers[src_l_idx].pop(c_idx)
            if not current_layers[src_l_idx]:
                current_layers.pop(src_l_idx)
                # Adjust indices if dst was after src
                if dst_l_idx > src_l_idx:
                    dst_l_idx -= 1
            
            # Try to insert into destination
            # If dst_l_idx is len(current_layers), it's a new layer
            if dst_l_idx == len(current_layers):
                current_layers.append([comp])
                fits = True
            else:
                # Check if it fits
                fits = True
                for p in current_layers[dst_l_idx]:
                    if comp[0] in p or comp[1] in p:
                        fits = False
                        break
                if fits:
                    current_layers[dst_l_idx].append(comp)
                else:
                    # Revert
                    # Re-insert into source (at end, order doesn't matter)
                    if src_l_idx < len(current_layers):
                        current_layers[src_l_idx].append(comp)
                    else:
                        current_layers.append([comp])
                    continue
            
            if fits:
                candidate_comps = _layers_to_flat(current_layers)
                if _is_sorted_network_vectorized(candidate_comps, bits):
                    # Accept move (size unchanged, structure changed)
                    pass
                else:
                    # Revert
                    # Remove from dst
                    current_layers[dst_l_idx].pop()
                    if not current_layers[dst_l_idx] and dst_l_idx == len(current_layers) - 1:
                        current_layers.pop()
                    # Re-insert into src
                    if src_l_idx < len(current_layers):
                        current_layers[src_l_idx].append(comp)
                    else:
                        current_layers.append([comp])
                        
        else:
            # 4. Swap two comparators in the same layer? 
            # This doesn't change the network function. 
            # Instead, let's try a "re-shuffle" of a layer.
            # Pick a layer and randomly re-assign its comparators to other layers if they fit?
            # This is complex. Let's just do a random valid replacement in a layer.
            # Replace a comparator in a layer with another that fits.
            non_empty_layers = [i for i, l in enumerate(current_layers) if l]
            if not non_empty_layers:
                continue
            l_idx = rng.choice(non_empty_layers)
            c_idx = rng.integers(0, len(current_layers[l_idx]))
            old_comp = current_layers[l_idx][c_idx]
            
            # Pick a new comp
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            
            # Check if new_comp fits in the layer (excluding old_comp)
            fits = True
            for p in current_layers[l_idx]:
                if p == old_comp:
                    continue
                if new_comp[0] in p or new_comp[1] in p:
                    fits = False
                    break
            
            if fits and new_comp != old_comp:
                current_layers[l_idx][c_idx] = new_comp
                candidate_comps = _layers_to_flat(current_layers)
                
                if _is_sorted_network_vectorized(candidate_comps, bits):
                    # Accept
                    pass
                else:
                    current_layers[l_idx][c_idx] = old_comp

    return best_comps

# EVOLVE-BLOCK-END
