import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # 1. Generate Pool of Weight-4 Vectors
    # Norm squared = 4. Norm = 2.
    # Constraint: dist^2 >= 4 => 4+4 - 2*u.v >= 4 => u.v <= 2.
    # Conflict if u.v > 2.
    
    candidates = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    if time.time() > deadline - 10:
        return candidates[:10]

    # 2. Compute Conflict Matrix
    # We need to know which pairs conflict.
    # u.v > 2 means they conflict.
    dots = candidates @ candidates.T
    # conflict[i, j] is True if i and j cannot be in the same set
    conflict = (dots > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Precompute neighbor lists for faster blocking/unblocking?
    # Or just use the matrix directly with bitsets.
    # Since n_cands ~ 5280, a 5280x5280 uint8 matrix is ~28MB. This is fine.
    
    # 3. Local Search Strategy
    # State: A boolean array `selected` of length n_cands.
    # Validity: No two selected vertices have conflict[i,j] == 1.
    
    def is_valid(selected):
        # Check if any pair in selected conflicts
        # This is O(k^2) where k is size of selected. 
        # Faster: Check if any selected vertex has a conflict with another selected vertex.
        # sum(conflict[selected, :].sum(axis=1)) == 0
        # Actually, just check: for any i in selected, does it conflict with any j in selected?
        # We can do: selected_counts = conflict @ selected
        # If any selected_counts[i] > 0 for i in selected, it's invalid.
        # Wait, conflict[i,i]=0. So if selected[i]=1, we need sum_{j in selected, j!=i} conflict[i,j] == 0.
        # Let S be the set of selected indices.
        # We want to ensure no edges within S.
        # This is equivalent to: for all i in S, (conflict[i, :] @ selected) == 0.
        # Vectorized: 
        # conflicts_with_selected = conflict @ selected
        # return np.all(conflicts_with_selected[selected] == 0)
        
        # Even faster check for validity of a *change*:
        # When adding i: check if any j in S has conflict[i,j]=1.
        # When removing i: always valid if it was valid before? No, removing never creates conflicts.
        # When swapping i (in) and j (out): 
        #   Remove j: safe.
        #   Add i: check if i conflicts with any k in S\{j}.
        
        pass # We will implement incremental checks in the loop.

    def get_size(selected):
        return np.sum(selected)

    # Initialize with a Greedy solution to provide a good starting point
    # Greedy: Sort by degree (ascending) and pick if no conflict.
    degrees = np.sum(conflict, axis=1)
    order = np.argsort(degrees)
    
    selected = np.zeros(n_cands, dtype=bool)
    current_size = 0
    
    for idx in order:
        # Check if idx conflicts with any currently selected
        # conflict[idx, :] @ selected
        # If 0, we can add it.
        if np.dot(conflict[idx, :], selected) == 0:
            selected[idx] = True
            current_size += 1
            
    best_selected = selected.copy()
    best_size = current_size
    
    # Local Search: Simulated Annealing style
    # Moves:
    # 1. Add a random unselected vertex if valid.
    # 2. Remove a random selected vertex (to allow swaps later).
    # 3. Swap: Remove one selected, add one unselected.
    
    # To make it efficient, we maintain a list of "candidates for addition" that don't conflict with current set?
    # That's expensive to maintain.
    
    # Let's use a simple hill climbing with random restarts and swaps.
    # We will track the "blocked" status implicitly or explicitly.
    
    # Optimization: 
    # Instead of checking full dot product every time (O(N)), we can precompute neighbor sets?
    # N=5280 is small enough that O(N) dot product is ~5k ops. 
    # 100,000 steps * 5k ops = 500M ops. Might be tight in Python/NumPy overhead.
    # Let's try to optimize the validity check.
    
    # Alternative: Use bitsets? Python integers are arbitrary precision.
    # Convert conflict rows to bitsets?
    # n_cands = 5280. Bitset is 5280 bits ~ 660 bytes.
    # 5280 * 660 bytes ~ 3.5 MB.
    # Operations on Python ints are very fast (C level).
    
    # Let's try the Bitset approach for speed.
    
    # Convert conflict matrix to list of Python integers (bitsets)
    # conflict_bits[i] is an int where bit j is 1 if i conflicts with j.
    # Note: Python ints are signed, but we can use them as unsigned bitsets.
    
    # Creating bitsets:
    # This might take a bit of time but should be okay.
    # conflict is uint8. 
    # We can use np.packbits? No, standard int conversion is easier if we treat rows as bytes?
    # Actually, converting 5280 rows of 5280 bits to Python ints manually might be slow.
    # Let's stick to NumPy dot products but optimize the loop structure.
    
    # Hybrid: 
    # Maintain `selected` as a boolean array.
    # Maintain `available` as a boolean array (vertices that can be added without conflict).
    # When we add i:
    #   available[conflict[i, :] == 1] = False
    #   available[i] = False
    # When we remove i:
    #   We need to re-evaluate available for its neighbors?
    #   If we remove i, vertices that were blocked *only* by i might become available.
    #   This is hard to track incrementally without reference counting.
    
    # Let's use Reference Counting for "blocked" status.
    # `block_count[j]` = number of selected vertices that conflict with j.
    # j is available if block_count[j] == 0 and not selected[j].
    
    block_count = np.zeros(n_cands, dtype=np.int32)
    
    # Initialize block_count from the greedy solution
    # For each selected i, add 1 to block_count of its neighbors.
    # This is O(N * avg_degree).
    # Or simply: block_count = conflict @ selected
    block_count = (conflict @ selected).astype(np.int32)
    
    # Now, local search loop.
    # We will try to improve `current_size`.
    
    # Precompute a list of indices for random sampling
    all_indices = np.arange(n_cands)
    
    # We will run for a fixed number of steps or until time limit.
    # Steps:
    # 1. Try to add a random available vertex.
    # 2. If no available, try to swap: remove a random selected, then try to add a random previously blocked vertex.
    
    # To allow "adding", we need a list of available vertices.
    # Maintaining a dynamic list of available vertices in Python is slow.
    # We can just sample random vertices and check if they are available.
    
    # Let's run a loop.
    steps = 0
    max_steps = 200000 # Arbitrary, will be cut by time
    
    while time.time() < deadline - 5:
        steps += 1
        if steps % 1000 == 0:
            if time.time() > deadline - 5:
                break
            
        # Strategy: 
        # 1. With prob 0.5, try to add.
        # 2. With prob 0.5, try to swap.
        
        if random.random() < 0.5:
            # Try to add
            # Sample a random vertex
            idx = np.random.randint(n_cands)
            if not selected[idx] and block_count[idx] == 0:
                # Add it
                selected[idx] = True
                current_size += 1
                # Update block counts
                block_count[conflict[idx, :] == 1] += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_selected = selected.copy()
            else:
                # Failed to add. Maybe try a swap?
                pass
        else:
            # Try to swap
            # Pick a random selected vertex to remove
            # We need to pick from selected. 
            # Sampling from a boolean array efficiently:
            # We can keep a list of selected indices? 
            # Or just sample random and check if selected.
            # Given density, this is fine.
            
            for _ in range(10): # Try a few times to find a selected vertex
                idx_out = np.random.randint(n_cands)
                if selected[idx_out]:
                    break
            else:
                continue # No selected vertex found? Unlikely unless size 0.
            
            # Remove idx_out
            selected[idx_out] = False
            current_size -= 1
            # Update block counts (decrement)
            block_count[conflict[idx_out, :] == 1] -= 1
            
            # Now try to add a vertex that was previously blocked or just any available.
            # Ideally, we want to add a vertex that conflicts with idx_out, 
            # because removing idx_out might have unblocked it.
            # Let's pick a random neighbor of idx_out first.
            # Neighbors are where conflict[idx_out, :] == 1.
            
            # Get list of neighbors? 
            # conflict[idx_out, :] is a vector.
            # We can sample from it.
            
            added = False
            # Try a few random neighbors
            for _ in range(20):
                # Sample a random index
                idx_in = np.random.randint(n_cands)
                # Check if it was a neighbor (conflict) and is now available
                # Actually, we just need it to be available (block_count == 0) and not selected.
                # It doesn't *have* to be a neighbor, but it's a good heuristic.
                if not selected[idx_in] and block_count[idx_in] == 0:
                    # Add it
                    selected[idx_in] = True
                    current_size += 1
                    block_count[conflict[idx_in, :] == 1] += 1
                    added = True
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = selected.copy()
                    break
            
            if not added:
                # Failed to add. Restore idx_out? 
                # Or just leave it removed? 
                # Leaving it removed reduces size, which is bad.
                # Better to revert the removal if we can't add.
                selected[idx_out] = True
                current_size += 1
                block_count[conflict[idx_out, :] == 1] += 1

    # Return the best set found
    if best_size == 0:
        return candidates[:1]
        
    result_vectors = candidates[best_selected]
    return result_vectors

# EVOLVE-BLOCK-END
