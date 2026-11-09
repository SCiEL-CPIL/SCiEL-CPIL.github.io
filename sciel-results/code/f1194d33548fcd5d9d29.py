import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N
BLOCK_SIZE = 64
NUM_BLOCKS = NUM_INPUTS // BLOCK_SIZE

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Pre-compute target states for all 2^18 inputs
    # Target for input k: wire i is 1 if i >= N - popcount(k)
    indices = np.arange(NUM_INPUTS, dtype=np.uint32).reshape(NUM_BLOCKS, BLOCK_SIZE)
    
    # Compute bits and popcounts
    bits = (indices[:, :, None] >> np.arange(N)[None, None, :]) & 1
    popcounts = bits.sum(axis=2)
    
    # Helper to pack 2D bool array (B, 64) into 1D uint64 array (B,)
    def pack_bits_to_u64(arr_2d):
        flat = arr_2d.flatten()
        packed = np.packbits(flat, axis=0, bitorder='little')
        packed = packed.reshape(NUM_BLOCKS, 8)
        return packed.view(np.uint64).flatten()
    
    # Compute targets for each wire
    targets = []
    for i in range(N):
        mask = (popcounts >= (N - i))
        targets.append(pack_bits_to_u64(mask))
    
    # Initialize wires (identity permutation)
    wires = []
    for i in range(N):
        wires.append(pack_bits_to_u64(bits[:, :, i]))
    
    # Function to apply comparator
    def apply_comp(wires, i, j):
        new_wires = wires.copy()
        a = new_wires[i]
        b = new_wires[j]
        new_a = a & b
        new_b = a | b
        new_wires[i] = new_a
        new_wires[j] = new_b
        return new_wires
    
    # Function to calculate error (Hamming distance to target)
    # Optimized: unpack uint64 to uint8, then count bits
    def calc_error(wires):
        err = 0
        for i in range(N):
            diff = wires[i] ^ targets[i]
            # View uint64 as uint8 (8 bytes)
            u8 = diff.view(np.uint8).reshape(NUM_BLOCKS, 8)
            # Unpack bits: each byte becomes 8 bits
            unpacked = np.unpackbits(u8, axis=1, count=8, bitorder='little')
            err += unpacked.sum()
        return err

    # Greedy search to find a good starting point
    current_wires = wires
    current_error = calc_error(current_wires)
    network = []
    
    # Greedy phase
    while time.time() < deadline - 15.0:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        
        for i in range(N):
            for j in range(i + 1, N):
                new_wires = apply_comp(current_wires, i, j)
                new_error = calc_error(new_wires)
                if new_error < best_error:
                    best_error = new_error
                    best_comp = (i, j)
                    best_new_wires = new_wires
                    
        if best_comp is None:
            break
            
        current_wires = best_new_wires
        current_error = best_error
        network.append(best_comp)
        
        if current_error == 0:
            break
            
    # Pruning phase: Remove comparators that don't help
    def prune_network(net, w):
        # net is a list of (i,j)
        # w is the final wires state after applying net
        # We want to check if removing any comparator increases error.
        # This is expensive to do fully for every comp in every step.
        # Simple heuristic: Iterate through comps, if removing it doesn't change final error, remove it.
        # But "final error" depends on the whole network.
        # Let's just try to remove comps one by one from the end, or random order.
        
        # To be efficient, we'll do a few passes of random removal attempts.
        # If time permits, we can do a more thorough pruning.
        pass # We will implement a simple prune in the main loop logic instead of a separate heavy function to keep it simple and fast per step.

    # New Search Strategy: Iterated Local Search (ILS)
    # 1. Greedy Build
    # 2. Prune (remove useless comps)
    # 3. Mutate
    # 4. Greedy Build (continue)
    # 5. Prune
    # Repeat.

    best_network = network.copy()
    best_error = current_error
    
    # Helper to get error of a network starting from identity
    def get_network_error(net):
        w = wires.copy()
        for i, j in net:
            w = apply_comp(w, i, j)
        return calc_error(w), w

    # Initial Prune of the greedy result
    # We check if removing any comparator from the end reduces or maintains error.
    # Actually, removing a comparator *increases* error or keeps it same.
    # We want to remove if error stays same.
    def prune(net):
        # Try to remove comparators in reverse order
        # If removing net[k] results in same error, remove it.
        # This is O(N^2 * Len) which is expensive.
        # Optimization: Only try to remove if the comparator is "redundant".
        # A simpler prune: Randomly shuffle and try to remove.
        
        # Let's do a few passes of random removal
        for _ in range(3): # 3 passes
            removed = True
            while removed and time.time() < deadline - 10.0:
                removed = False
                # Pick a random index
                if not net: break
                idx = np.random.randint(0, len(net))
                # Get error without this comparator
                test_net = net[:idx] + net[idx+1:]
                err_test, _ = get_network_error(test_net)
                err_full, _ = get_network_error(net) # This is expensive!
                
                # Better: Calculate error of current net once, then test removal.
                # But we need to be careful with time.
                # Let's just skip complex pruning for now and rely on the "Greedy from mutated state" to naturally find smaller networks if we start from smaller ones?
                # No, greedy *adds* comps.
                # We need a "Reverse Greedy" or "Prune" step.
                
                # Let's implement a fast prune:
                # Check if the comparator is a no-op on the *current* wires before it?
                # If wire i == wire j before the comparator, it's useless.
                # This is a valid local check.
                
                # Let's try this local check:
                # For each comp in net, check if wires[i] == wires[j] before applying it.
                # If yes, remove it.
                w = wires.copy()
                new_net = []
                for i, j in net:
                    if w[i] != w[j]:
                        new_net.append((i, j))
                        w = apply_comp(w, i, j)
                    else:
                        # It's a no-op, skip adding it
                        pass
                if len(new_net) < len(net):
                    net = new_net
                    # Recalculate error to be sure
                    err, _ = get_network_error(net)
                    return err
                return None # No change

    # Apply local prune to initial network
    err_pruned = prune(best_network)
    if err_pruned is not None:
        best_error = err_pruned

    # ILS Loop
    while time.time() < deadline - 5.0:
        # Mutate best_network
        current_net = best_network.copy()
        
        # Mutation strategies
        r = np.random.rand()
        if r < 0.3:
            # Insert random comparator
            pos = np.random.randint(0, len(current_net) + 1)
            i = np.random.randint(0, N)
            j = np.random.randint(i + 1, N)
            current_net.insert(pos, (i, j))
        elif r < 0.6:
            # Delete random comparator
            if current_net:
                pos = np.random.randint(0, len(current_net))
                current_net.pop(pos)
        elif r < 0.8:
            # Swap two comparators
            if len(current_net) >= 2:
                i1 = np.random.randint(0, len(current_net))
                i2 = np.random.randint(0, len(current_net))
                current_net[i1], current_net[i2] = current_net[i2], current_net[i1]
        else:
            # Swap wires (permutation of indices in comps)
            # This is complex, skip for now.
            pass
            
        # Greedy Build (Add comparators to reduce error)
        # Start from the state after current_net
        w = wires.copy()
        for i, j in current_net:
            w = apply_comp(w, i, j)
        
        err = calc_error(w)
        
        # Try to add comparators
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_err = err
            best_comp = None
            best_new_wires = None
            
            for i in range(N):
                for j in range(i + 1, N):
                    new_wires = apply_comp(w, i, j)
                    new_err = calc_error(new_wires)
                    if new_err < best_err:
                        best_err = new_err
                        best_comp = (i, j)
                        best_new_wires = new_wires
            
            if best_comp is not None:
                w = best_new_wires
                err = best_err
                current_net.append(best_comp)
                improved = True
                
        # Prune the resulting network
        # Use the local "no-op" prune
        w_check = wires.copy()
        pruned_net = []
        for i, j in current_net:
            if w_check[i] != w_check[j]:
                pruned_net.append((i, j))
                w_check = apply_comp(w_check, i, j)
        
        # Check error of pruned net
        if len(pruned_net) < len(current_net):
            final_err, _ = get_network_error(pruned_net)
            if final_err <= best_error:
                best_network = pruned_net
                best_error = final_err
        
    return best_network
