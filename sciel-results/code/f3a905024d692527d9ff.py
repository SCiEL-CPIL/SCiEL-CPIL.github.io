import numpy as np
import time

N = 17

# Cache for input matrix to avoid regeneration
_INPUTS_CACHE = None

def _get_inputs():
    global _INPUTS_CACHE
    if _INPUTS_CACHE is None:
        _INPUTS_CACHE = np.zeros((1 << N, N), dtype=np.uint8)
        for k in range(N):
            _INPUTS_CACHE[:, k] = (np.arange(1 << N) >> k) & 1
    return _INPUTS_CACHE

def is_valid_fast(network):
    if not network:
        return True
    inputs = _get_inputs().copy() # Must copy as we modify
    for i, j in network:
        # min to i, max to j
        # If inputs[:, i] == 1 and inputs[:, j] == 0, swap
        mask_swap = (inputs[:, i] == 1) & (inputs[:, j] == 0)
        if np.any(mask_swap):
            inputs[mask_swap, i] = 0
            inputs[mask_swap, j] = 1
    diffs = np.diff(inputs, axis=1)
    return not np.any(diffs < 0)

# EVOLVE-BLOCK-START
def construct():
    # Seed: Odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    best_network = network
    best_size = len(network)
    current_network = network
    
    # Phase 1: Greedy Removal
    # We iterate passes until no improvement or timeout
    while time.time() - start_time < timeout:
        improved = False
        # Try removing each comparator in reverse order
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() - start_time > timeout:
                break
            
            candidate = current_network[:idx] + current_network[idx+1:]
            
            if is_valid_fast(candidate):
                current_network = candidate
                improved = True
                if len(current_network) < best_size:
                    best_size = len(current_network)
                    best_network = current_network
                # Restart the pass to ensure we check the new list structure
                break
        
        if not improved:
            # No single removal works.
            # Try Phase 2: Random Swaps + Removal?
            # If we have time, try a few random swaps to escape local minima
            if time.time() - start_time < timeout - 5.0 and len(current_network) > 70:
                # Try 50 random swaps
                for _ in range(50):
                    if time.time() - start_time > timeout:
                        break
                    if len(current_network) < 2:
                        break
                    i, j = np.random.choice(len(current_network), 2, replace=False)
                    # Swap i and j
                    current_network[i], current_network[j] = current_network[j], current_network[i]
                    # Try to remove the swapped one? Or just verify?
                    # Just verify if the swapped network is valid (it should be if original was valid and we just swapped order? No, order matters in networks).
                    # Actually, swapping comparators in a sorting network does NOT necessarily preserve validity.
                    # So we must verify.
                    if is_valid_fast(current_network):
                        # Now try to remove a comparator from this valid swapped network
                        for idx2 in range(len(current_network) - 1, -1, -1):
                            candidate2 = current_network[:idx2] + current_network[idx2+1:]
                            if is_valid_fast(candidate2):
                                current_network = candidate2
                                if len(current_network) < best_size:
                                    best_size = len(current_network)
                                    best_network = current_network
                                break
                    # Restore swap if it didn't lead to improvement?
                    # Or just keep searching.
            else:
                break
                
    return best_network
# EVOLVE-BLOCK-END
