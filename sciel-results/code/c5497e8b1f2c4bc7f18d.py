"""
Size-efficient sorting networks on 18 wires: Local search with vectorized verification.

Strategy:
1. Start with a valid 153-comparator Odd-Even Transposition network.
2. Use vectorized NumPy checks to verify sorting correctness over all 2^18 Boolean inputs.
3. Perform local search:
   - Attempt to delete comparators.
   - Attempt to relocate comparators (move from index i to index j).
   - Accept changes that maintain validity and reduce size (or same size with better depth/ordering, though size is the primary metric).
4. Return the best network found before the time limit.
"""
import time
import numpy as np

N = 18
ALL_COUNT = 1 << N
DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

# Pre-compute all Boolean inputs as a (N, ALL_COUNT) array of uint8
# This allows vectorized simulation of the network.
# Input k is represented by its binary bits.
def _init_masks():
    idx = np.arange(ALL_COUNT, dtype=np.uint32)
    # We want wire w to contain the w-th bit of each input index.
    # Shape: (N, ALL_COUNT)
    masks = np.stack([(idx >> w) & 1 for w in range(N)], axis=0).astype(np.uint8)
    return masks

ALL_MASKS = _init_masks()

def _apply_net(net, masks):
    """
    Apply comparator network to all 2^N Boolean inputs.
    Returns array of shape (N, ALL_COUNT).
    """
    if not net:
        return masks.copy()
    vals = masks.copy()
    for a, b in net:
        va = vals[a]
        vb = vals[b]
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        vals[a] = lo
        vals[b] = hi
    return vals

def _is_sorted(vals):
    """
    Check if the output is non-decreasing for all inputs.
    vals shape: (N, ALL_COUNT)
    """
    return np.all(vals[1:] >= vals[:-1])

def _verify(net):
    """
    Verify if the network sorts all inputs.
    Returns True if valid.
    """
    vals = _apply_net(net, ALL_MASKS)
    return _is_sorted(vals)

def _generate_oet(n):
    """Generate Odd-Even Transposition network."""
    net = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    
    # Initial network: Odd-Even Transposition (153 comparators for N=18)
    current_net = _generate_oet(N)
    
    # Verify initial network (should be True)
    if not _verify(current_net):
        # Fallback if something went wrong, though OET is always valid
        return current_net

    best_net = list(current_net)
    best_len = len(best_net)
    
    # Local Search Parameters
    # We alternate between deletion and relocation attempts.
    # Relocation is crucial because simply deleting might get stuck in local minima
    # where no single deletion works, but moving a comparator makes a deletion possible.
    
    max_iterations = 100000
    no_improve_count = 0
    max_no_improve = 5000
    
    import random
    random.seed(42) # Deterministic for reproducibility within run
    
    for _ in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        if no_improve_count > max_no_improve:
            # Reset no_improve counter if we've been stuck for a while,
            # or break if we are very close to deadline.
            # To avoid infinite loops on bad local minima, we might just break or restart.
            # For now, let's just break to ensure we return a valid result.
            # A more advanced strategy would be to randomize the current state slightly.
            pass 

        # Choose a move type: 70% Deletion, 30% Relocation
        if len(current_net) <= 1:
            break
            
        if random.random() < 0.7:
            # Try Deletion
            idx = random.randrange(len(current_net))
            candidate = current_net[:idx] + current_net[idx+1:]
            
            if _verify(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = list(current_net)
                    best_len = len(best_net)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Try Relocation
            # Pick a random comparator to move
            src_idx = random.randrange(len(current_net))
            comp = current_net[src_idx]
            
            # Pick a random destination index
            # We remove from src_idx and insert at dst_idx.
            # dst_idx is in the range [0, len(current_net)-1]
            # Note: If dst_idx > src_idx, the index shifts.
            dst_idx = random.randrange(len(current_net))
            
            if src_idx == dst_idx:
                continue
            
            # Construct candidate
            # Remove comp
            temp_net = current_net[:src_idx] + current_net[src_idx+1:]
            
            # Insert comp at dst_idx
            # If dst_idx was after src_idx in the original list, 
            # the "hole" is at src_idx. 
            # If we insert at dst_idx in the *original* list context:
            # It's safer to think of it as:
            # 1. Remove item at src_idx.
            # 2. Insert item at dst_idx in the new list.
            # However, standard list splice semantics:
            # If we want to move element at i to j:
            # del net[i]
            # net.insert(j, item)
            
            # Let's just use list operations on a copy
            candidate = list(current_net)
            item = candidate.pop(src_idx)
            candidate.insert(dst_idx, item)
            
            if _verify(candidate):
                current_net = candidate
                # Relocation doesn't change length, so it doesn't immediately improve best_len
                # But it might enable future deletions.
                # We don't update best_net here unless we consider "ordering" as part of the score, 
                # but the score is strictly comparator count.
                # So we only update best if we delete.
                # However, keeping current_net updated is vital for subsequent deletions.
                no_improve_count += 1 # No size improvement
            else:
                no_improve_count += 1

    # Final verification of the best network found
    if _verify(best_net):
        return best_net
    
    # Fallback to initial valid network if best_net somehow became invalid (shouldn't happen)
    return _generate_oet(N)

if __name__ == "__main__":
    # Simple self-test
    net = construct()
    print(f"Construct finished with {len(net)} comparators.")
    print(f"Verification: {_verify(net)}")
