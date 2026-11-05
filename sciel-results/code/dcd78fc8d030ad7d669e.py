"""Size-efficient sorting networks on 17 wires - optimized with bit-packing."""
import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _make_bitpack_inputs():
    """
    Create bit-packed representation of all 2^N inputs.
    Returns an array of N uint64 arrays, each of length NUM_INPUTS//64.
    bit[k][b] has bit p set iff input (b*64+p) has wire k set.
    """
    nblocks = NUM_INPUTS // 64
    bits = np.arange(NUM_INPUTS, dtype=np.uint64)
    # Reshape to (nblocks, 64) so that column p of row b is input b*64+p
    bits_2d = bits.reshape(nblocks, 64)
    
    # For each wire k, we want an array where element b has bit p set
    # iff wire k of input b*64+p is set.
    # wire k of input x is (x >> k) & 1
    # For input b*64+p: ((b*64+p) >> k) & 1
    # = ((b << 6 | p) >> k) & 1
    # = (b << (6-k) | (p >> k)) & 1   when k <= 6
    # More generally: ((b*64 + p) >> k) & 1
    # Let's compute this directly using broadcasting.
    
    # Actually, let's use a simpler approach:
    # For input x, wire k is set iff (x >> k) & 1 == 1
    # We need: for each block b, for each position p (0..63),
    #   wire k of input (b*64+p) is set
    # This equals ((b*64+p) >> k) & 1
    
    # Let's precompute for each k:
    # mask_k[p] = 1 if (p >> k) & 1 else 0  -- but this isn't quite right
    # because (b*64+p) >> k depends on both b and p.
    
    # Better approach: just compute it directly.
    # For k in 0..6: (b*64+p) >> k = (b << (6-k)) | (p >> k), and & 1 gives
    #   ((b << (6-k)) | (p >> k)) & 1
    # For k >= 7: (b*64+p) >> k = (b >> (k-6)) | ((p >> k) ... ) -- more complex
    
    # Let's just do it the straightforward way with numpy:
    packed = np.empty((N, nblocks), dtype=np.uint64)
    for k in range(N):
        # For each input x = b*64+p, wire k = (x >> k) & 1
        # We want packed[k][b] to have bit p set iff wire k of input b*64+p is 1
        # 
        # (b*64+p) >> k & 1:
        # Let's compute this as a function of b and p.
        # b ranges 0..nblocks-1, p ranges 0..63
        # x = b*64 + p
        # (x >> k) & 1
        
        # For efficiency, note that for fixed k:
        # (b*64+p) >> k = (b*64 >> k) + (p >> k) when there's no carry... 
        # Actually (b*64+p) >> k is not simply separable.
        
        # Let's just compute it directly.
        # Create array of all x values for this k
        # x = b*64 + p, where b is 0..nblocks-1, p is 0..63
        # We can vectorize: for each p, compute (b*64+p) >> k & 1 for all b
        # Then pack into uint64.
        
        # Alternative: use the fact that we already have bits array
        # bits[x] = x, and we want for each k, for each block b:
        #   sum over p of (((b*64+p) >> k) & 1) * (1 << p)
        
        # Let's do it column by column (p by p)
        col_vals = np.zeros(nblocks, dtype=np.uint64)
        for p in range(64):
            # For all b, x = b*64 + p
            # wire k = ((b*64+p) >> k) & 1
            # = ((b << 6 + p) >> k) & 1
            # Let's compute:
            x_vals = np.arange(nblocks, dtype=np.uint64) * 64 + p
            wire_vals = ((x_vals >> k) & 1).astype(np.uint64)
            col_vals |= (wire_vals << p)
        packed[k] = col_vals
    
    return packed

def _verify_network(comparators, packed):
    """
    Verify network sorts all inputs using bit-packed representation.
    packed: (N, nblocks) uint64 array
    Returns True if network is valid.
    """
    nblocks = packed.shape[1]
    wires = packed.copy()
    
    for i, j in comparators:
        wi = wires[i]
        wj = wires[j]
        # min = AND, max = OR (for 0/1 values)
        wires[i] = wi & wj
        wires[j] = wi | wj
    
    # Check sorted: for each adjacent pair (k, k+1), no position has wire[k]=1 and wire[k+1]=0
    # i.e., wires[k] & ~wires[k+1] should be all zeros
    for k in range(N - 1):
        if np.any(wires[k] & ~wires[k + 1]):
            return False
    return True

def _depth_of_network(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    # depth[i] = depth at which wire i last had a comparator
    # Process comparators in order; a comparator (i,j) at position t
    # has depth = max(depth[i], depth[j]) + 1
    depth = [0] * N
    for i, j in comparators:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)

def construct():
    start_time = time.time()
    
    # Generate odd-even transposition network
    comps = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
    
    # Create bit-packed inputs
    packed = _make_bitpack_inputs()
    
    # Verify base is correct
    if not _verify_network(comps, packed):
        return comps
    
    # Phase 1: Greedy removal
    changed = True
    while changed and (time.time() - start_time) < DEADLINE - 3:
        changed = False
        for idx in range(len(comps)):
            if (time.time() - start_time) > DEADLINE - 3:
                break
            # Try removing comparator at idx
            new_comps = comps[:idx] + comps[idx+1:]
            if _verify_network(new_comps, packed):
                comps = new_comps
                changed = True
                break  # Restart scan
    
    # Phase 2: Local search with random swaps/moves
    # Try to improve by moving comparators or swapping them
    best_comps = list(comps)
    best_len = len(comps)
    
    # Precompute some random seeds for reproducibility
    rng = random.Random(42)
    
    # Try various local moves:
    # 1. Move a comparator to a different position
    # 2. Swap two comparators
    # 3. Try different orderings of independent comparators
    
    max_rounds = 50  # Number of local search rounds to attempt
    
    for rnd in range(max_rounds):
        if (time.time() - start_time) > DEADLINE - 2:
            break
        
        # Try several random perturbations
        for attempt in range(20):
            if (time.time() - start_time) > DEADLINE - 2:
                break
            
            candidate = list(best_comps)
            move_type = rng.randint(0, 2)
            
            if move_type == 0 and len(candidate) >= 2:
                # Swap two comparators
                i, j = rng.sample(range(len(candidate)), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
            elif move_type == 1 and len(candidate) >= 2:
                # Move a comparator to a new position
                i = rng.randrange(len(candidate))
                j = rng.randrange(len(candidate))
                if i != j:
                    comp = candidate.pop(i)
                    candidate.insert(j, comp)
            elif move_type == 2 and len(candidate) >= 2:
                # Reverse a segment
                i, j = rng.sample(range(len(candidate)), 2)
                if i > j:
                    i, j = j, i
                candidate[i:j+1] = candidate[i:j+1][::-1]
            
            if len(candidate) != best_len:
                continue
            
            if _verify_network(candidate, packed):
                # Check if it's actually different (avoid no-ops)
                if candidate != best_comps:
                    best_comps = candidate
                    # After finding a valid rearrangement, try to remove more
                    changed = True
                    while changed and (time.time() - start_time) < DEADLINE - 2:
                        changed = False
                        for idx in range(len(best_comps)):
                            if (time.time() - start_time) > DEADLINE - 2:
                                break
                            new_comps = best_comps[:idx] + best_comps[idx+1:]
                            if _verify_network(new_comps, packed):
                                best_comps = new_comps
                                changed = True
                                break
    
    return best_comps

# EVOLVE-BLOCK-END
