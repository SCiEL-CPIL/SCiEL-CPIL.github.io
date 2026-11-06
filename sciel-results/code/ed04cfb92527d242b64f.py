"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import numpy as np

N = 16
NUM_INPUTS = 1 << N
FULL_MASK = (1 << NUM_INPUTS) - 1

# EVOLVE-BLOCK-START
def _build_initial_wires():
    """Build the 16 wire bitsets representing all 2^16 binary inputs."""
    wires = []
    for i in range(N):
        val = 0
        block = 1 << i
        for start in range(0, NUM_INPUTS, 2 * block):
            mask = ((1 << block) - 1) << (start + block)
            val |= mask
        wires.append(val)
    return wires

def _verify_bitparallel(comparators):
    """Verify sorting network using bit-parallel simulation over all 2^16 inputs."""
    wires = _build_initial_wires()
    for i, j in comparators:
        swap_mask = wires[i] & ~wires[j] & FULL_MASK
        if swap_mask:
            wires[i] = wires[i] & ~swap_mask
            wires[j] = wires[j] | swap_mask
    # Check sorted: for each adjacent pair, wire[i] <= wire[i+1] bitwise
    for i in range(N - 1):
        if wires[i] & ~wires[i + 1] & FULL_MASK:
            return False
    return True

def _verify_from_state(comparators, state_wires):
    """Verify starting from a given wire state (for incremental checking)."""
    wires = list(state_wires)
    for i, j in comparators:
        swap_mask = wires[i] & ~wires[j] & FULL_MASK
        if swap_mask:
            wires[i] = wires[i] & ~swap_mask
            wires[j] = wires[j] | swap_mask
    for i in range(N - 1):
        if wires[i] & ~wires[i + 1] & FULL_MASK:
            return False
    return True

def construct():
    """
    Construct a size-efficient sorting network for 16 wires using
    bit-parallel zero-one principle verification and greedy pruning.
    """
    # Start with odd-even transposition network (120 comparators)
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Greedy removal with multiple passes and randomization
    for _ in range(20):
        removed_any = False
        indices = list(range(len(network)))
        random.shuffle(indices)
        for idx in indices:
            if idx >= len(network):
                continue
            candidate = network[:idx] + network[idx + 1:]
            if _verify_bitparallel(candidate):
                network = candidate
                removed_any = True
        if not removed_any:
            break
    
    # Try moving comparators to different positions to enable further pruning
    for _ in range(5):
        moved_any = False
        indices = list(range(len(network)))
        random.shuffle(indices)
        for idx in indices:
            if idx >= len(network):
                continue
            comp = network[idx]
            # Try inserting this comparator at various earlier positions
            for new_idx in range(idx):
                candidate = network[:new_idx] + network[new_idx:idx] + [comp] + network[idx + 1:]
                if _verify_bitparallel(candidate):
                    network = candidate
                    moved_any = True
                    break
        # After moving, try more removals
        for _ in range(5):
            removed_any = False
            indices = list(range(len(network)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(network):
                    continue
                candidate = network[:idx] + network[idx + 1:]
                if _verify_bitparallel(candidate):
                    network = candidate
                    removed_any = True
            if not removed_any:
                break
        if not moved_any:
            break
    
    return network
# EVOLVE-BLOCK-END
