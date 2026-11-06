"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE = 55.0  # seconds, safety margin before 60s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires using Bit-Parallel verification.
    Uses Python big integers to represent the state of each wire across all 2^18 inputs.
    min(a,b) is bitwise AND, max(a,b) is bitwise OR.
    Sorted check: (wire[i] & ~wire[i+1]) == 0 for all i.
    """
    start_time = time.time()

    # --- Initialize Bit-Parallel Wires ---
    # wire[i] is a Python int where bit k is 1 if input k has 1 on wire i.
    # Input k is the integer k. Wire i corresponds to bit i of k.
    # So wire[i] has bits set at positions k where (k >> i) & 1 == 1.
    # This is equivalent to: for all k, bit k of wire[i] is bit i of k.
    
    # We can construct these masks efficiently.
    # Mask for bit i: pattern of 1s and 0s.
    # For N=18, we need masks for 2^18 bits.
    # A mask for bit i is a number with 1s at positions k where k has bit i set.
    # This is a repeating pattern of 2^i zeros, 2^i ones.
    
    def make_mask(i):
        # Create a big int with 1s at positions where bit i is set.
        # Total bits needed: 2^N.
        # Pattern: 0...0 1...1 0...0 1...1 ...
        # Block size: 2^(i+1). First half 0, second half 1.
        # We can build this by shifting and ORing.
        # Or simpler: 
        # m = 0
        # for k in range(TOTAL_INPUTS):
        #    if (k >> i) & 1: m |= (1 << k)
        # This loop is 262k * 18 = 4.7M iterations. Might be slow in Python?
        # Let's try to generate it faster.
        
        # Pattern: 2^i zeros, 2^i ones.
        # Repeat this pattern 2^(N-i-1) times.
        # One block: (2^i ones) << (2^i zeros)? No.
        # Value of one block (2^(i+1) bits): (2^i - 1) << 2^i.
        # Block = ((1 << (2**i)) - 1) << (2**i)
        
        block_len = 1 << (i + 1)
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        
        # How many blocks?
        num_blocks = TOTAL_INPUTS // block_len
        
        mask = 0
        # We need to repeat this block.
        # This is effectively a geometric series of blocks shifted by block_len.
        # mask = block * (1 + 2^block_len + 2^(2*block_len) + ...)
        # Let's just loop and shift, it's only 2^(N-i-1) iterations.
        # For i=0, 2^17 iterations. Too slow.
        
        # Alternative:
        # Use the property that the mask for bit i is:
        # sum_{j=0}^{2^(N-i-1)-1} (2^(j * 2^(i+1) + 2^i) ... 2^(j * 2^(i+1) + 2^(i+1) - 1))
        
        # Actually, there's a simpler way using Python's string or just bit manipulation if we can.
        # Let's just use the loop but optimize it or use NumPy to create the mask once.
        # NumPy is available and fast for creating large arrays.
        import numpy as np
        # Create an array of 0s and 1s
        # indices = np.arange(TOTAL_INPUTS)
        # bits = (indices >> i) & 1
        # mask = sum(bit * (1 << k) for k, bit in enumerate(bits)) -> Too slow.
        
        # We can create the mask by packing bits.
        # But we need it as a Python Int.
        # Let's use a different approach: 
        # We don't actually need to pre-compute the mask as a single big int if we can verify differently?
        # No, the bit-parallel approach relies on it.
        
        # Let's try to generate the mask using a recursive/doubling method or just accept the cost if it's one-time.
        # 2^18 = 262144.
        # Creating 18 masks.
        # If we use a loop of 262k for each mask, that's 4.7M operations. In Python, this might take ~0.5s - 1s. Acceptable.
        
        mask = 0
        # Optimization: We can build the mask by ORing shifted blocks.
        # But the number of blocks for i=0 is 131072.
        # Let's just use the simple loop for clarity and speed check.
        # Actually, we can use:
        # mask = 0
        # for k in range(TOTAL_INPUTS):
        #     if (k >> i) & 1:
        #         mask |= (1 << k)
        
        # Let's try a faster generation:
        # The mask is periodic.
        # We can build it by taking the base block and multiplying by a geometric series factor.
        # Geometric series factor G = 1 + 2^L + 2^(2L) + ... where L = block_len.
        # G = (2^(L * num_blocks) - 1) / (2^L - 1)
        # This involves huge numbers, but Python handles it.
        
        L = 1 << (i + 1)
        num_blocks = TOTAL_INPUTS // L
        # G = (1 << (L * num_blocks)) - 1) // ((1 << L) - 1)
        # L * num_blocks = TOTAL_INPUTS.
        # So G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        
        # This creates a huge number (2^262144 bits), which is ~32KB. This is fine.
        # block is ~2^(2^i) bits.
        
        # mask = block * G
        # Is this correct?
        # block has 1s in the second half of the L-bit block.
        # Multiplying by G shifts block by 0, L, 2L, etc.
        # Yes, this works perfectly.
        
        G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        mask = block * G
        return mask

    # Pre-compute masks
    masks = [make_mask(i) for i in range(N)]

    def verify_network(network):
        """Verify network sorts all inputs. Returns True if correct."""
        wires = masks[:]
        
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
            
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
        return True

    def find_violation(network):
        """
        Run the network and return the first (i, j) pair where wire[i] > wire[j]
        (i.e., wire[i] has a 1 where wire[j] has a 0), or None if sorted.
        This identifies the specific out-of-order wire pair to guide repair.
        """
        wires = masks[:]
        
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
            
        for i in range(N - 1):
            violation = wires[i] & ~wires[i+1]
            if violation != 0:
                return (i, i + 1)
        return None

    # --- Start with odd-even transposition network ---
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    # Verify initial network
    assert verify_network(network), "Initial network should be correct"

    best_network = network[:]
    best_size = len(network)

    # --- Phase 1: Greedy removal ---
    # With fast verification, we can do more passes.
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 10:
        removed = False
        for idx in range(len(network)):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Try removing idx
            candidate = network[:]
            candidate.pop(idx)
            if verify_network(candidate):
                network = candidate
                if len(network) < best_size:
                    best_network = network[:]
                    best_size = len(network)
                removed = True
                break  # Restart since indices changed

    # --- Phase 2: Error-guided local search ---
    # Remove comparators; when broken, use find_violation to guide targeted repair.
    current = best_network[:]
    for iteration in range(10000):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
            
        # Random removal
        idx = random.randint(0, len(current) - 1)
        candidate = current[:]
        candidate.pop(idx)
        
        # If it's still valid, accept
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
            continue
            
        # If broken, find the specific violation and repair it
        violation = find_violation(candidate)
        if violation is None:
            # Shouldn't happen since verify_network returned False, but safety
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
            continue
        
        vi, vj = violation  # The out-of-order adjacent wire pair
        
        repaired = False
        # Strategy 1: Insert comparator (vi, vj) at various positions near the removal
        # The violation is between wires vi and vj, so adding (vi, vj) should help
        for pos in range(max(0, idx - 3), min(len(candidate) + 1, idx + 4)):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            temp = candidate[:]
            temp.insert(pos, (vi, vj))
            if verify_network(temp):
                current = temp
                repaired = True
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
                break
        
        # Strategy 2: If that didn't work, try other comparators involving vi or vj
        if not repaired:
            for _ in range(15):
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                # Pick a wire near the violation
                if random.random() < 0.5:
                    i = max(0, vi - 1) if vi > 0 else vi
                    j = random.randint(i + 1, N - 1)
                else:
                    j = min(N - 1, vj + 1) if vj < N - 1 else vj
                    i = random.randint(0, j - 1)
                
                pos = random.randint(0, len(candidate))
                temp = candidate[:]
                temp.insert(pos, (i, j))
                if verify_network(temp):
                    current = temp
                    repaired = True
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
                    break
        
        # Strategy 3: Random fallback
        if not repaired:
            for _ in range(5):
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                pos = random.randint(0, len(candidate))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                temp = candidate[:]
                temp.insert(pos, (i, j))
                if verify_network(temp):
                    current = temp
                    repaired = True
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
                    break

    # --- Phase 3: Swap-based local search + Final Greedy Removal ---
    # Try swapping adjacent comparators to find better orderings, then final removal.
    final_net = best_network[:]
    
    # Swap-based local search: try swapping pairs of comparators
    swap_improved = True
    while swap_improved and (time.time() - start_time) < DEADLINE - 4:
        swap_improved = False
        for idx in range(len(final_net) - 1):
            if (time.time() - start_time) > DEADLINE - 3:
                break
            candidate = final_net[:]
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            if verify_network(candidate):
                final_net = candidate
                swap_improved = True
                break  # Restart after a successful swap
    
    # Final greedy removal
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 2:
        removed = False
        for idx in range(len(final_net)):
            if (time.time() - start_time) > DEADLINE - 1:
                break
            candidate = final_net[:]
            candidate.pop(idx)
            if verify_network(candidate):
                final_net = candidate
                removed = True
                break

    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    return best_network
# EVOLVE-BLOCK-END
