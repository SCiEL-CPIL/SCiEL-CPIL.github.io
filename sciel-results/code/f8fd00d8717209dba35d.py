"""Size-efficient sorting network search for 16 wires."""
import time
import numpy as np

N = 16
TOTAL_BITS = 1 << N  # 65536

# Precompute the sorted-order mask: after sorting, the top k bits are 1.
# For a given input vector x (as integer), sorted output has the top popcount(x) bits set.
# We represent each input as an N-bit integer.
# After the network, the output must have all 1s in the highest positions.

# Precompute: for each input mask (0..65535), what is the sorted mask?
# sorted_mask[x] has popcount(x) ones in the top bits.
def _build_sorted_masks():
    masks = np.zeros(TOTAL_BITS, dtype=np.uint64)
    for x in range(TOTAL_BITS):
        cnt = bin(x).count('1')
        # top cnt bits set
        m = ((1 << cnt) - 1) << (N - cnt) if cnt > 0 else 0
        masks[x] = m
    return masks

SORTED_MASKS = _build_sorted_masks()

def _apply_comparator_to_mask(x, i, j):
    """Apply compare-exchange (i,j) to a single N-bit integer x.
    min goes to i, max to j.
    Bit i and bit j: if bit i=0 and bit j=1, swap them (i gets 1? no: min to i, max to j).
    Actually: compare values at i and j. min value -> wire i, max value -> wire j.
    For 0/1: if x has bit i=1 and bit j=0, then min=0 (to i), max=1 (to j) -> swap.
    If x has bit i=0 and bit j=1, then min=0 (to i), max=1 (to j) -> already correct.
    So we swap bits i and j iff bit_i=1 and bit_j=0.
    """
    bi = (x >> i) & 1
    bj = (x >> j) & 1
    if bi == 1 and bj == 0:
        # swap: clear i, set j
        x = x ^ (1 << i) ^ (1 << j)
    return x

def _apply_network_bitparallel(nets):
    """Apply the network to all 65536 inputs simultaneously using bit-parallel tricks.
    Returns a boolean array of shape (65536,) indicating whether input is sorted.
    Actually, we need to check if output == sorted_mask[input].
    
    More efficient: simulate the network on all inputs at once.
    Each wire holds a 65536-bit vector (one bit per input).
    We can represent each wire as a Python integer of 65536 bits, or use numpy uint64 arrays.
    
    Let's use numpy: each wire is an array of 65536 uint8 (0/1) values.
    But that's 16 * 65536 = ~1MB, fine.
    
    Actually, for speed, let's use bit-parallel with Python ints (arbitrary precision).
    Each wire w has a 65536-bit integer W_w where bit k = value on wire w for input k.
    
    Initialize: for input k (0..65535), wire i has bit k = (k >> i) & 1.
    So W_i = sum over k where bit i of k is 1 of (1 << k).
    
    We can precompute W_i for all i.
    """
    # Precompute initial wire states
    # W_i has bit k set iff (k & (1<<i)) != 0
    # This is a pattern: for i=0, bits 1,3,5,... are set -> 0xAAAA...
    # For i=1, bits 2,3,6,7,10,11,... -> 0xCCCC...
    # General: W_i = sum_{k=0}^{65535} ((k>>i)&1) * (1<<k)
    
    wires = [0] * N
    for i in range(N):
        w = 0
        # Bit i of k is 1 for k in [2^i * (2m+1), 2^i*(2m+1) + 2^i - 1] for m=0,...
        # Simpler: iterate
        # Actually, we can build it iteratively or use the pattern.
        # Let's just compute it once at module load for the initial state.
        pass
    
    return wires

# Let's precompute initial wire bit-vectors
def _build_initial_wires():
    wires = [0] * N
    for i in range(N):
        w = 0
        for k in range(TOTAL_BITS):
            if (k >> i) & 1:
                w |= (1 << k)
        wires[i] = w
    return wires

INITIAL_WIRES = _build_initial_wires()

def _verify_network(nets):
    """Verify that the network sorts all inputs. Returns True if valid."""
    wires = list(INITIAL_WIRES)
    for (i, j) in nets:
        # Compare-exchange: min to i, max to j
        # For each input k, if wire_i[k]=1 and wire_j[k]=0, swap.
        # swap_mask = wires[i] & ~wires[j]  (bits where i=1, j=0)
        swap_mask = wires[i] & (~wires[j]) & ((1 << TOTAL_BITS) - 1)
        wires[i] = wires[i] ^ swap_mask
        wires[j] = wires[j] ^ swap_mask
    # Check: for each input k, the output pattern must match SORTED_MASKS[k]
    # Output on wire i for input k is bit k of wires[i].
    # The sorted output for input k has bits set in top popcount(k) positions.
    # So the integer formed by wires[0..N-1] at bit k should equal SORTED_MASKS[k].
    # i.e., sum_{i=0}^{N-1} ((wires[i] >> k) & 1) * (1 << i) == SORTED_MASKS[k]
    
    # More efficient: for each k, check.
    # Or: build the output integer for all k at once.
    # output_int[k] = sum_i (bit k of wires[i]) * 2^i
    # We want output_int == SORTED_MASKS element-wise.
    
    # Let's check by reconstructing
    # For efficiency, check a subset first? No, must be all.
    # Build output array
    out = np.zeros(TOTAL_BITS, dtype=np.uint64)
    for i in range(N):
        w = wires[i]
        # Extract bits 0..65535 of w into an array
        # w is a Python int with 65536 bits
        # Convert to bytes
        b = w.to_bytes(TOTAL_BITS // 8, byteorder='little')
        # Expand each bit to a byte
        # This is slow. Let's use a different approach.
        
    # Alternative: check each k
    for k in range(TOTAL_BITS):
        val = 0
        for i in range(N):
            if (wires[i] >> k) & 1:
                val |= (1 << i)
        if val != SORTED_MASKS[k]:
            return False
    return True

def _verify_network_fast(nets):
    """Faster verification using numpy."""
    wires = list(INITIAL_WIRES)
    for (i, j) in nets:
        swap_mask = wires[i] & (~wires[j]) & ((1 << TOTAL_BITS) - 1)
        wires[i] = wires[i] ^ swap_mask
        wires[j] = wires[j] ^ swap_mask
    
    # Convert each wire to a numpy array of bits
    # wires[i] is a Python int with 65536 bits
    # We need to check: for each k, the N-bit value (bit k of each wire) == SORTED_MASKS[k]
    
    # Build an N x 65536 bit matrix
    # For each wire i, get the bits as a numpy array
    bits = np.zeros((N, TOTAL_BITS), dtype=np.uint8)
    for i in range(N):
        w = wires[i]
        # Extract bits
        for k in range(TOTAL_BITS):
            bits[i, k] = (w >> k) & 1
        # This is slow. Let's use to_bytes and unpack.
    
    # Better approach: use the fact that we can check via integer comparison
    # Build the output vector as a list of 65536 integers
    outputs = np.zeros(TOTAL_BITS, dtype=np.uint64)
    for i in range(N):
        w = wires[i]
        # Get all bits of w
        # Use int.to_bytes
        byte_len = TOTAL_BITS // 8
        b = w.to_bytes(byte_len, byteorder='little')
        # Unpack bits
        # Each byte has 8 bits
        arr = np.frombuffer(b, dtype=np.uint8)
        # Expand to bits
        # bit 0 of byte = LSB
        # We want bits[0..65535]
        # Use lookup or bit extraction
        # For now, do it in a loop but vectorize with numpy
        for byte_idx in range(byte_len):
            byte_val = arr[byte_idx]
            base = byte_idx * 8
            for bit_idx in range(8):
                if (byte_val >> bit_idx) & 1:
                    outputs[base + bit_idx] |= (1 << i)
    
    return np.array_equal(outputs, SORTED_MASKS)

# Even faster: precompute the verification differently.
# Let's use a simple but reasonably fast method for the search.

def _verify(nets):
    """Verify network. Returns True if valid."""
    wires = list(INITIAL_WIRES)
    for (i, j) in nets:
        swap_mask = wires[i] & (~wires[j]) & ((1 << TOTAL_BITS) - 1)
        wires[i] = wires[i] ^ swap_mask
        wires[j] = wires[j] ^ swap_mask
    
    # Check all inputs
    for k in range(TOTAL_BITS):
        val = 0
        for i in range(N):
            if (wires[i] >> k) & 1:
                val |= (1 << i)
        if val != SORTED_MASKS[k]:
            return False
    return True

def construct():
    """Search for a minimal sorting network on 16 wires."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin
    
    # Start with odd-even transposition network: 120 comparators
    nets = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial
    if not _verify(nets):
        return nets
    
    best_nets = list(nets)
    best_len = len(nets)
    
    # Local search: try to remove comparators
    # Strategy: iterate through comparators, try removing each one, verify, keep if still valid
    # Repeat until no more can be removed
    
    max_rounds = 200
    for round_idx in range(max_rounds):
        if time.time() > deadline - 5.0:
            break
        
        removed_any = False
        current_nets = list(best_nets)
        
        for idx in range(len(current_nets) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            
            # Try removing current_nets[idx]
            candidate = current_nets[:idx] + current_nets[idx+1:]
            
            if _verify(candidate):
                current_nets = candidate
                removed_any = True
        
        if not removed_any:
            break
        
        best_nets = current_nets
        best_len = len(best_nets)
    
    # Also try swapping/adding? For now, just removal.
    # Try to add comparators in different positions and then remove more
    # This is complex; let's stick with removal for now.
    
    return best_nets
