"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE = 85.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires by starting from a known-good
    (odd-even transposition) network and greedily removing redundant
    comparators, verified bit-parallel over all 2^18 Boolean inputs.
    """
    start_time = time.time()
    
    # --- Helper: bit-parallel correctness check ---
    # Represent all 2^N inputs as N bit-vectors of length 2^N.
    # wire[i] is a uint8 array of length 2^N, where bit k of wire[i][mask]
    # gives the value on wire i for input mask.
    # Actually, for speed, we store as uint64 chunks? No, 2^18 = 262144 bits.
    # We can use Python ints (arbitrary precision) for the bit-vectors.
    # wire[i] is an int where bit k is the value on wire i for input k.
    
    def make_wires():
        """Create initial wire bit-vectors: wire i has bit k = (k >> i) & 1"""
        wires = []
        for i in range(N):
            # Build integer where bit k is set if (k >> i) & 1
            # k ranges from 0 to 2^N - 1
            val = 0
            # Efficient construction: for bit i of the index
            # Pattern: 0 for 2^i zeros, then 2^i ones, repeated
            block = (1 << (1 << i)) - 1
            # We need to repeat this block 2^(N-i-1) times? No.
            # The value of bit i of k is 1 for k in [2^i, 2*2^i-1], [3*2^i, 4*2^i-1], etc.
            # Number of blocks of size 2^(i+1) in 2^N is 2^(N-i-1).
            # Each block has 2^i zeros then 2^i ones.
            # So the bit pattern for wire i is: (2^i zeros, 2^i ones) repeated 2^(N-i-1) times.
            # As an integer, this is a repeating pattern.
            # Let's just build it directly for clarity, it's fast enough.
            # Actually, for N=18, 2^18 = 262144 bits. Building 18 such ints is fast.
            w = 0
            for k in range(1 << N):
                if (k >> i) & 1:
                    w |= (1 << k)
            wires.append(w)
        return wires
    
    def apply_compare(wires, i, j):
        """Apply compare-exchange between wires i and j. Returns new wires list."""
        # min to i, max to j
        # min(a,b) = a & b, max(a,b) = a | b
        wi = wires[i]
        wj = wires[j]
        new_wi = wi & wj
        new_wj = wi | wj
        new_wires = wires[:]
        new_wires[i] = new_wi
        new_wires[j] = new_wj
        return new_wires
    
    def is_sorted(wires):
        """Check if wires are sorted: wire 0 <= wire 1 <= ... <= wire N-1 (bitwise)"""
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
        return True
    
    def verify_network(network):
        """Verify network sorts all inputs. Returns True if correct."""
        wires = make_wires()
        for i, j in network:
            wires = apply_compare(wires, i, j)
        return is_sorted(wires)
    
    # --- Start with odd-even transposition network ---
    # For N wires, odd-even transposition needs N phases.
    # Phase p: compare wires (p%2, p%2+2, ...) and (p%2+1, p%2+3, ...)
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify initial network
    assert verify_network(network), "Initial network should be correct"
    
    best_network = network[:]
    best_size = len(network)
    
    # --- Greedy removal: try to remove each comparator ---
    # Iterate through the network and try to remove each comparator.
    # If the network is still correct after removal, keep it removed.
    # Repeat until no more removals are possible.
    
    removed = True
    while removed and (time.time() - start_time) < DEADLINE:
        removed = False
        for idx in range(len(network)):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx+1:]
            if verify_network(test_network):
                network = test_network
                best_network = network[:]
                best_size = len(network)
                removed = True
                break  # Restart loop since indices changed
    
    # --- Try random removals from the current network ---
    if (time.time() - start_time) < DEADLINE - 10:
        current = best_network[:]
        for _ in range(1000):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            if not current:
                break
            idx = random.randint(0, len(current) - 1)
            test = current[:idx] + current[idx+1:]
            if verify_network(test):
                current = test
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
    
    # --- Try swapping adjacent comparators ---
    if (time.time() - start_time) < DEADLINE - 10:
        current = best_network[:]
        for _ in range(500):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            if len(current) < 2:
                break
            idx = random.randint(0, len(current) - 2)
            test = current[:]
            test[idx], test[idx+1] = test[idx+1], test[idx]
            if verify_network(test):
                current = test
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
    
    return best_network
# EVOLVE-BLOCK-END
