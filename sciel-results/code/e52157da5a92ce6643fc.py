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
            # Pattern: 2^i zeros, 2^i ones.
            # Period width: 2^(i+1)
            # Number of periods: 2^(N-i-1)
            period_width = 1 << (i + 1)
            num_periods_bits = N - i - 1
            
            # Construct the single period: lower half 0, upper half 1
            # Upper half is 2^i ones.
            mask = (1 << (1 << i)) - 1
            period = mask << (1 << i)
            
            # Repeat period num_periods times using iterative doubling
            x = period
            shift = period_width
            for _ in range(num_periods_bits):
                x |= x << shift
                shift <<= 1
            wires.append(x)
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
    
    def verify_network(network, skip_idx=None):
        """Verify network sorts all inputs. Returns True if correct."""
        wires = make_wires()
        for idx, (i, j) in enumerate(network):
            if idx == skip_idx:
                continue
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
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
            if verify_network(network, skip_idx=idx):
                network.pop(idx)
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
            if verify_network(current, skip_idx=idx):
                current.pop(idx)
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
            # Swap in place for testing
            current[idx], current[idx+1] = current[idx+1], current[idx]
            if verify_network(current):
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
            else:
                # Revert swap
                current[idx], current[idx+1] = current[idx+1], current[idx]
    
    return best_network
# EVOLVE-BLOCK-END
