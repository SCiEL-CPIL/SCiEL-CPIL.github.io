"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random

def _verify_network_fast(network):
    """
    Verify network sorts all 2^N Boolean inputs using the zero-one principle.
    Represents each wire's state across all 65536 inputs as a Python int bitmask.
    """
    n = N
    wire_state = []
    for w in range(n):
        period = 1 << (w + 1)
        half = 1 << w
        period_mask = 0
        for bit_pos in range(half, period):
            period_mask |= (1 << bit_pos)
        mask = 0
        for rep in range(1 << (n - w - 1)):
            mask |= (period_mask << (rep * period))
        wire_state.append(mask)
    
    for (i, j) in network:
        a = wire_state[i]
        b = wire_state[j]
        wire_state[i] = a & b
        wire_state[j] = a | b
    
    for k in range(n - 1):
        if (wire_state[k] & ~wire_state[k + 1]) != 0:
            return False
    return True

def _generate_bitonic_network(n):
    """Generate a correct Bitonic Sorting Network for n wires (power of 2)."""
    network = []
    
    def bitonic_merge(size, offset, direction):
        """
        Merge two bitonic sequences into one sorted sequence.
        direction: 0 for ascending, 1 for descending.
        """
        if size <= 1:
            return
        half = size // 2
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if direction == 0:
                network.append((a, b))
            else:
                network.append((b, a))
        bitonic_merge(half, offset, direction)
        bitonic_merge(half, offset + half, direction)
    
    def bitonic_sort(size, offset, direction):
        """
        Sort a sequence into a bitonic sequence (or sorted if size=2).
        direction: 0 for ascending, 1 for descending.
        """
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, offset, 0)
        bitonic_sort(half, offset + half, 1)
        bitonic_merge(size, offset, direction)
    
    bitonic_sort(n, 0, 0)
    return network

def _generate_odd_even_network(n):
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s safety margin
    
    # Start with Bitonic Sorting Network (should be ~80-90 comparators for N=16)
    network = _generate_bitonic_network(N)
    
    if not _verify_network_fast(network):
        # Fallback to Odd-Even if Bitonic fails (shouldn't happen)
        network = _generate_odd_even_network(N)
    
    best_network = network
    best_count = len(network)
    
    # Local Search: Prune redundant comparators
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(best_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = best_network[:idx] + best_network[idx+1:]
            
            if _verify_network_fast(candidate):
                best_network = candidate
                improved = True
                break
        
    # Try to further optimize with random insertions and prunings
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline:
                break
            
            current = best_network[:]
            # Add some random comparators to allow rearrangement
            for _ in range(3):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(current))
                current.insert(pos, (i, j))
            
            # Prune again
            improved = True
            while improved and time.time() < deadline:
                improved = False
                indices = list(range(len(current)))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() > deadline:
                        break
                    candidate = current[:idx] + current[idx+1:]
                    if _verify_network_fast(candidate):
                        current = candidate
                        improved = True
                        break
            
            if len(current) < best_count:
                best_network = current
                best_count = len(current)
    
    return best_network
# EVOLVE-BLOCK-END
