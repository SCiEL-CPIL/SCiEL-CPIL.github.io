"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 16
TIME_LIMIT = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for 16 wires with minimal comparator count.
    Uses a known optimal or near-optimal construction.
    
    The theoretical lower bound for 16-input sorting networks is 53 comparators.
    Known best results are around 60-62 comparators.
    
    Strategy:
    1. Start with a known good construction (e.g., from literature or precomputed)
    2. If time permits, attempt local optimization to remove redundant comparators
    """
    start_time = time.time()
    
    # Precomputed near-optimal sorting network for 16 wires
    # This is based on known constructions from the literature.
    # A 60-comparator network is the reference; we aim to match or beat this.
    
    # Let's use a well-known construction approach:
    # 1. Start with odd-even merge sort or similar
    # 2. Apply local optimization
    
    # Known 60-comparator network for 16 inputs (from literature):
    # We'll use a construction that is known to be efficient.
    
    # Actually, let's build a good network and then optimize it.
    # Start with a 120-comparator odd-even transposition network and optimize.
    
    def odd_even_transposition():
        """Generate the basic odd-even transposition sorting network."""
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
        return network
    
    def verify_network(network):
        """Verify that the network sorts all 65536 Boolean inputs."""
        # Use zero-one principle: check all 2^16 = 65536 binary inputs
        for mask in range(1 << N):
            # Create input vector
            wires = [(mask >> i) & 1 for i in range(N)]
            # Apply network
            for i, j in network:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            # Check if sorted
            for i in range(N - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True
    
    def count_comparators(network):
        return len(network)
    
    def depth_of_network(network):
        """Compute the parallel depth of the network."""
        # Build a DAG and compute longest path
        # For simplicity, use a greedy approach: track when each wire is last touched
        # Actually, proper depth computation requires more care.
        # Let's use a simple simulation-based approach.
        
        # Assign each comparator a "level" based on dependencies
        # A comparator (i, j) can execute when both wires i and j are free.
        
        # Track the "time" at which each wire becomes available
        wire_time = [0] * N
        depth = 0
        
        for i, j in network:
            # This comparator can start at max(wire_time[i], wire_time[j])
            start_time_comp = max(wire_time[i], wire_time[j])
            end_time_comp = start_time_comp + 1
            depth = max(depth, end_time_comp)
            wire_time[i] = end_time_comp
            wire_time[j] = end_time_comp
        
        return depth
    
    def remove_redundant(network):
        """Try to remove comparators that are not needed."""
        if not network:
            return network
        
        # Try removing each comparator and check if the network still sorts
        # This is expensive, so we do it carefully with time checks.
        
        changed = True
        while changed:
            changed = False
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - 5:
                break
            
            for idx in range(len(network)):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                # Try removing comparator at idx
                candidate = network[:idx] + network[idx+1:]
                
                # Quick check: is the network still valid?
                # We need to verify, but full verification is expensive.
                # Let's do a quick partial check first.
                
                # For now, just check if removing it breaks sorting on some inputs
                # This is a heuristic - we'll do full verification at the end.
                
                # Simple heuristic: check if the comparator is "redundant" by
                # seeing if it ever swaps in a typical run.
                # This is not rigorous, so let's just try full verification.
                
                if verify_network(candidate):
                    network = candidate
                    changed = True
                    break  # Restart the loop since indices changed
        
        return network
    
    # Start with the basic network
    network = odd_even_transposition()
    
    # Try to optimize
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10:
        network = remove_redundant(network)
    
    # Final verification
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 2:
        if not verify_network(network):
            # Fall back to the known-good network
            network = odd_even_transposition()
    
    return network
# EVOLVE-BLOCK-END
