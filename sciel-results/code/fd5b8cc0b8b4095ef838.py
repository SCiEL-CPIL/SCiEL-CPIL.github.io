import time
import random
import sys

N = 16

def construct():
    # Precompute input masks for zero-one principle verification
    # Bit k of mask[i] is 1 if the i-th input of the k-th test case is 1.
    masks = [0] * N
    for i in range(N):
        m = 0
        bit = 1 << i
        for k in range(1 << 16):
            if (k >> i) & 1:
                m |= 1 << k
        masks[i] = m

    def verify(network):
        # Copy masks
        curr = masks[:]
        for u, v in network:
            if u == v:
                continue
            a = curr[u]
            b = curr[v]
            curr[u] = a & b
            curr[v] = a | b
        
        # Check if sorted: curr[i] subset curr[i+1]
        for i in range(N - 1):
            if curr[i] & ~curr[i + 1]:
                return False
        return True

    # Generate initial network: Odd-Even Transposition
    # This uses 120 comparators.
    initial_network = []
    for p in range(N):
        if p % 2 == 0:
            for i in range(0, N - 1, 2):
                initial_network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                initial_network.append((i, i + 1))
    
    if not verify(initial_network):
        # Should not happen, but just in case
        return initial_network

    best_network = initial_network[:]
    best_count = len(best_network)
    
    start_time = time.time()
    limit = 55.0 # 55 seconds to be safe (limit is 60s, but we need to return)
    
    # Simulated Annealing / Local Search
    # State: current network
    # Move: Remove a random comparator
    # Cost: 1 if valid (we want to minimize count, so valid removal is good), invalid is bad.
    
    curr_network = best_network[:]
    
    # To speed up, we can also try to add comparators if stuck, but removal is the main goal.
    # We can also try to swap?
    # Let's stick to removal and random restarts/additions.
    
    # To improve performance, we can pre-compute some things?
    # No, the state changes.
    
    # Let's optimize the inner loop.
    # verify() is the bottleneck.
    # 65536 bits * 16 wires.
    # Python big ints are efficient.
    
    # Heuristic: Try to remove comparators that are "redundant".
    # How to check redundancy?
    # A comparator (u, v) is redundant if removing it still results in a sorted network.
    # This is exactly what verify(new_network) does.
    
    # We can also try to remove multiple comparators at once?
    # If we remove one and it's valid, we can try to remove another.
    
    # Let's add a "kick" mechanism: if we're stuck for a while, add a random comparator.
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() - start_time < limit:
        if not curr_network:
            break
            
        # Try to remove a random comparator
        idx = random.randint(0, len(curr_network) - 1)
        removed = curr_network[idx]
        new_network = curr_network[:idx] + curr_network[idx+1:]
        
        if verify(new_network):
            curr_network = new_network
            if len(curr_network) < best_count:
                best_network = curr_network[:]
                best_count = len(best_network)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Add a random comparator to escape local minima
            # Find a pair that is not already in the network?
            # Or just add any valid pair.
            # If we add a duplicate, it's fine (idempotent), but wasteful.
            # Let's try to add a pair that is not present.
            present = set(curr_network)
            all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
            candidates = [p for p in all_pairs if p not in present]
            if candidates:
                curr_network.append(random.choice(candidates))
            else:
                # All pairs present? Unlikely.
                curr_network.append((0, 1))
                
            no_improve_count = 0
            max_no_improve = random.randint(1000, 3000)
            
    return best_network

if __name__ == "__main__":
    net = construct()
    print(len(net))
