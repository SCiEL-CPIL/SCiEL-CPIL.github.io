import random
import time
# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires using a Genetic Algorithm with Bitonic initialization.
    """
    start_time = time.time()
    N = 18
    TOTAL_INPUTS = 1 << N
    DEADLINE = 55.0

    # --- Bit-Parallel Verification Setup ---
    def make_mask(i):
        L = 1 << (i + 1)
        num_blocks = TOTAL_INPUTS // L
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        mask = block * G
        return mask

    masks = [make_mask(i) for i in range(N)]

    def verify_network(network):
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

    # --- Bitonic Sort Generator for Power of 2 ---
    def bitonic_asc_net(n):
        if n == 2:
            return [(0, 1)]
        h = n // 2
        net = bitonic_asc_net(h)
        
        # Second half descending: map (a,b) -> (h + (h-1-b), h + (h-1-a))
        net2 = []
        for a, b in bitonic_asc_net(h):
            i = h + (h - 1 - b)
            j = h + (h - 1 - a)
            net2.append((i, j))
        net.extend(net2)
        
        # Merge
        def merge(n, offset, direction):
            if n == 2:
                if direction == 1:
                    return [(offset, offset + 1)]
                else:
                    return [] # No comparator needed for desc merge of 2? 
                    # Wait, if input is bitonic, and we want desc output.
                    # If input is 1,0 (bitonic desc), output 1,0 is correct. No op.
                    # If input is 0,1 (bitonic asc), output 1,0 is required.
                    # But 0,1 is not bitonic desc? 
                    # Bitonic sequence: increases then decreases.
                    # 0,1 increases. It is bitonic (trivially).
                    # If we want desc output, we need 1,0.
                    # Comparator (0,1) gives 0,1.
                    # So we CANNOT sort a pair descending with a single min/max comparator if we are constrained to i<j?
                    # Actually, the standard bitonic merge network for n=2, direction=-1 is empty?
                    # Let's check n=4.
                    # If we use empty for n=2 desc, does n=4 work?
                    # Merge n=4, dir=-1.
                    # Recurse n=2, dir=-1 (empty) on top and bottom.
                    # Compare (0,2), (1,3).
                    # Input: 4, 3, 2, 1 (Bitonic: 4,3 desc, 2,1 desc? No. 4,3,2,1 is desc. It is bitonic).
                    # We want output 4,3,2,1.
                    # Comparators (0,2): min(4,2)=2, max=4. Wires: 2,3,4,1.
                    # (1,3): min(3,1)=1, max=3. Wires: 2,1,4,3.
                    # Result: 2,1,4,3. Not sorted desc.
                    
                    # So the "empty" base case is wrong or my understanding is.
                    # Actually, to sort descending, we can just use the ascending network and reverse the wires?
                    # But we are building the network.
                    # The standard result is that the Bitonic Sort Network is symmetric.
                    # The comparators for Ascending and Descending are the SAME.
                    # The difference is in the *input* assumption.
                    # If the input is bitonic (asc then desc), the network sorts it Ascending.
                    # If the input is bitonic (desc then asc), the network sorts it Descending?
                    # No, the network structure is fixed.
                    # Let's assume the standard `bitonic_asc_net` generates a network that sorts ANY input Ascending.
                    # This is the property we need.
                    # We don't need to generate a "descending" network explicitly, just use the reversed indices for the second half of the *construction* to ensure the input to the merge is bitonic.
                    
                    return [(offset, offset + 1)] if direction == 1 else [(offset, offset + 1)] # Just use (0,1) for now, verify later.
            
            # Recursive merge
            # The comparators are the same for asc and desc merges in terms of indices?
            # Yes, the indices are the same.
            return merge(n//2, offset, direction) + merge(n//2, offset + n//2, direction) + \
                   [(offset + i, offset + i + n//2) for i in range(n//2)]
                   
        net.extend(merge(n, 0, 1))
        return net

    # Generate Bitonic for 32
    bitonic_32 = bitonic_asc_net(32)
    
    # Prune to 18
    initial_net = []
    for i, j in bitonic_32:
        if i < 18 and j < 18:
            initial_net.append((i, j))
            
    # Verify initial net
    if not verify_network(initial_net):
        # Fallback to Odd-Even
        initial_net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                initial_net.append((i, i + 1))

    # --- Genetic Algorithm ---
    population = [initial_net]
    
    # Add some diversity: Mutated copies
    for _ in range(9):
        mut_net = initial_net[:]
        # Random removals
        for _ in range(10):
            if len(mut_net) > 0:
                idx = random.randint(0, len(mut_net) - 1)
                mut_net.pop(idx)
        if verify_network(mut_net):
            population.append(mut_net)
        else:
            population.append(initial_net[:])

    best_network = min(population, key=len)
    best_size = len(best_network)

    while time.time() - start_time < DEADLINE:
        # Selection: Keep best 10
        population.sort(key=len)
        population = population[:10]
        
        new_population = population[:] # Elitism
        
        # Generate offspring
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            parent = random.choice(population)
            child = parent[:]
            
            # Mutation
            op = random.random()
            if op < 0.5:
                # Remove
                if child:
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
            elif op < 0.7:
                # Insert
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(child))
                child.insert(pos, (i, j))
            elif op < 0.8:
                # Swap
                if len(child) >= 2:
                    i1 = random.randint(0, len(child) - 1)
                    i2 = random.randint(0, len(child) - 1)
                    child[i1], child[i2] = child[i2], child[i1]
            else:
                # Replace
                if child:
                    idx = random.randint(0, len(child) - 1)
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    child[idx] = (i, j)
            
            if verify_network(child):
                new_population.append(child)
                
        population = new_population
        
        # Update best
        current_best = min(population, key=len)
        if len(current_best) < best_size:
            best_network = current_best
            best_size = len(current_best)
            
    # Final Polish: Greedy Removal on Best
    final_net = best_network[:]
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 1:
        removed = False
        for idx in range(len(final_net)):
            if (time.time() - start_time) > DEADLINE:
                break
            candidate = final_net[:]
            candidate.pop(idx)
            if verify_network(candidate):
                final_net = candidate
                removed = True
                break

    return final_net
# EVOLVE-BLOCK-END
