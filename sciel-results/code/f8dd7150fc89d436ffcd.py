"""
Size-efficient sorting network for 17 wires.
Strategy:
1. Start from a known-good seed (odd-even transposition, 136 comparators).
2. Use a genetic algorithm to minimize comparator count.
   - Representation: list of (i, j) pairs.
   - Fitness: number of comparators (lower is better).
   - Verification: Zero-One Principle via bit-vector simulation.
   - Operations: 
       * Crossover: splice two networks.
       * Mutation: add/remove/swap comparators, or move a comparator.
   - Constraint: network must be valid (sorts all 2^17 Boolean inputs).
3. Deadline guard: track elapsed time, return best valid result before limit.
"""

import time
import random
import numpy as np

N = 17
TIME_LIMIT = 85.0  # Leave 5s safety margin for 90s wall-clock

def _time_left(deadline):
    return deadline - time.time()

def _verify_network(comparators):
    """
    Verify that the network sorts all 2^N Boolean inputs (Zero-One Principle).
    Uses bit-vector simulation: each wire holds a 64-bit integer representing
    multiple test cases at once. We need 2^17 = 131072 test cases.
    We process them in batches of 64.
    """
    if len(comparators) == 0:
        return N <= 1
    
    # Precompute all possible inputs? No, too many. 
    # Instead, for each batch of 64 inputs, simulate the network.
    # But 131072 / 64 = 2048 batches. Each batch simulates len(comparators) steps.
    # Total ops: 2048 * len(comparators). If len is ~100, that's 204,800 ops. Fast.
    
    # We need to check if the output is sorted (0s then 1s) for each input.
    # A sorted output of length N with k ones is: [0]* (N-k) + [1]*k.
    
    # Optimization: Instead of simulating all 2^N inputs, we can use the fact that
    # the network is monotone. But simplest is to just check all.
    # However, 2^17 is 131,072. Let's see if we can do it faster.
    
    # Alternative: Use the "zero-one principle" which says it's sufficient to check
    # all 2^N inputs. There is no smaller set known for general networks.
    
    # Let's optimize the simulation.
    # Represent each wire as a 64-bit int.
    # For input mask m (0 to 2^N - 1), the initial state is the bits of m.
    
    # We can process 64 inputs at a time.
    # For each input index in the batch, we have a 17-bit number.
    # We pack 64 such numbers into 17 numpy uint64 arrays? No, just Python ints are fine.
    
    # Actually, let's just iterate over all 2^N inputs and simulate.
    # 131,072 inputs * ~100 comparators = 13 million operations. In Python, this might be slow.
    # Let's use NumPy vectorization across the inputs.
    
    # Create an array of all 2^N inputs, shape (2^N, N).
    # This is 131072 * 17 bytes = ~2MB. Fine.
    
    # But we can do better. We only need to check if the network is a sorting network.
    # A faster check: 
    # 1. Check if the network is "almost" sorted.
    # 2. Use the fact that if it sorts all permutations of 0s and 1s, it's correct.
    
    # Let's try the NumPy approach for speed.
    # inputs = np.arange(2**N, dtype=np.uint8).reshape(-1, 1)
    # bits = np.zeros((2**N, N), dtype=np.uint8)
    # for i in range(N):
    #     bits[:, i] = (inputs >> (N-1-i)) & 1
    
    # This is too slow to generate in Python loop.
    
    # Better:
    # bits = np.zeros((2**N, N), dtype=np.uint8)
    # for i in range(N):
    #     bits[:, i] = ((np.arange(2**N) >> (N-1-i)) & 1).astype(np.uint8)
    
    # Even better: use unpackbits or similar.
    # Actually, let's just do a simple loop but with early termination if possible? No.
    
    # Let's try a different verification: 
    # A network is a sorting network iff it sorts all 2^N inputs.
    # We can simulate using bit operations on Python ints, processing 64 inputs at a time.
    
    # Let's implement the 64-at-a-time simulation.
    
    total_inputs = 1 << N
    batch_size = 64
    num_batches = (total_inputs + batch_size - 1) // batch_size
    
    for batch_start in range(0, total_inputs, batch_size):
        batch_end = min(batch_start + batch_size, total_inputs)
        count = batch_end - batch_start
        
        # Initialize wires for this batch of inputs.
        # wires[i] is a 64-bit int, bit k corresponds to input (batch_start + k).
        wires = [0] * N
        for k in range(count):
            inp = batch_start + k
            for i in range(N):
                if (inp >> (N - 1 - i)) & 1:
                    wires[i] |= (1 << k)
        
        # Simulate network
        for (a, b) in comparators:
            # min to a, max to b
            # min = wires[a] & wires[b]
            # max = wires[a] | wires[b]
            wa = wires[a]
            wb = wires[b]
            wires[a] = wa & wb
            wires[b] = wa | wb
        
        # Check if sorted: for each input k, the output should be 0s then 1s.
        # i.e., once we see a 1, all subsequent bits must be 1.
        # This is equivalent to: for each k, the output bits are non-decreasing.
        # Or: for each i from 0 to N-2, if wire[i] has a 1 where wire[i+1] has a 0, it's not sorted.
        # Actually, sorted means: wire[0] <= wire[1] <= ... in terms of the bit values? No.
        # For a specific input k, the output is a sequence of 0s and 1s. It is sorted if it's of the form 0...01...1.
        # This means that for all i, if output[i] = 1 then output[i+1] must be 1? No, that's not right.
        # 0,0,1,1 is sorted. 1,0,0,1 is not.
        # Condition: there is no i such that output[i]=1 and output[i+1]=0.
        # In bit-vector form: for all i, ~(wires[i] & ~wires[i+1]) should be 0? 
        # Wait, if wires[i] has a 1 at position k and wires[i+1] has a 0 at position k, that's a violation.
        # So we check: for all i, (wires[i] & ~wires[i+1]) == 0.
        
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
    
    return True

def _depth_of_network(comparators):
    """Calculate the parallel depth of the network."""
    if not comparators:
        return 0
    wire_last = [-1] * N
    depth = 0
    for (a, b) in comparators:
        # A comparator can be placed at depth max(wire_last[a], wire_last[b]) + 1
        # But since we are given a sequential list, we assume they are in order.
        # The depth is the length of the longest chain of dependent comparators.
        # Actually, for a list of comparators, the depth is the minimum number of parallel steps.
        # This is equivalent to coloring the comparators such that no two in the same step share a wire.
        # Greedy coloring:
        current_step = 0
        used_in_step = set()
        for (a, b) in comparators:
            if a in used_in_step or b in used_in_step:
                current_step += 1
                used_in_step = set()
            used_in_step.add(a)
            used_in_step.add(b)
        return current_step + 1

def _make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _random_network():
    """Generate a random network with random number of comparators."""
    num_comp = random.randint(100, 150)
    comps = []
    for _ in range(num_comp):
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
        comps.append((i, j))
    return comps

def _crossover(parent1, parent2):
    """Splice crossover."""
    if not parent1 or not parent2:
        return parent1 if parent1 else parent2
    point = random.randint(0, min(len(parent1), len(parent2)))
    child = parent1[:point] + parent2[point:]
    return child

def _mutate(comparators):
    """Random mutation."""
    comps = list(comparators)
    if not comps:
        return comps
    
    op = random.choice(['add', 'remove', 'swap', 'move', 'new_comp'])
    
    if op == 'add' and len(comps) < 4096:
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
        pos = random.randint(0, len(comps))
        comps.insert(pos, (i, j))
        
    elif op == 'remove' and len(comps) > 10:
        pos = random.randint(0, len(comps) - 1)
        comps.pop(pos)
        
    elif op == 'swap' and len(comps) >= 2:
        i, j = random.sample(range(len(comps)), 2)
        comps[i], comps[j] = comps[j], comps[i]
        
    elif op == 'move' and len(comps) >= 2:
        i, j = random.sample(range(len(comps)), 2)
        comp = comps.pop(i)
        comps.insert(j, comp)
        
    elif op == 'new_comp':
        pos = random.randint(0, len(comps) - 1)
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
        comps[pos] = (i, j)
        
    return comps

def construct():
    """
    Construct a sorting network for N=17 wires.
    Returns a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Seed network
    seed = _make_seed()
    best = seed
    best_score = len(seed)
    
    # Verify seed
    if not _verify_network(seed):
        raise ValueError("Seed network is invalid")
    
    # Genetic algorithm
    pop_size = 20
    population = [seed]
    for _ in range(pop_size - 1):
        population.append(_mutate(seed))
    
    generation = 0
    while _time_left(deadline) > 1.0:
        generation += 1
        
        # Evaluate fitness
        scored = []
        for ind in population:
            if len(ind) >= best_score:
                # Skip if not better
                scored.append((len(ind), ind))
            else:
                if _verify_network(ind):
                    scored.append((len(ind), ind))
                else:
                    scored.append((float('inf'), ind))
        
        # Sort by fitness
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] < best_score:
            best_score = scored[0][0]
            best = scored[0][1]
            # Print progress
            print(f"Gen {generation}: New best {best_score}")
        
        # Selection: keep top half
        survivors = [ind for _, ind in scored[:pop_size // 2]]
        
        # Create next generation
        new_population = survivors.copy()
        while len(new_population) < pop_size:
            if random.random() < 0.5:
                # Mutation
                parent = random.choice(survivors)
                child = _mutate(parent)
            else:
                # Crossover
                p1, p2 = random.sample(survivors, 2)
                child = _crossover(p1, p2)
                child = _mutate(child)
            new_population.append(child)
        
        population = new_population
        
        # Check time
        if _time_left(deadline) < 5.0:
            break
    
    return best

# For testing
if __name__ == "__main__":
    net = construct()
    print(f"Network size: {len(net)}")
    print(f"Depth: {_depth_of_network(net)}")
    print(f"Valid: {_verify_network(net)}")
