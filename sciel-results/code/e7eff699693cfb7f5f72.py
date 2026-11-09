"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 17

# EVOLVE-BLOCK-START

def _build_seed_network():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _verify_network_fast(comparators):
    """
    Verify sorting network using fast Python big-integer bitwise operations.
    Returns the number of unsorted Boolean inputs (0 if sorted).
    """
    # For each input wire, we track which of the 131072 inputs have a 1 on that wire.
    # We represent the state of all inputs for a single wire as a big integer.
    # Bit k of the integer is 1 if input k has a 1 on this wire.
    
    # Initialize state:
    # Input k is an integer from 0 to 2^N - 1.
    # Wire i has a 1 if bit i of input k is 1.
    # We want to construct an integer S_i such that bit k of S_i is 1 iff bit i of k is 1.
    
    # This can be done efficiently.
    # The pattern of bits for wire i repeats every 2^(i+1).
    # In each block of 2^(i+1), the first 2^i bits are 0, the next 2^i are 1.
    # So we can construct the mask for wire i.
    
    states = [0] * N
    for i in range(N):
        # Construct mask for wire i
        # Pattern: 0...0 1...1 (2^i zeros, 2^i ones) repeated
        # Total bits: 2^N
        # Number of repeats: 2^(N-i-1)
        
        # Instead of building bit by bit, we can use the property of bit extraction.
        # But building the integer is fast enough for N=17 (131072 bits ~ 16KB).
        
        # Let's build the mask for wire i.
        # The mask for wire 0 is 010101...
        # The mask for wire 1 is 00110011...
        # We can generate this by repeating a pattern.
        
        # Pattern length: 2^(i+1)
        # Low bits (0s): 2^i
        # High bits (1s): 2^i
        
        # Let's just iterate through the bits? No, 131072 iterations per wire * 17 wires is ~2M ops.
        # That's fine in Python if we do it once.
        # But we can be smarter.
        
        # Actually, we can represent the state as a list of integers if we process in chunks,
        # but big ints are good.
        
        # Let's precompute the masks for all wires once?
        # No, we need to do this inside the function or cache it.
        # Since N is fixed, we can cache the initial states.
        
        pass

    # Let's use a simpler, robust approach for the mask generation that is fast.
    # We can use the fact that the inputs are 0..2^N-1.
    # Bit i of input x is (x >> i) & 1.
    # We want an integer M_i such that bit x of M_i is (x >> i) & 1.
    
    # This is equivalent to: M_i = sum_{x=0}^{2^N-1} ((x >> i) & 1) * 2^x
    
    # This sum can be computed recursively or using pattern repetition.
    # Let's just compute it once and store it in a global or closure.
    
    # For now, let's implement a helper that returns the list of initial masks.
    # We will cache this.
    
    if not hasattr(_verify_network_fast, "masks"):
        masks = []
        for i in range(N):
            m = 0
            # Iterate x from 0 to 2^N - 1
            # This loop is 131072 iterations.
            # Doing this for 17 wires is 2.2M iterations.
            # In Python, this might take ~0.1s - 0.2s. Acceptable for setup.
            # But we can do it faster.
            
            # Faster construction:
            # The pattern for wire i has period 2^(i+1).
            # In one period, the lower half is 0, upper half is 1.
            # Let P = 2^(i+1).
            # Block value B = (2^(2^i) - 1) << 2^i  (This is the upper half ones)
            # Wait, 2^i ones starting at bit 2^i.
            # So B = ((1 << (1 << i)) - 1) << (1 << i)
            
            # Then we repeat this block 2^(N-i-1) times.
            # M_i = B * (1 + 2^P + 2^(2P) + ... + 2^((K-1)P))
            # where K = 2^(N-i-1)
            
            # Geometric series sum: (2^(K*P) - 1) / (2^P - 1)
            # Note 2^(K*P) = 2^(2^(N-i-1) * 2^(i+1)) = 2^(2^N)
            
            # So M_i = B * ( (1 << (1 << N)) - 1 ) // ( (1 << P) - 1 )
            
            P = 1 << (i + 1)
            B = ((1 << (1 << i)) - 1) << (1 << i)
            total_bits = 1 << N
            # Sum = (2^total_bits - 1) / (2^P - 1)
            # This is an integer division.
            # (2^total_bits - 1) is a big integer with total_bits ones.
            # (2^P - 1) is a big integer with P ones.
            # The division is exact.
            
            # Python big int division is fast.
            num = (1 << total_bits) - 1
            den = (1 << P) - 1
            m = B * (num // den)
            masks.append(m)
        _verify_network_fast.masks = masks

    masks = _verify_network_fast.masks
    states = masks[:]
    
    for i, j in comparators:
        # Compare exchange on wires i and j
        # min goes to i, max goes to j
        # For each input bit k:
        # if state_i[k] == 0 and state_j[k] == 0 -> 0, 0
        # if state_i[k] == 1 and state_j[k] == 0 -> 0, 1 (min=0 to i, max=1 to j)
        # if state_i[k] == 0 and state_j[k] == 1 -> 0, 1 (min=0 to i, max=1 to j)
        # if state_i[k] == 1 and state_j[k] == 1 -> 1, 1
        
        # So:
        # new_state_i = state_i & state_j
        # new_state_j = state_i | state_j
        
        new_si = states[i] & states[j]
        new_sj = states[i] | states[j]
        states[i] = new_si
        states[j] = new_sj
        
    # Check if sorted
    # Sorted means:
    # For each wire i, the set of inputs with 1 on wire i must be a suffix of the inputs?
    # No.
    # The zero-one principle says the network sorts all inputs if and only if it sorts all Boolean inputs.
    # A Boolean input is sorted if the sequence of bits on the wires is non-decreasing?
    # No, the output must be sorted in ascending order.
    # Since the values are 0 and 1, the sorted output must have all 0s followed by all 1s.
    # This means that if we look at the states (which represent which inputs have a 1 on each wire),
    # the sets of inputs must be nested?
    
    # Let's think about the output.
    # The output is a sequence of N wires.
    # For a specific input x, the output is a bit vector (out_0, ..., out_{N-1}).
    # The network sorts x if out_0 <= out_1 <= ... <= out_{N-1}.
    # This is equivalent to: for all i < j, out_i <= out_j.
    # out_i is the bit x of states[i].
    # So we require: for all i < j, bit x of states[i] <= bit x of states[j].
    # This is equivalent to: states[i] is a subset of states[j] (as sets of input indices)?
    # If bit x of states[i] is 1, then bit x of states[j] must be 1.
    # Yes, states[i] must be a subset of states[j] for all i < j.
    
    # So we check:
    # states[0] & ~states[1] == 0
    # states[1] & ~states[2] == 0
    # ...
    # states[N-2] & ~states[N-1] == 0
    
    # If any of these is non-zero, it means there is an input x where wire i has 1 and wire j has 0 (i<j), which is unsorted.
    # We want to count the number of unsorted inputs.
    # An input x is unsorted if there exists i < j such that out_i[x]=1 and out_j[x]=0.
    # This is equivalent to: x is in (states[i] & ~states[j]) for some i < j.
    # The set of unsorted inputs is the union of these sets.
    # We can compute the union of all (states[i] & ~states[j]) for i < j.
    
    # However, a simpler check for "is it sorted" is just checking adjacent wires?
    # If out_i <= out_{i+1} for all i, then the sequence is sorted.
    # So we just need to check i and i+1.
    # Unsorted inputs are those where out_i[x]=1 and out_{i+1}[x]=0 for some i.
    # Union over i of (states[i] & ~states[i+1]).
    
    unsorted_mask = 0
    for i in range(N - 1):
        unsorted_mask |= (states[i] & ~states[i+1])
        
    # Count bits in unsorted_mask
    # Python 3.10+ has int.bit_count()
    # For compatibility, we can use bin(x).count('1') but that's slow for big ints.
    # Let's assume Python 3.10+ or use a fast bit count.
    # The task says "Python standard library... available". Usually implies modern Python.
    # I will use bit_count() and fallback if necessary, but bit_count is very fast.
    
    try:
        return unsorted_mask.bit_count()
    except AttributeError:
        return bin(unsorted_mask).count('1')

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial network
    network = _build_seed_network()
    current_cost = _verify_network_fast(network)
    best_network = network[:]
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T = 1000.0  # Initial temperature
    cooling_rate = 0.999
    
    # We want to minimize size (len(network)) subject to cost == 0.
    # Objective function for SA:
    # If cost > 0, we want to reduce cost.
    # If cost == 0, we want to reduce size.
    # Let's define a combined cost:
    # C = current_cost * LARGE_NUMBER + len(network)
    # But we can't just use this because we want to allow moves that increase cost if they lead to smaller size later?
    # Actually, standard approach for constraint satisfaction + optimization:
    # 1. Satisfy constraints (cost == 0).
    # 2. Minimize objective (size).
    
    # Strategy:
    # If current cost > 0:
    #   Accept move if new_cost < current_cost (always)
    #   Accept move if new_cost == current_cost with prob p
    #   Accept move if new_cost > current_cost with prob exp(-(new_cost-current_cost)/T)
    #   (Moves that increase size are generally bad here, but maybe necessary to escape? 
    #    Actually, if cost > 0, adding comparators usually helps reduce cost. Removing hurts.
    #    So we should bias towards adding comparators when cost > 0.)
    
    # If current cost == 0:
    #   We want to minimize size.
    #   Accept move if new_cost == 0 and new_size < current_size (always)
    #   Accept move if new_cost == 0 and new_size == current_size (with prob p)
    #   Accept move if new_cost == 0 and new_size > current_size (with prob exp(-(new_size-current_size)/T))
    #   Accept move if new_cost > 0 (with prob exp(-(new_cost * PENALTY)/T))
    #   Here, we must be careful not to break validity permanently.
    #   If we break validity, we need to be able to fix it.
    
    # Let's use a unified cost function:
    # Total_Cost = current_cost * 1000 + len(network)
    # This strongly prioritizes correctness.
    # Then we run SA on Total_Cost.
    
    def get_cost(net):
        c = _verify_network_fast(net)
        return c * 1000 + len(net)
    
    current_total_cost = get_cost(network)
    best_total_cost = current_total_cost
    
    random.seed(42)
    
    while time.time() < deadline:
        # Generate a move
        # Options:
        # 1. Remove a random comparator
        # 2. Add a random comparator
        # 3. Swap two comparators (changes depth, not size, but might help? No, size is the metric. Depth is reported separately but not scored. So swapping doesn't change size or correctness. It's a no-op for the score. Skip.)
        # 4. Replace a comparator with another?
        
        # Let's stick to Add and Remove.
        # Weight the choice based on current state.
        # If cost > 0, we likely need to add comparators.
        # If cost == 0, we likely need to remove comparators.
        
        if current_cost > 0:
            # 80% Add, 20% Remove
            if random.random() < 0.8:
                move_type = 'add'
            else:
                move_type = 'remove'
        else:
            # 80% Remove, 20% Add
            if random.random() < 0.8:
                move_type = 'remove'
            else:
                move_type = 'add'
        
        new_network = network[:]
        
        if move_type == 'remove':
            if len(new_network) == 0:
                continue
            idx = random.randint(0, len(new_network) - 1)
            removed = new_network.pop(idx)
            # Check if it's a duplicate? If we remove one of duplicates, it's fine.
            # We don't need to check validity of the move itself, just the result.
        else:
            # Add a random comparator
            # To avoid too many duplicates, we can pick from a set of "useful" comparators?
            # Or just random i, j.
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_network.append((i, j))
            # Optional: limit max size to avoid explosion
            if len(new_network) > 200: 
                continue
        
        new_cost = _verify_network_fast(new_network)
        new_total_cost = new_cost * 1000 + len(new_network)
        
        delta = new_total_cost - current_total_cost
        
        if delta <= 0:
            # Better or equal
            network = new_network
            current_cost = new_cost
            current_total_cost = new_total_cost
            
            if new_total_cost < best_total_cost:
                best_total_cost = new_total_cost
                best_network = network[:]
        else:
            # Worse
            # Accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                network = new_network
                current_cost = new_cost
                current_total_cost = new_total_cost
        
        # Cool down
        T *= cooling_rate
        
        # If T is too low, restart?
        # If we are stuck at cost=0 and can't reduce size, maybe restart with higher T?
        # Or just keep going.
        
        # Safety check: if we are at cost 0 and size is minimal possible (16 for 17 wires? No, lower bound is 63), 
        # we can't do better than 63. If we reach 63, we can stop.
        if best_cost == 0 and len(best_network) <= 63:
            break

    # Return the best network found
    # Ensure it's valid. If best_cost > 0, we return the best one we found (which might be invalid?).
    # The task says "invalid result ... scores 0".
    # So we MUST return a valid network.
    # If we didn't find a valid one, we should return the seed (which is valid).
    # The seed is always valid.
    
    # Check if best_network is valid.
    # best_cost is the cost of best_network.
    # If best_cost > 0, it's invalid.
    # In that case, return the seed.
    
    if best_cost > 0:
        return _build_seed_network()
    
    return best_network

# EVOLVE-BLOCK-END
