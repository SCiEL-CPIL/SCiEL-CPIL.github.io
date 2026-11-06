import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 18
    total_inputs = 1 << N
    
    # 1. Precompute Initial Wires (BVO)
    # Wire k has bit i set if input i has 1 at position k
    init_wires = [0] * N
    # Optimization: Use a pre-calculated pattern or fast loop
    # For N=18, total_inputs = 262144.
    # Looping 18 * 262144 = ~4.7M iterations is acceptable in Python if optimized, 
    # but we can do better.
    # We can build the integer for wire k using the formula for the bit pattern.
    # Wire k pattern: 0...0 1...1 0...0 1...1
    # Period is 2^(k+1). Half period is 2^k.
    # In each period, the second half is 1s.
    # Value = sum( 2^i for i where (i mod 2^(k+1)) >= 2^k )
    
    # Let's use a fast construction method.
    for k in range(N):
        w = 0
        # We can iterate in chunks of 2^(k+1)
        period = 1 << (k + 1)
        half = 1 << k
        # Number of full periods
        num_periods = total_inputs // period
        remainder = total_inputs % period
        
        # This loop is small (at most 2^(N-k-1) iterations)
        # For k=0, 2^17 = 131072 iterations. This is too slow in pure Python loop if done naively.
        # We need a faster way to set bits.
        
        # Alternative: 
        # The bits set are i such that (i >> k) & 1 == 1.
        # This is equivalent to:
        # i = m * 2^(k+1) + 2^k + r, where 0 <= r < 2^k.
        # We can construct the integer by OR-ing shifted masks.
        
        # Mask for one block of 2^(k+1) bits:
        # Bits 0..2^k-1 are 0. Bits 2^k..2^(k+1)-1 are 1.
        # Mask = ((1 << half) - 1) << half
        mask = ((1 << half) - 1) << half
        
        # We need to repeat this mask 'num_periods' times, shifted by 'period' each time.
        # And handle the remainder.
        
        w = 0
        # Unroll the loop? No, just a simple loop is fine if we use bit operations.
        # But 131k iterations for k=0 is ~0.05s. Total for all k is dominated by k=0,1.
        # Sum 2^(N-k-1) for k=0..17 is 2^18 - 1 = 262143.
        # So total operations is ~262k bit shifts and ORs. This is very fast.
        
        for m in range(num_periods):
            w |= (mask << (m * period))
        
        # Handle remainder
        if remainder > 0:
            # The remainder part is the first 'remainder' bits of the next block.
            # The next block starts with 'half' zeros, then 'half' ones.
            # If remainder <= half, no bits are set in the remainder part (since first half is 0).
            # If remainder > half, the bits set are from index 'half' to 'remainder-1'.
            if remainder > half:
                # Number of 1s is remainder - half
                num_ones = remainder - half
                # These are the lowest 'num_ones' bits of the mask, shifted to the end of the period?
                # No, they are at positions:
                # offset = num_periods * period
                # local indices: half ... half + num_ones - 1
                # So we take the lowest 'num_ones' bits of the mask?
                # Mask has 1s at half ... period-1.
                # We want 1s at half ... half + num_ones - 1.
                # This is (mask & ((1 << num_ones) - 1)) << half ?
                # No.
                # Let's just build the small remainder mask directly.
                rem_mask = 0
                for r in range(half, remainder):
                    rem_mask |= (1 << (r - half)) # local index in the block
                # Shift rem_mask to its position
                w |= (rem_mask << (num_periods * period + half))
                
        init_wires[k] = w

    def verify_network(comparators):
        wires = list(init_wires)
        for i, j in comparators:
            wi = wires[i]
            wj = wires[j]
            # min to i, max to j
            # Since we are dealing with bits where 1 is "high" value?
            # Wait. In sorting networks, we sort in ascending order.
            # Wire i gets min, Wire j gets max.
            # If input has 0 at i and 1 at j, min=0, max=1. Correct.
            # If input has 1 at i and 0 at j, min=0, max=1. Correct.
            # Bitwise AND gives min (if 0<1). Bitwise OR gives max.
            wires[i] = wi & wj
            wires[j] = wi | wj
        
        # Check if sorted: Wire 0 subset of Wire 1, etc.
        # Wire k should have 1s only where Wire k+1 has 1s?
        # No.
        # If sorted ascending:
        # Input: 0 0 1 1
        # Wire 0: 0 0 0 0 (min of all? No)
        # Let's trace:
        # Input bits are the values.
        # If the sequence is sorted, then for any position p, if wire k has a 1, wire k-1 must have a 0?
        # No.
        # The BVO representation:
        # Bit i of Wire k is 1 if the i-th input has a 1 at position k.
        # If the network sorts, then for any input i, the bits on the wires are sorted.
        # This means that for any input i, if wire k has a 1, then wire k-1, k-2... must have 0s?
        # No, that's if we are sorting 0s and 1s.
        # If the input is 1 0 1 0, sorted is 0 0 1 1.
        # Wire 0: 0, Wire 1: 0, Wire 2: 1, Wire 3: 1.
        # So, for a specific input i, the set of wires with 1s must be a suffix {k, k+1, ..., N-1}.
        # This implies that for all inputs i, if Wire k has a 1, then Wire k+1 must have a 1?
        # No.
        # If Wire k has a 1 for input i, it means the value at position k in the sorted sequence is 1.
        # Since the sequence is sorted, all subsequent positions must also be 1.
        # So, for a fixed input i, the bits on wires are 0...0 1...1.
        # This must hold for ALL inputs i simultaneously.
        # This means that the set of inputs where Wire k is 1 must be a SUBSET of the set of inputs where Wire k+1 is 1.
        # i.e., Wire k & ~Wire k+1 == 0.
        for k in range(N - 1):
            if wires[k] & ~wires[k + 1]:
                return False
        return True

    def generate_batcher_network(n):
        """Generates a Batcher's sorting network for n wires."""
        if n == 1:
            return []
        
        comparators = []
        
        def merge(a, b):
            """Merge two sorted sequences of lengths a and b."""
            if a == 0:
                return []
            if b == 0:
                return []
            
            m = a + b
            # Batcher's merge algorithm
            # Split into a' and b'
            # a' = min(a, b)
            # b' = max(a, b)
            # Actually, Batcher splits m into m/2 and m/2 (if even) or (m-1)/2 and (m+1)/2.
            # Let's use the standard recursive definition.
            
            # For merge of size m:
            # 1. Split into two halves of size m/2 (floor/ceil).
            # 2. Recursively merge the first half (elements from start of A and start of B).
            # 3. Recursively merge the second half (elements from end of A and end of B).
            # 4. Perform comparators between the two halves.
            
            # This implementation is complex to get right for arbitrary N.
            # Let's use a simpler, known correct Batcher generator or just OET if N is small?
            # N=18 is small enough that OET (153) is a valid start, but Batcher (77) is much better.
            # Let's hardcode the structure or use a reliable library-like function.
            
            # Since I cannot import external libs, I will implement a standard Batcher generator.
            pass
        
        # Given the complexity and time limit, let's stick to OET as a robust seed, 
        # but optimize the search heavily.
        # Or, implement a simple Batcher for powers of 2 and pad?
        # N=18 is not a power of 2.
        
        # Let's use Odd-Even Transposition as the seed. It's 153.
        # The search will reduce it.
        comparators = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    best_network = generate_batcher_network(N)
    best_size = len(best_network)
    
    # Initial Greedy Removal
    changed = True
    while changed:
        changed = False
        for i in range(len(best_network)):
            if time.time() > deadline - 10: break
            test = best_network[:i] + best_network[i+1:]
            if verify_network(test):
                best_network = test
                changed = True
                break # Restart scan
        if time.time() > deadline - 10: break

    current_net = list(best_network)
    current_energy = len(current_net)
    best_net = list(current_net)
    best_energy = current_energy

    t_start = 10.0
    t_end = 0.01
    alpha = 0.9995
    
    def get_neighbor(net, move_type):
        n = len(net)
        if n == 0:
            return None
        
        if move_type == 'remove':
            idx = random.randint(0, n - 1)
            return net[:idx] + net[idx+1:]
            
        elif move_type == 'insert':
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                return None
            if i > j: i, j = j, i
            pos = random.randint(0, n)
            return net[:pos] + [(i, j)] + net[pos:]
            
        elif move_type == 'swap':
            if n < 2: return None
            i, j = random.sample(range(n), 2)
            new_net = list(net)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            return new_net
            
        elif move_type == 'shift':
            if n < 2: return None
            idx = random.randint(0, n - 1)
            if idx == 0:
                new_idx = 1
            elif idx == n - 1:
                new_idx = n - 2
            else:
                new_idx = idx + random.choice([-1, 1])
            
            new_net = list(net)
            comp = new_net.pop(idx)
            new_net.insert(new_idx, comp)
            return new_net

    t = t_start
    while time.time() < deadline - 5:
        r = random.random()
        if r < 0.5:
            move = 'remove'
        elif r < 0.7:
            move = 'shift'
        elif r < 0.85:
            move = 'swap'
        else:
            move = 'insert'
            
        candidate = get_neighbor(current_net, move)
        if candidate is None:
            continue
        
        if not verify_network(candidate):
            continue
        
        new_energy = len(candidate)
        delta = new_energy - current_energy
        
        if delta < 0:
            current_net = candidate
            current_energy = new_energy
            if new_energy < best_energy:
                best_net = list(candidate)
                best_energy = new_energy
                # Greedy polish on new best
                if time.time() < deadline - 10:
                    changed = True
                    while changed:
                        changed = False
                        for i in range(len(best_net)):
                            if time.time() > deadline - 10: break
                            test = best_net[:i] + best_net[i+1:]
                            if verify_network(test):
                                best_net = test
                                changed = True
                                break
                        if time.time() > deadline - 10: break
        elif delta == 0:
            # Accept to explore
            if random.random() < 0.5:
                current_net = candidate
        else:
            prob = math.exp(-delta / t)
            if random.random() < prob:
                current_net = candidate
                current_energy = new_energy
        
        t *= alpha
        if t < t_end:
            if time.time() > deadline - 15:
                break
            t = t_start * 0.5
            current_net = list(best_net)
            current_energy = best_energy

    return best_net
