import random
import time

def construct():
    MAX_SUM = 256 * 30 + 10
    
    # Precompute conflict masks
    # conflict_mask[s] has bits set at 2s, 3s, ... up to MAX_SUM
    # This allows checking if any multiple of s exists in the bitset via bitwise AND.
    conflict_masks = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        multiple = s * 2
        while multiple <= MAX_SUM:
            mask |= (1 << multiple)
            multiple += s
        conflict_masks[s] = mask

    def is_valid(bits):
        # Iterate through set bits in 'bits'
        # For each set bit s, check if bits has any bit set in conflict_masks[s]
        b = bits
        while b:
            lsb = b & -b
            s = lsb.bit_length() - 1
            if bits & conflict_masks[s]:
                return False
            b ^= lsb
        return True

    best_set = []
    best_size = 0
    candidates = list(range(1, 257))
    
    start_time = time.time()
    timeout = 55  # Leave 5s buffer
    
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        
        # Vary the strategy
        # 1. Random
        # 2. Descending
        # 3. Ascending
        # 4. Random with bias to large numbers (sort by random + size)
        
        if attempts % 4 == 0:
            cand = sorted(candidates, reverse=True)
        elif attempts % 4 == 1:
            cand = candidates[:]
            random.shuffle(cand)
        elif attempts % 4 == 2:
            cand = candidates[:]
            # Bias towards larger numbers by sorting with a random noise that favors larger
            # Or just shuffle and reverse?
            random.shuffle(cand)
            cand.sort(key=lambda x: x * random.random())
        else:
            # Try to keep some structure
            cand = candidates[:]
            random.shuffle(cand)
            
        current_set = []
        bits = 0
        
        for x in cand:
            # Calculate new bits
            # Note: bits represents sums of current_set
            # Adding x creates sums: x, and s+x for all s in bits
            new_bits = bits | (bits << x) | (1 << x)
            
            # Optimization: If new_bits has more than MAX_SUM bits, it's invalid (collision)
            # But we don't track count easily.
            # However, if (new_bits & (new_bits << 1)) is not a good check for collision.
            # Collision means two different subsets have same sum.
            # This is detected if popcount(new_bits) < popcount(bits) + popcount(bits<<x) + 1?
            # No, simply: if the number of set bits in new_bits is less than expected, there was a collision.
            # Expected = popcount(bits) + popcount(bits << x) + 1 (for x itself, unless x is in bits, which it isn't as elements are distinct and positive).
            # Actually, x is not in bits (bits are sums of subsets of previous elements).
            # So expected new bits = popcount(bits) + popcount(bits << x) + 1.
            # If popcount(new_bits) < expected, collision occurred.
            
            # Python int bit_count is fast.
            expected = bits.bit_count() + (bits << x).bit_count() + 1
            if new_bits.bit_count() < expected:
                continue
            
            if is_valid(new_bits):
                bits = new_bits
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we find a very large set, we might try to optimize it further
            # But for now, just record it.
            if best_size >= 20:
                # We can break early if we hit a theoretical high bound?
                # Max size is unknown, but 20 is very good.
                pass

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
