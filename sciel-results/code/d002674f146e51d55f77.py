import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Precompute all 4-APs as tuples of indices (0-indexed)
    four_aps = []
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute all 3-APs as tuples of indices (0-indexed)
    three_aps = []
    for a in range(N):
        max_d = (N - 1 - a) // 2
        for d in range(1, max_d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    def count_3aps_from_bitset(bits):
        """Count 3-APs given a set of 0-indexed elements."""
        count = 0
        for (a, b, c) in three_aps:
            if bits[a] and bits[b] and bits[c]:
                count += 1
        return count
    
    def is_4free_from_bitset(bits):
        """Check if bitset represents a 4-AP-free set."""
        for (a, b, c, d) in four_aps:
            if bits[a] and bits[b] and bits[c] and bits[d]:
                return False
        return True
    
    # Generate initial 4-free set using greedy
    bits = [0] * N
    for x in range(N):
        valid = True
        for (a, b, c, d) in four_aps:
            if x in (a, b, c, d):
                others = [v for v in (a, b, c, d) if v != x]
                if all(bits[v] for v in others):
                    valid = False
                    break
        if valid:
            bits[x] = 1
    
    current_count = count_3aps_from_bitset(bits)
    best_bits = bits[:]
    best_count = current_count
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline and T > T_min:
        # Pick a random move: add, remove, or swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Try adding a random element
            empty = [i for i in range(N) if not bits[i]]
            if not empty:
                T *= cooling_rate
                continue
            y = random.choice(empty)
            new_bits = bits[:]
            new_bits[y] = 1
            if not is_4free_from_bitset(new_bits):
                T *= cooling_rate
                continue
            new_count = count_3aps_from_bitset(new_bits)
            delta = new_count - current_count
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                bits = new_bits
                current_count = new_count
                if current_count > best_count:
                    best_bits = bits[:]
                    best_count = current_count
        elif move_type == 'remove':
            # Try removing a random element
            filled = [i for i in range(N) if bits[i]]
            if not filled:
                T *= cooling_rate
                continue
            x = random.choice(filled)
            new_bits = bits[:]
            new_bits[x] = 0
            # Removing always keeps 4-free
            new_count = count_3aps_from_bitset(new_bits)
            delta = new_count - current_count
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                bits = new_bits
                current_count = new_count
                if current_count > best_count:
                    best_bits = bits[:]
                    best_count = current_count
        else:  # swap
            filled = [i for i in range(N) if bits[i]]
            empty = [i for i in range(N) if not bits[i]]
            if not filled or not empty:
                T *= cooling_rate
                continue
            x = random.choice(filled)
            y = random.choice(empty)
            new_bits = bits[:]
            new_bits[x] = 0
            new_bits[y] = 1
            if not is_4free_from_bitset(new_bits):
                T *= cooling_rate
                continue
            new_count = count_3aps_from_bitset(new_bits)
            delta = new_count - current_count
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                bits = new_bits
                current_count = new_count
                if current_count > best_count:
                    best_bits = bits[:]
                    best_count = current_count
        
        T *= cooling_rate
    
    # Convert best_bits back to 1-indexed set
    result = set()
    for i in range(N):
        if best_bits[i]:
            result.add(i + 1)
    
    return sorted(result)
