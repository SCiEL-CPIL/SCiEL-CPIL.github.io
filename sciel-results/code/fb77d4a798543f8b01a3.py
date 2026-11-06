"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Bitset representation: x in 1..512 maps to bit (x-1)
    # S is an integer where bit i is 1 if (i+1) is in the set
    
    def bitset_from_set(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
    
    def set_from_bitset(b):
        s = []
        i = 0
        while b:
            if b & 1:
                s.append(i + 1)
            b >>= 1
            i += 1
        return s
    
    def score_bitset(b):
        count = 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    def popcount(b):
        return bin(b).count('1')
    
    # Initialize with a random set
    current_set = random.sample(range(1, 513), 32)
    current_bit = bitset_from_set(current_set)
    current_score = score_bitset(current_bit)
    
    best_set = list(current_set)
    best_bit = current_bit
    best_score = current_score
    
    # Simulated annealing
    temperature = 50.0
    cooling = 0.99995
    min_temp = 0.01
    
    rng = random.Random(42)
    
    while temperature > min_temp:
        if time.time() > deadline:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Swap: replace one element with another
            i = rng.randrange(32)
            old_elem = current_set[i]
            new_elem = rng.randint(1, 512)
            if new_elem == old_elem or new_elem in current_set:
                continue
            
            new_set = list(current_set)
            new_set[i] = new_elem
            new_bit = bitset_from_set(new_set)
            new_score = score_bitset(new_bit)
            
            delta = new_score - current_score
            if delta > 0 or rng.random() < (2.718281828 ** (delta / temperature)):
                current_set = new_set
                current_bit = new_bit
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    best_bit = current_bit
        elif move_type < 0.9:
            # Add one element and remove a different one
            new_elem = rng.randint(1, 512)
            if new_elem in current_set:
                continue
            i = rng.randrange(32)
            old_elem = current_set[i]
            
            new_set = list(current_set)
            new_set[i] = new_elem
            new_bit = bitset_from_set(new_set)
            new_score = score_bitset(new_bit)
            
            delta = new_score - current_score
            if delta > 0 or rng.random() < (2.718281828 ** (delta / temperature)):
                current_set = new_set
                current_bit = new_bit
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    best_bit = current_bit
        else:
            # Remove one element and add two different ones
            i = rng.randrange(32)
            j = rng.randrange(32)
            if i == j:
                continue
            old1 = current_set[i]
            old2 = current_set[j]
            
            new_elem1 = rng.randint(1, 512)
            new_elem2 = rng.randint(1, 512)
            if new_elem1 in current_set or new_elem2 in current_set or new_elem1 == new_elem2:
                continue
            
            new_set = list(current_set)
            new_set[i] = new_elem1
            new_set[j] = new_elem2
            new_bit = bitset_from_set(new_set)
            new_score = score_bitset(new_bit)
            
            delta = new_score - current_score
            if delta > 0 or rng.random() < (2.718281828 ** (delta / temperature)):
                current_set = new_set
                current_bit = new_bit
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    best_bit = current_bit
        
        temperature *= cooling
    
    return best_set
# EVOLVE-BLOCK-END
