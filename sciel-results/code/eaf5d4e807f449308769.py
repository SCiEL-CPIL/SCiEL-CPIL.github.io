import time
import random
import math

def _score_bitset(bits):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d are all set in 'bits'.
    'bits' is a Python integer where bit i is 1 if i is in the set.
    We assume the set contains values in 1..512, so bits 0 is unused or irrelevant.
    """
    if bits == 0:
        return 0
    
    # Max possible d is when a=1, a+2d=512 => 2d=511 => d=255
    # But to be safe and general, we can check up to 256.
    # Actually, if max_val is M, max d is (M-1)//2.
    # Since range is 1..512, max d is 255.
    
    count = 0
    # We iterate d from 1 to 255.
    # For a given d, we want to know if there exists a such that 
    # bit[a] & bit[a+d] & bit[a+2d] is 1.
    # This is equivalent to: bits & (bits >> d) & (bits >> 2*d) != 0
    
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
            
    return count

def _set_to_bits(s):
    bits = 0
    for x in s:
        bits |= (1 << x)
    return bits

def _bits_to_set(bits):
    s = set()
    # We only care about 1..512
    for i in range(1, 513):
        if (bits >> i) & 1:
            s.add(i)
    return s

def _greedy_construct():
    """
    Construct a set of 32 integers by greedily adding the element that maximizes
    the number of distinct differences.
    """
    s = set()
    bits = 0
    # Start with 1? Or just start empty.
    # Let's try starting with a few random seeds or just 1.
    # Actually, starting with 1 is a good anchor.
    s.add(1)
    bits |= (1 << 1)
    
    current_score = 0 # Score of empty/1 set is 0
    
    for _ in range(31):
        best_elem = None
        best_score = -1
        
        # Try all possible candidates
        # Optimization: We can skip candidates that are obviously bad?
        # For now, try all 1..512 not in s.
        for x in range(1, 513):
            if x in s:
                continue
            
            new_bits = bits | (1 << x)
            score = _score_bitset(new_bits)
            
            if score > best_score:
                best_score = score
                best_elem = x
        
        if best_elem is not None:
            s.add(best_elem)
            bits |= (1 << best_elem)
            current_score = best_score
        else:
            # Fallback if something goes wrong (shouldn't happen)
            x = random.randint(1, 512)
            while x in s:
                x = random.randint(1, 512)
            s.add(x)
            bits |= (1 << x)
            
    return s, bits

def _local_search_fast(bits, s, deadline, seed):
    """
    Fast local search using bitset scoring.
    """
    random.seed(seed)
    current_bits = bits
    current_score = _score_bitset(current_bits)
    best_bits = current_bits
    best_score = current_score
    current_set = _bits_to_set(current_bits)
    
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Generate a move: 1-swap
        # Pick an element to remove
        if not current_set:
            break
            
        to_remove = random.choice(list(current_set))
        
        # Pick a new element to add
        new_elem = random.randint(1, 512)
        while new_elem in current_set:
            new_elem = random.randint(1, 512)
            
        # Apply move
        current_set.remove(to_remove)
        current_set.add(new_elem)
        
        new_bits = _set_to_bits(current_set)
        new_score = _score_bitset(new_bits)
        
        # Acceptance
        if new_score >= current_score:
            current_score = new_score
            current_bits = new_bits
            if new_score > best_score:
                best_score = new_score
                best_bits = new_bits
        else:
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current_score = new_score
                current_bits = new_bits
                
        T *= cooling
        
        # If we hit a high score, we can tighten the cooling or just continue
        # To save time, if we've been stuck for a while, we might want to restart or cool faster
        # But simple SA is robust enough for this short duration.
        
    return _bits_to_set(best_bits), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for 150s limit
    
    best_set = None
    best_score = -1
    
    # 1. Greedy Construction
    greedy_set, greedy_bits = _greedy_construct()
    greedy_score = _score_bitset(greedy_bits)
    
    if greedy_score > best_score:
        best_score = greedy_score
        best_set = greedy_set
        
    # 2. Local Search from Greedy
    if time.time() < deadline - 10:
        ls_set, ls_score = _local_search_fast(greedy_bits, greedy_set, time.time() + 20, 123)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set

    # 3. Diverse Initializations + Local Search
    # We have ~120s left. Let's run a few more diverse starts.
    # Time budget per run: ~20s
    
    initializations = []
    
    # Random
    initializations.append(set(random.sample(range(1, 513), 32)))
    
    # Two blocks
    initializations.append(set(list(range(1, 17)) + list(range(257, 273))))
    
    # Four blocks
    initializations.append(set(list(range(1, 9)) + list(range(129, 137)) + list(range(257, 265)) + list(range(385, 393))))
    
    # Geometric-ish (powers of 2 scaled)
    geo_set = set()
    val = 1
    while len(geo_set) < 32 and val <= 512:
        geo_set.add(val)
        val = int(val * 1.2) + 1 # Ensure distinct and increasing
    # Fill rest with random
    while len(geo_set) < 32:
        x = random.randint(1, 512)
        if x not in geo_set:
            geo_set.add(x)
    initializations.append(geo_set)
    
    for i, init_set in enumerate(initializations):
        if time.time() > deadline - 5:
            break
            
        # Ensure valid set
        init_list = list(init_set)
        if len(init_list) < 32:
            while len(init_list) < 32:
                x = random.randint(1, 512)
                if x not in init_list:
                    init_list.append(x)
        else:
            init_list = init_list[:32]
            
        init_set = set(init_list)
        init_bits = _set_to_bits(init_set)
        
        # Quick local search
        run_deadline = min(time.time() + 20, deadline - 2)
        ls_set, ls_score = _local_search_fast(init_bits, init_set, run_deadline, 1000 + i)
        
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set

    # 4. If time remains, do a final polish on the best set found
    if time.time() < deadline - 2 and best_set:
        best_bits = _set_to_bits(best_set)
        final_set, final_score = _local_search_fast(best_bits, best_set, deadline - 1, 999)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set

    if best_set is None:
        return list(range(1, 33))
        
    return list(best_set)
