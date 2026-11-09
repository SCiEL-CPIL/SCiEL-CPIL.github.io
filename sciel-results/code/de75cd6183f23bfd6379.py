import time
import random
import sys

def construct():
    """
    Returns exactly 32 distinct integers in 1..512.
    Maximizes the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512
    
    # Helper to convert a bitmask (int) to a list of elements
    # Bit i represents integer i (1-indexed in our problem, so bit 0 is unused or represents 1? 
    # Let's use bit i to represent integer i. So bit 1 is 1, bit 512 is 512.
    # Python ints are arbitrary precision, so this is fine.
    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        """
        # To be fast, we can iterate over d and check if there exists an a.
        # a, a+d, a+2d in set.
        # This is equivalent to: (mask & (mask >> d) & (mask >> (2*d))) != 0
        # We need to count distinct d.
        
        valid_diffs = 0
        # Max possible d is such that a+2d <= 512. Since a>=1, 2d <= 511 => d <= 255.
        # Actually if a=1, 1+2d <= 512 => 2d <= 511 => d <= 255.
        for d in range(1, 256):
            # Check if there is any a such that bits a, a+d, a+2d are set.
            # mask has bits for 1..512.
            # mask >> d shifts bits down. Bit a in original becomes bit a-d in shifted.
            # We want to check intersection of:
            # 1. Original mask (has a)
            # 2. mask >> d (has a-d, so if bit a-d is set in shifted, bit a+d is set in original)
            #    Wait.
            # Let S be the set.
            # We want a in S, a+d in S, a+2d in S.
            # This means bit a in M, bit a+d in M, bit a+2d in M.
            # (M >> d) has bit i set if M has bit i+d set. So (M >> d) has bit a set if M has bit a+d set.
            # (M >> 2d) has bit a set if M has bit a+2d set.
            # So we need (M & (M >> d) & (M >> 2d)) to have at least one bit set.
            # The bit set would correspond to 'a'.
            
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                valid_diffs += 1
                
        return valid_diffs

    def score_list(lst):
        """
        Score a list of 32 distinct integers.
        """
        m = list_to_mask(lst)
        return score_mask(m)

    # Initial best solution
    # Start with a random set or a heuristic set.
    # A good heuristic: spread elements out.
    # Or just random.
    # Let's try a few random starts and pick the best.
    
    best_score = -1
    best_list = None
    
    # Generate initial candidates
    # 1. Random uniform
    # 2. Structured: e.g., multiples of something, or specific patterns
    
    candidates = []
    
    # Random starts
    for _ in range(20):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        candidates.append(lst)
        
    # Structured starts:
    # Arithmetic progression: 1, 1+k, ..., 1+31k. Max k such that 1+31k <= 512. 31k <= 511 => k=16.
    # 1, 17, 33...
    candidates.append(list(range(1, 1 + 31*16, 16)))
    
    # Binary-like: 1, 2, 4, 8... up to 32 elements? 2^31 is too big.
    # Just first 32 powers of 2? 2^0..2^31. 2^31 is huge.
    # 1, 2, 4, ..., 2^31 is out of range.
    # 1, 2, 4, 8, 16, 32, 64, 128, 256. Only 9 elements.
    # So mix?
    
    # Evaluate initial candidates
    for lst in candidates:
        s = score_list(lst)
        if s > best_score:
            best_score = s
            best_list = lst[:]
            
    # Simulated Annealing
    current_list = best_list[:]
    current_score = best_score
    current_mask = list_to_mask(current_list)
    
    T = 5.0  # Initial temperature
    cooling_rate = 0.999
    
    # Precompute neighbors? No, too many.
    # Move: Pick one element, change it to another random element not in the set.
    # Or swap two elements?
    # Or add/remove? Size must be fixed at 32.
    # So swap or replace.
    # Replace: Pick index i, pick new value v not in set.
    
    while time.time() < deadline:
        # Generate a neighbor
        # Pick a random index to change
        idx = random.randint(0, N - 1)
        old_val = current_list[idx]
        
        # Pick a new value
        # To avoid duplicates, we can sample from 1..512 and check.
        # Or maintain a set of available values.
        # For simplicity, sample and check.
        # Since 512 is small, this is fast.
        
        new_val = random.randint(1, MAX_VAL)
        if new_val == old_val:
            continue
            
        # Check if new_val is already in the set (other than old_val)
        # We can use a set for O(1) lookup, but list is small (32).
        # Let's use a set for speed in the loop.
        # Actually, maintaining a set in SA loop might be overhead.
        # 32 elements, linear search is fine.
        
        if new_val in current_list:
            continue
            
        # Calculate delta score
        # Remove old_val, add new_val.
        
        # Update mask
        new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
        
        new_score = score_mask(new_mask)
        delta = new_score - current_score
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # Boltzmann factor
            # Avoid overflow
            if T < 1e-10:
                accept = False
            else:
                accept = random.random() < pow(sys.float_info.max, delta / T)
                # Better: exp(delta/T)
                # import math
                # accept = random.random() < math.exp(delta / T)
                # Using pow(sys.float_info.max, delta/T) is a common trick to avoid import math if delta is very negative?
                # No, math.exp is fine.
                # Let's just use math.exp.
                
        # Wait, I didn't import math. Let's add it or use a different method.
        # I'll add import math at the top.
        
        if accept:
            current_list[idx] = new_val
            current_score = new_score
            current_mask = new_mask
            
            if current_score > best_score:
                best_score = current_score
                best_list = current_list[:]
                
        # Cool down
        T *= cooling_rate
        
    return best_list

# Add import math
import math

# Re-define or fix the accept logic?
# I used pow(sys.float_info.max, ...) which is wrong.
# Let's rewrite the loop part properly with math.exp.

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512
    
    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def score_mask(mask):
        valid_diffs = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                valid_diffs += 1
        return valid_diffs

    def score_list(lst):
        m = list_to_mask(lst)
        return score_mask(m)

    best_score = -1
    best_list = None
    
    candidates = []
    
    # Random starts
    for _ in range(50):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        candidates.append(lst)
        
    # Structured starts
    candidates.append(list(range(1, 1 + 31*16, 16)))
    # Another structure: dense at start
    candidates.append(list(range(1, 33)))
    # Another: dense at end
    candidates.append(list(range(481, 513)))
    # Mixed: 1..16 and 497..512
    candidates.append(list(range(1, 17)) + list(range(497, 513)))

    for lst in candidates:
        s = score_list(lst)
        if s > best_score:
            best_score = s
            best_list = lst[:]
            
    current_list = best_list[:]
    current_score = best_score
    current_mask = list_to_mask(current_list)
    
    T = 5.0
    cooling_rate = 0.999
    
    # Use a set for current elements to speed up duplicate check
    current_set = set(current_list)
    
    while time.time() < deadline:
        idx = random.randint(0, N - 1)
        old_val = current_list[idx]
        
        new_val = random.randint(1, MAX_VAL)
        if new_val == old_val or new_val in current_set:
            continue
            
        new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
        new_score = score_mask(new_mask)
        delta = new_score - current_score
        
        if delta > 0:
            accept = True
        else:
            if T < 1e-10:
                accept = False
            else:
                accept = random.random() < math.exp(delta / T)
                
        if accept:
            current_list[idx] = new_val
            current_set.remove(old_val)
            current_set.add(new_val)
            current_score = new_score
            current_mask = new_mask
            
            if current_score > best_score:
                best_score = current_score
                best_list = current_list[:]
                
        T *= cooling_rate
        
    return best_list
