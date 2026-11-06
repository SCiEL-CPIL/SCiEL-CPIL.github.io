import time
import random
import math
import numpy as np

def _score_numpy(mask):
    """
    Fast scoring using numpy: count distinct d > 0 such that a, a+d, a+2d are in set.
    mask is a 512-bit integer where bit i is 1 if (i+1) is in the set.
    """
    # Convert mask to boolean array of length 512
    arr = np.zeros(512, dtype=np.uint8)
    m = mask
    for i in range(512):
        if m & 1:
            arr[i] = 1
        m >>= 1
    
    count = 0
    # For each d from 1 to 255, check if there exists a with arr[a-1]=1, arr[a+d-1]=1, arr[a+2d-1]=1
    # Equivalent to: any(arr[i] & arr[i+d] & arr[i+2d]) for i in 0..512-2d-1
    # We can vectorize over d
    for d in range(1, 256):
        # Check positions 0 to 512-2d-1
        end = 512 - 2 * d
        if end <= 0:
            break
        segment = arr[:end]
        shifted1 = arr[d:d + end]
        shifted2 = arr[2*d:2*d + end]
        if np.any(segment & shifted1 & shifted2):
            count += 1
    return count

def _score_bitmask(mask):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
    mask is an integer where bit i is 1 if (i+1) is in the set.
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = random.Random(42)
    
    # Try multiple initial structures and pick the best
    candidates = []
    
    # 1. Arithmetic Progression 1..32
    s1 = set(range(1, 33))
    candidates.append(s1)
    
    # 2. Arithmetic Progression with step 2
    s2 = set(range(1, 64, 2))
    candidates.append(s2)
    
    # 3. Arithmetic Progression with step 16
    s3 = set(1 + 16 * i for i in range(32))
    candidates.append(s3)
    
    # 4. Grid 4x8 with wider spacing
    s4 = set(1 + i * 16 + j for i in range(4) for j in range(8))
    candidates.append(s4)
    
    # 5. Random
    s5 = set(rng.sample(range(1, 513), 32))
    candidates.append(s5)
    
    # 6. Two interleaved APs
    s6 = set()
    for i in range(16):
        s6.add(1 + 2*i)
        s6.add(33 + 2*i)
    candidates.append(s6)
    
    # 7. Power-of-2 based
    s7 = set()
    for i in range(32):
        s7.add(1 + (i * 16) % 512)
    candidates.append(s7)
    
    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    best_mask = 0
    best_score = -1
    for s in candidates:
        m = set_to_mask(s)
        sc = _score_bitmask(m)
        if sc > best_score:
            best_score = sc
            best_mask = m
    
    # Multi-restart local search with numpy scoring
    overall_best_mask = best_mask
    overall_best_score = best_score
    
    num_restarts = 8
    restart_budget = 12.0  # seconds per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start from best found so far or a random perturbation
        if restart == 0:
            current_mask = overall_best_mask
            current_score = overall_best_score
        else:
            # Perturb the best solution
            current_mask = overall_best_mask
            # Remove 4-8 random elements and add new ones
            num_perturb = rng.randint(4, 8)
            for _ in range(num_perturb):
                # Remove a random set bit
                bits_set = []
                temp = current_mask
                idx = 0
                while temp:
                    if temp & 1:
                        bits_set.append(idx)
                    temp >>= 1
                    idx += 1
                if not bits_set:
                    break
                remove_idx = rng.choice(bits_set)
                current_mask &= ~(1 << remove_idx)
                
                # Add a random unset bit
                while True:
                    add_idx = rng.randint(0, 511)
                    if not (current_mask & (1 << add_idx)):
                        current_mask |= (1 << add_idx)
                        break
            
            current_score = _score_numpy(current_mask)
            if current_score > overall_best_score:
                overall_best_score = current_score
                overall_best_mask = current_mask
        
        # Local search: try single moves (remove one, add one)
        # Use numpy scoring for speed
        restart_start = time.time()
        restart_deadline = restart_start + restart_budget
        
        # Precompute list of set bits for efficient neighbor generation
        while time.time() < restart_deadline:
            # Get set bits
            bits_set = []
            temp = current_mask
            idx = 0
            while temp:
                if temp & 1:
                    bits_set.append(idx)
                temp >>= 1
                idx += 1
            
            if not bits_set:
                break
            
            # Try a batch of random moves
            batch_size = 50
            improved = False
            for _ in range(batch_size):
                remove_idx = rng.choice(bits_set)
                
                # Find an unset bit to add
                while True:
                    add_idx = rng.randint(0, 511)
                    if not (current_mask & (1 << add_idx)):
                        break
                
                new_mask = current_mask
                new_mask &= ~(1 << remove_idx)
                new_mask |= (1 << add_idx)
                
                new_score = _score_numpy(new_mask)
                
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    if current_score > overall_best_score:
                        overall_best_score = current_score
                        overall_best_mask = current_mask
                    # Update bits_set
                    bits_set = []
                    temp = current_mask
                    idx = 0
                    while temp:
                        if temp & 1:
                            bits_set.append(idx)
                        temp >>= 1
                        idx += 1
            
            if not improved and rng.random() < 0.1:
                # Occasionally try a 2-move perturbation to escape local optima
                if len(bits_set) >= 2:
                    r1, r2 = rng.sample(bits_set, 2)
                    new_mask = current_mask
                    new_mask &= ~(1 << r1)
                    new_mask &= ~(1 << r2)
                    
                    # Add two new elements
                    added = 0
                    while added < 2:
                        add_idx = rng.randint(0, 511)
                        if not (new_mask & (1 << add_idx)):
                            new_mask |= (1 << add_idx)
                            added += 1
                    
                    new_score = _score_numpy(new_mask)
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > overall_best_score:
                            overall_best_score = current_score
                            overall_best_mask = current_mask
    
    # Convert best mask back to list
    result = []
    temp = overall_best_mask
    idx = 0
    while temp:
        if temp & 1:
            result.append(idx + 1)
        temp >>= 1
        idx += 1
    
    return result
