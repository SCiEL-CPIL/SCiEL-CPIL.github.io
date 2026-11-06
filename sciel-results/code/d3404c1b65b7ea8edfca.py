import time
import random
import math

def _score_bitset(bits):
    """
    Count distinct d > 0 such that a, a+d, a+2d are all in the set.
    bits is an integer where bit i represents presence of i+1.
    """
    count = 0
    # Max possible d: if a=1, a+2d <= 512 => 2d <= 511 => d <= 255
    # We iterate d from 1 to 255
    for d in range(1, 256):
        # Check if there is any a such that a, a+d, a+2d are in set
        # This is true if (bits & (bits >> d) & (bits >> (2*d))) != 0
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    return count

def _make_bits(arr):
    """Convert a list of integers (1..512) to a bitmask."""
    bits = 0
    for x in arr:
        bits |= (1 << (x - 1))
    return bits

def _bits_to_arr(bits):
    """Convert bitmask back to a sorted list of integers."""
    arr = []
    # Iterate through 512 bits
    for i in range(512):
        if (bits >> i) & 1:
            arr.append(i + 1)
    return arr

def _random_initial(rng):
    """Generate a random initial set of 32 distinct integers in 1..512."""
    return sorted(rng.sample(range(1, 513), 32))

def _dense_initial(rng):
    """Generate a consecutive block shifted randomly."""
    start = rng.randint(1, 481)
    return list(range(start, start + 32))

def _greedy_initial(rng):
    """
    Construct a set by greedily adding elements that maximize the score increment.
    Start with a random element, then add elements one by one that give the highest score gain.
    """
    # Start with a random seed element
    start_val = rng.randint(1, 512)
    bits = 1 << (start_val - 1)
    current_score = 0
    
    # We need to add 31 more elements
    # To make it faster, we don't re-evaluate all 512 candidates fully from scratch every time 
    # but we can sample or just do a limited greedy pass.
    # For simplicity and robustness, we'll just do a simple random restart if this is too slow, 
    # but a greedy build is often a good start.
    
    # Actually, a simple dense block is a very strong competitor for small differences.
    # Let's stick to dense and random for initializations to keep it fast, 
    # as the hill climbing will do the heavy lifting.
    pass

def _hill_climb(bits, deadline, rng):
    """
    Perform first-improvement hill climbing on the bitmask.
    Returns the best bits and score found.
    """
    cur_bits = bits
    cur_score = _score_bitset(cur_bits)
    best_bits = cur_bits
    best_score = cur_score
    
    # Precompute set positions for faster access
    set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
    
    # We will try to improve the score by swapping elements.
    # To avoid getting stuck, we allow "sideways" moves if we haven't improved for a while?
    # No, strict first improvement is usually better for this metric if we have enough restarts.
    # However, since the landscape is rugged, we might need to accept worse moves sometimes.
    # Let's try a simple hill climb first. If it stalls, we break and restart.
    
    stall_count = 0
    max_stall = 2000  # Number of failed attempts to improve before giving up on this local search
    
    while time.time() < deadline:
        improved = False
        # Try to find an improving move
        # We iterate through the current elements and try to swap them with random unused elements
        # Or we can try all possible swaps? 32 * 480 is ~15k checks. 
        # Each check is O(255) bit ops. 15k * 255 is ~4M ops. This is fast in Python? 
        # Bit ops are C-speed, so yes, this is feasible.
        
        # Let's try a systematic local search: 
        # For each element in the set, try replacing it with every possible other value.
        # This is 32 * 511 = ~16k trials per full pass.
        
        # To speed up, we can just pick a random element to replace and a random new value.
        # But systematic is more thorough.
        
        # Let's do a hybrid: 
        # 1. Try random swaps (fast exploration)
        # 2. If no improvement for a while, do a systematic pass (thorough exploitation)
        
        if stall_count < 100:
            # Random exploration
            if not set_positions:
                break
            idx = rng.randrange(len(set_positions))
            old_pos = set_positions[idx]
            old_val = old_pos + 1
            
            # Pick a random new value
            new_val = rng.randint(1, 512)
            if new_val == old_val:
                continue
            new_pos = new_val - 1
            
            if (cur_bits >> new_pos) & 1:
                continue
                
            trial_bits = (cur_bits & ~(1 << old_pos)) | (1 << new_pos)
            trial_score = _score_bitset(trial_bits)
            
            if trial_score > cur_score:
                cur_bits = trial_bits
                cur_score = trial_score
                # Update set_positions
                set_positions.remove(old_pos)
                set_positions.append(new_pos)
                set_positions.sort() # Keep it sorted for consistency, though not strictly needed for random pick
                improved = True
                stall_count = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_bits = cur_bits
            else:
                stall_count += 1
        else:
            # Systematic exploitation: try swapping each element with a few nearby values?
            # Or just full random again but reset stall count?
            # Let's try swapping one element with a random one, but if we fail 2000 times, 
            # it's likely a local optimum. We should break out and let the outer loop restart.
            break

        if improved:
            continue
        
    return best_bits, best_score

def _sa(bits, deadline, rng):
    """
    Simulated Annealing to escape local optima.
    """
    cur_bits = bits
    cur_score = _score_bitset(cur_bits)
    best_bits = cur_bits
    best_score = cur_score
    
    T = 10.0
    T_min = 0.1
    alpha = 0.995
    
    set_positions = [i for i in range(512) if (cur_bits >> i) & 1]
    
    while time.time() < deadline and T > T_min:
        # Generate a move
        # Random swap
        idx = rng.randrange(len(set_positions))
        old_pos = set_positions[idx]
        old_val = old_pos + 1
        
        new_val = rng.randint(1, 512)
        if new_val == old_val:
            continue
        new_pos = new_val - 1
        
        if (cur_bits >> new_pos) & 1:
            continue
            
        trial_bits = (cur_bits & ~(1 << old_pos)) | (1 << new_pos)
        trial_score = _score_bitset(trial_bits)
        
        delta = trial_score - cur_score
        
        if delta >= 0:
            accept = True
        else:
            # P = exp(delta / T)
            # delta is negative integer
            # To avoid underflow/overflow issues, we can just use rng.random()
            if T < 0.01:
                accept = False
            else:
                # log P = delta / T
                # Since delta is small negative int and T is float, this is safe
                accept = rng.random() < math.exp(delta / T)
        
        if accept:
            cur_bits = trial_bits
            cur_score = trial_score
            set_positions.remove(old_pos)
            set_positions.append(new_pos)
            set_positions.sort()
            
            if cur_score > best_score:
                best_score = cur_score
                best_bits = cur_bits
                
        T *= alpha
        
    return best_bits, best_score

def construct():
    start_time = time.time()
    # 150s limit, use 145s for safety
    deadline = start_time + 145.0
    
    rng = random.Random(12345)
    
    best_overall_bits = None
    best_overall_score = 0
    
    # We will run a series of restarts.
    # Each restart: Initialize -> Hill Climb -> (Optional SA if time permits)
    
    # Initializations to try
    init_strategies = [
        lambda r: _dense_initial(r),
        lambda r: _random_initial(r),
        lambda r: _dense_initial(r),
        lambda r: _random_initial(r),
        lambda r: _dense_initial(r),
        lambda r: _random_initial(r),
    ]
    
    # We'll cycle through these strategies
    strategy_idx = 0
    
    while time.time() < deadline - 5.0:
        # Get initial bits
        init_func = init_strategies[strategy_idx % len(init_strategies)]
        strategy_idx += 1
        
        init_arr = init_func(rng)
        bits = _make_bits(init_arr)
        
        # Phase 1: Hill Climbing (Fast, exploits local structure)
        # Give it a portion of the remaining time, e.g., 2 seconds or until stall
        hc_deadline = min(deadline - 2.0, start_time + 3.0 + (strategy_idx * 0.5))
        # Actually, let's just run HC until it stalls or hits a short time limit
        hc_start = time.time()
        hc_deadline_local = hc_start + 2.0 # 2 seconds per restart for HC
        
        bits, score = _hill_climb(bits, hc_deadline_local, rng)
        
        # Phase 2: Simulated Annealing (if time remains and we haven't found a perfect score)
        # Max possible score is 255 (all d from 1 to 255). 
        # If score is high, we might skip SA to save time for other restarts.
        # But SA can find better structures.
        if time.time() < deadline - 5.0 and score < 255:
            sa_deadline = deadline - 5.0
            # Allocate remaining time to SA for this restart? 
            # No, we want multiple restarts. 
            # Let's cap SA time per restart.
            sa_deadline_local = min(sa_deadline, time.time() + 5.0)
            bits, score = _sa(bits, sa_deadline_local, rng)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_bits = bits
            
    # Final Polish: Hill Climb on the best solution found with remaining time
    if best_overall_bits is None:
        # Fallback
        best_overall_bits = _make_bits(list(range(1, 33)))
        best_overall_score = _score_bitset(best_overall_bits)
    else:
        if time.time() < deadline - 1.0:
            # Do a final hill climb on the global best
            final_deadline = deadline - 1.0
            best_overall_bits, best_overall_score = _hill_climb(best_overall_bits, final_deadline, rng)
            
    result = _bits_to_arr(best_overall_bits)
    
    # Ensure validity
    if len(result) != 32 or any(x < 1 or x > 512 for x in result) or len(set(result)) != 32:
        result = list(range(1, 33))
        
    return result
