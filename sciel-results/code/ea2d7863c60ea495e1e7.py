"""Optimized construction for 32 integers in 1..512 maximizing distinct 3-term AP differences.
Uses bitset representation for O(1) bitwise scoring and fast swaps."""
import time
import random

# EVOLVE-BLOCK-START

def _score_bitset(b):
    """Count distinct d > 0 such that a, a+d, a+2d all in set represented by bitset b.
    b is an integer where bit i is 1 if (i+1) is in the set."""
    count = 0
    # Max difference d is at most (512-1)/2 = 255.5 -> 255
    # Actually if a=1, a+2d <= 512 => 2d <= 511 => d <= 255.
    # So d in 1..255.
    for d in range(1, 256):
        # Check if there exists a such that bits a-1, a+d-1, a+2d-1 are all set.
        # b & (b >> d) & (b >> 2*d) will have bit a-1 set if such a exists.
        if (b & (b >> d) & (b >> (2 * d))):
            count += 1
    return count

def _greedy_init(rng):
    """Greedy initialization using bitsets."""
    current_bit = 0
    current_set = []
    current_score = 0
    
    # Candidates 1..512
    candidates = list(range(1, 513))
    rng.shuffle(candidates)
    
    # To speed up greedy, we can sample a subset of candidates at each step
    # or just iterate through all if it's fast enough.
    # 32 steps * 512 candidates * score_cost.
    # score_cost is ~256 bitwise ops.
    # 32 * 512 * 256 ~ 4M ops. Should be < 0.1s in Python C-optimized bitwise ops.
    
    for _ in range(32):
        best_elem = None
        best_gain = -1
        best_bit = current_bit
        best_score = current_score
        
        # Sample a subset of remaining candidates to speed up, or use all if time permits
        # Let's use all for quality, but break early if time is tight? 
        # No, just use all. It's fast enough.
        
        # Optimization: Instead of scoring every candidate from scratch,
        # we could try to estimate gain, but exact is better for init.
        
        # To avoid O(512) full scores if we can prune:
        # Not easy to prune without complex logic.
        
        # We will iterate through a random subset of size 50 to find a good candidate,
        # then maybe refine? 
        # Actually, for 32 elements, greedy with full scan is fine.
        
        # Let's try a random sample of 100 candidates to find the best one.
        # This reduces cost by 5x.
        sample_size = min(100, len(candidates))
        sample = rng.sample(candidates, sample_size)
        
        for elem in sample:
            new_bit = current_bit | (1 << (elem - 1))
            new_score = _score_bitset(new_bit)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_elem = elem
                best_bit = new_bit
                best_score = new_score
        
        if best_elem is not None:
            current_set.append(best_elem)
            current_bit = best_bit
            current_score = best_score
            candidates.remove(best_elem)
        else:
            # Fallback: pick random
            elem = candidates.pop()
            current_set.append(elem)
            current_bit |= (1 << (elem - 1))
            current_score = _score_bitset(current_bit)
            
    return current_set, current_bit, current_score

def _simulated_annealing(current_set, current_bit, current_score, rng, deadline):
    """Simulated annealing with bitset updates."""
    best_set = list(current_set)
    best_bit = current_bit
    best_score = current_score
    
    temperature = 50.0
    cooling = 0.99999
    min_temp = 0.01
    e = 2.718281828459045
    
    while temperature > min_temp:
        if time.time() > deadline:
            break
        
        # Choose move type
        move_type = rng.random()
        
        new_bit = current_bit
        new_score = -1
        valid_move = False
        indices_to_update = []
        new_vals = []
        
        if move_type < 0.5:
            # Single swap
            i = rng.randrange(32)
            old_elem = current_set[i]
            new_elem = rng.randint(1, 512)
            if new_elem == old_elem or new_elem in current_set:
                continue
            
            new_bit = (current_bit & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
            new_score = _score_bitset(new_bit)
            if new_score >= 0:
                valid_move = True
                indices_to_update = [i]
                new_vals = [new_elem]
                
        elif move_type < 0.8:
            # Double swap
            i, j = rng.sample(range(32), 2)
            old1, old2 = current_set[i], current_set[j]
            new1, new2 = rng.sample(range(1, 513), 2)
            # Check if new1, new2 are in set or equal to old1/old2 (handled by set check below)
            # We need to ensure new1, new2 are not in the current set (excluding old1, old2)
            # And new1 != new2
            
            # Simple check:
            if new1 == new2:
                continue
            if new1 in current_set or new2 in current_set:
                # If new1 is old1 or old2, it's effectively a swap, but we are removing them.
                # We just need to ensure the final set is valid.
                # Let's construct the new bit and check if it has exactly 32 bits?
                # No, simpler: check if new1 or new2 is in the set of *other* elements.
                # Or just construct and verify bit count.
                pass
            
            # Construct new bit
            new_bit = current_bit
            new_bit &= ~(1 << (old1 - 1))
            new_bit &= ~(1 << (old2 - 1))
            new_bit |= (1 << (new1 - 1))
            new_bit |= (1 << (new2 - 1))
            
            # Verify we have 32 distinct elements
            # If new1 or new2 was in the set (and not old1/old2), we might have < 32 or > 32?
            # If new1 was in set, and not old1/old2, then we added a bit that was already set?
            # No, we removed old1/old2. If new1 was in set (say old3), then bit new1-1 is still set.
            # We set it again. No change in bit count for that bit.
            # But we want distinct integers.
            # If new1 is in {old3..old32}, then the set of values changes, but the bit representation
            # might not reflect the "distinctness" if we just OR.
            # Wait, `new_bit` is a set of bits.
            # If new1 is already in `current_bit` (and not old1/old2), then `new_bit` will have that bit set.
            # The total number of set bits in `new_bit` might be 31 (if new1 was old3, we removed old1, old2, added new1, new2.
            # Bits removed: old1, old2. Bits added: new1, new2.
            # If new1 == old3, bit old3 was set, remains set.
            # So we have 32 - 2 + 2 = 32 bits?
            # Let's check:
            # Start: 32 bits.
            # Remove old1, old2: 30 bits.
            # Add new1, new2.
            # If new1 is in the remaining 30 bits, adding it doesn't increase count.
            # So count could be 31 or 32.
            # We need exactly 32 distinct integers.
            # So we must ensure new1 and new2 are NOT in the current set (excluding old1, old2).
            
            # Efficient check:
            # Create a mask of current set without old1, old2
            # mask = current_bit & ~(1 << (old1-1)) & ~(1 << (old2-1))
            # if (mask >> (new1-1)) & 1: invalid
            # if (mask >> (new2-1)) & 1: invalid
            
            mask = current_bit & ~(1 << (old1 - 1)) & ~(1 << (old2 - 1))
            if ((mask >> (new1 - 1)) & 1) or ((mask >> (new2 - 1)) & 1):
                continue
            
            new_score = _score_bitset(new_bit)
            if new_score >= 0:
                valid_move = True
                indices_to_update = [i, j]
                new_vals = [new1, new2]
                
        else:
            # Triple swap
            i, j, k = rng.sample(range(32), 3)
            old1, old2, old3 = current_set[i], current_set[j], current_set[k]
            new1, new2, new3 = rng.sample(range(1, 513), 3)
            
            mask = current_bit
            mask &= ~(1 << (old1 - 1))
            mask &= ~(1 << (old2 - 1))
            mask &= ~(1 << (old3 - 1))
            
            if ((mask >> (new1 - 1)) & 1) or ((mask >> (new2 - 1)) & 1) or ((mask >> (new3 - 1)) & 1):
                continue
            
            new_bit = mask
            new_bit |= (1 << (new1 - 1))
            new_bit |= (1 << (new2 - 1))
            new_bit |= (1 << (new3 - 1))
            
            new_score = _score_bitset(new_bit)
            if new_score >= 0:
                valid_move = True
                indices_to_update = [i, j, k]
                new_vals = [new1, new2, new3]
        
        if not valid_move:
            temperature *= cooling
            continue
        
        delta = new_score - current_score
        if delta > 0 or rng.random() < (e ** (delta / temperature)):
            # Accept move
            for idx, val in zip(indices_to_update, new_vals):
                current_set[idx] = val
            current_bit = new_bit
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                best_bit = current_bit
        
        temperature *= cooling
    
    return best_set, best_bit, best_score

def _polish(current_set, current_bit, current_score, deadline):
    """Local search: try all single swaps to improve score."""
    best_set = list(current_set)
    best_bit = current_bit
    best_score = current_score
    
    # Iterate a few times
    for _ in range(3):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(32):
            if time.time() > deadline:
                break
            
            old_elem = best_set[i]
            # Try replacing with any other element
            # To speed up, we can iterate through all 1..512 and check if in set
            # This is 512 checks.
            
            # Optimization: Precompute list of available elements?
            # Just loop 1..512
            
            for new_elem in range(1, 513):
                if new_elem == old_elem:
                    continue
                if new_elem in best_set:
                    continue
                
                new_bit = (best_bit & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
                new_score = _score_bitset(new_bit)
                
                if new_score > best_score:
                    best_set[i] = new_elem
                    best_bit = new_bit
                    best_score = new_score
                    improved = True
                    break # Restart outer loop? Or continue?
                    # Breaking inner loop, but outer loop continues.
                    # Since best_set changed, we should probably restart the i loop
                    # or just continue with the new state.
                    # Continuing is fine, might find more improvements.
        
        if not improved:
            break
            
    return best_set, best_bit, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = random.Random(42)
    
    global_best_set = None
    global_best_bit = 0
    global_best_score = -1
    
    # Multi-restart
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        # Different seeds for variety
        rng.seed(42 + restarts)
        
        # 1. Greedy Init
        cur_set, cur_bit, cur_score = _greedy_init(rng)
        
        if cur_score > global_best_score:
            global_best_score = cur_score
            global_best_set = cur_set[:]
            global_best_bit = cur_bit
        
        # 2. Simulated Annealing
        # Allocate time based on restart number? 
        # Just run until time is tight.
        # Give SA ~50% of remaining time?
        # Or just run SA for a fixed number of steps or time slice.
        
        # Let's run SA for a slice of time
        sa_deadline = min(deadline - 2.0, time.time() + 10.0)
        
        if time.time() < sa_deadline:
            cur_set, cur_bit, cur_score = _simulated_annealing(cur_set, cur_bit, cur_score, rng, sa_deadline)
            
            if cur_score > global_best_score:
                global_best_score = cur_score
                global_best_set = cur_set[:]
                global_best_bit = cur_bit
        
        # 3. Polishing
        # Polish the best set found so far? Or the current one?
        # Polish the current one, then update global if better.
        # Also polish the global best?
        
        if time.time() < deadline - 1.0:
            pol_set, pol_bit, pol_score = _polish(cur_set, cur_bit, cur_score, deadline - 1.0)
            
            if pol_score > global_best_score:
                global_best_score = pol_score
                global_best_set = pol_set[:]
                global_best_bit = pol_bit
        
        # Additionally, polish the global best if we have time
        if global_best_set and time.time() < deadline - 0.5:
            pol_set, pol_bit, pol_score = _polish(global_best_set, global_best_bit, global_best_score, deadline - 0.5)
            if pol_score > global_best_score:
                global_best_score = pol_score
                global_best_set = pol_set[:]
                global_best_bit = pol_bit

    if global_best_set is None:
        return list(range(1, 33))
    
    return global_best_set

# EVOLVE-BLOCK-END
