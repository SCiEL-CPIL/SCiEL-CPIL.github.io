import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    def is_valid_insertion(seq, x):
        """
        Check if x can be inserted into seq (sorted) at the correct position.
        Returns True if valid, and the index where it would be inserted.
        """
        # Find insertion index
        # seq is strictly increasing
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        
        # Check bounds
        if idx < len(seq) and seq[idx] == x:
            return False, -1 # Duplicate
            
        # Calculate new sums
        # Sums are S[j] - S[i-1]
        # We need to check if any new sum collides with existing sums.
        # Existing sums are all differences of prefix sums of seq.
        # New sums involving x:
        # 1. x itself
        # 2. x + suffix of seq[idx:]
        # 3. prefix of seq[:idx] + x
        # 4. prefix of seq[:idx] + x + suffix of seq[idx:]
        
        # This is expensive to do from scratch every time.
        # Better to maintain prefix sums and sum set.
        # But for local search, we might modify the sequence.
        # Let's just recompute for the candidate sequence.
        
        new_seq = seq[:idx] + [x] + seq[idx:]
        return check_validity(new_seq), idx

    def check_validity(seq):
        """
        Check if seq is a valid sequence.
        Returns True if all consecutive block sums are distinct.
        """
        if not seq:
            return True
        if seq[0] < 1 or seq[-1] > 256:
            return False
        # Check strictly increasing
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        
        sums = set()
        # Prefix sums
        prefix = [0] * (len(seq) + 1)
        for i in range(len(seq)):
            prefix[i+1] = prefix[i] + seq[i]
            
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                s = prefix[j+1] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_build():
        values = []
        sums = set()
        for x in range(1, 257):
            # Calculate new tails if x is appended to values
            # Since we are building from scratch, we append to the end.
            # But wait, the greedy in the prompt appends to the end?
            # "for x in range(1, 257): ... values.append(x)"
            # This assumes we are adding the next smallest valid number to the end.
            # This is a specific greedy.
            # Let's stick to this for the baseline.
            
            # Optimization: 
            # tails are sums of suffixes of values + x
            # If values = [a1, ..., ak], tails for x are:
            # x
            # x + ak
            # x + ak + a(k-1)
            # ...
            # x + sum(values)
            
            if not values:
                values.append(x)
                sums.update([x])
                continue
                
            # Check if x can be appended
            # We need to check if any new sum collides with existing sums.
            # New sums are x + suffix_sum for all suffixes of values, plus x itself.
            
            current_sum = 0
            valid = True
            # Suffix sums of values
            # values is [v1, ..., vk]
            # Suffixes: vk, v(k-1)+vk, ..., sum(values)
            
            # Iterate from end of values
            s = 0
            for v in reversed(values):
                s += v
                new_sum = x + s
                if new_sum in sums:
                    valid = False
                    break
            if valid:
                if x in sums:
                    valid = False
            
            if valid:
                values.append(x)
                # Update sums
                # New sums are x + suffix_sum for all suffixes of OLD values, plus x
                s = 0
                for v in reversed(values[:-1]): # Exclude x
                    s += v
                    sums.add(x + s)
                sums.add(x)
                
        return values

    # 1. Get baseline
    best_seq = greedy_build()
    best_len = len(best_seq)
    
    # 2. Local Search / Simulated Annealing
    # We will try to improve best_seq by inserting random numbers.
    # If we can't insert, we try to swap or replace.
    
    # To speed up validity check, we can precompute prefix sums and sum set for current seq.
    # But since we are doing many checks, let's use a fast check function.
    
    def fast_check(seq):
        if len(seq) == 0:
            return True
        if seq[0] < 1 or seq[-1] > 256:
            return False
        for i in range(len(seq)-1):
            if seq[i] >= seq[i+1]:
                return False
        sums = set()
        prefix = 0
        # Calculate all prefix sums
        # p[0] = 0
        # p[i] = sum(seq[0..i-1])
        # Sum(i..j) = p[j+1] - p[i]
        p = [0] * (len(seq) + 1)
        for i in range(len(seq)):
            p[i+1] = p[i] + seq[i]
            
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                s = p[j+1] - p[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Refine best_seq
    current_seq = list(best_seq)
    current_len = len(current_seq)
    
    # Helper to get random valid insertion
    def try_insert_random(seq):
        # Pick a random position and a random value
        # Value must be in 1..256 and not in seq
        if not seq:
            return [random.randint(1, 256)], 0
        
        # Try a few times
        for _ in range(10):
            x = random.randint(1, 256)
            if x in seq:
                continue
            # Find insertion index
            lo, hi = 0, len(seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if seq[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx < len(seq) and seq[idx] == x:
                continue
            
            new_seq = seq[:idx] + [x] + seq[idx:]
            if fast_check(new_seq):
                return new_seq, idx
        return None, -1

    # Main SA loop
    temperature = 1.0
    cooling = 0.99
    min_temp = 0.01
    
    while time.time() < deadline:
        # Try to insert
        new_seq, idx = try_insert_random(current_seq)
        if new_seq:
            current_seq = new_seq
            current_len = len(current_seq)
            if current_len > best_len:
                best_seq = list(current_seq)
                best_len = current_len
                temperature = 1.0 # Reset temperature on improvement? Or just keep going.
            continue
            
        # If no insertion, try to swap
        if len(current_seq) > 1:
            i = random.randint(0, len(current_seq) - 1)
            j = random.randint(0, len(current_seq) - 1)
            if i != j:
                # Swap i and j? No, must remain sorted.
                # Instead, try to replace element i with a random other value.
                old_val = current_seq[i]
                new_val = random.randint(1, 256)
                if new_val == old_val:
                    continue
                if new_val in current_seq:
                    continue
                
                # Remove old, insert new
                seq_wo_old = current_seq[:i] + current_seq[i+1:]
                # Insert new_val into seq_wo_old
                lo, hi = 0, len(seq_wo_old)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if seq_wo_old[mid] < new_val:
                        lo = mid + 1
                    else:
                        hi = mid
                idx_new = lo
                if idx_new < len(seq_wo_old) and seq_wo_old[idx_new] == new_val:
                    continue
                
                candidate = seq_wo_old[:idx_new] + [new_val] + seq_wo_old[idx_new:]
                
                if fast_check(candidate):
                    # Accept if it improves "potential" or with prob T
                    # Heuristic: Smaller sums are better? Or just accept to explore.
                    # Let's accept with probability temperature
                    if random.random() < temperature:
                        current_seq = candidate
                    temperature *= cooling
        else:
            temperature *= cooling

    return best_seq

# EVOLVE-BLOCK-END
