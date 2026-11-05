import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def score_bitset(B):
        """Count distinct d > 0 such that 3-AP with diff d exists in bitset B."""
        count = 0
        # Max d is such that 1 + 2d <= 512 => 2d <= 511 => d <= 255
        for d in range(1, 256):
            # Check if there is a 3-AP with difference d
            # B & (B >> d) & (B >> 2*d)
            if (B & (B >> d) & (B >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_bitset(S):
        B = 0
        for x in S:
            B |= (1 << (x - 1))
        return B

    def bitset_to_list(B):
        S = []
        while B:
            lsb = B & -B
            idx = lsb.bit_length() - 1
            S.append(idx + 1)
            B ^= lsb
        return S

    def get_available(B):
        """Return list of values 1..512 not in B."""
        available = []
        for v in range(1, 513):
            if not (B & (1 << (v - 1))):
                available.append(v)
        return available

    rng = random.Random(12345)
    
    best_set = None
    best_score = -1
    
    # Initial candidates
    candidates = []
    
    # 1. Simple AP
    candidates.append(list(range(1, 33)))
    
    # 2. Random sets
    for _ in range(50):
        s = rng.sample(range(1, 513), 32)
        candidates.append(s)
        
    # 3. Structured: Two blocks
    for start1 in range(1, 100, 10):
        for start2 in range(200, 400, 50):
            s1 = list(range(start1, start1 + 16))
            s2 = list(range(start2, start2 + 16))
            if max(s2) <= 512:
                candidates.append(s1 + s2)
                
    # 4. Structured: Three blocks
    for start1 in range(1, 50, 5):
        for start2 in range(150, 250, 25):
            for start3 in range(350, 450, 25):
                s1 = list(range(start1, start1 + 11))
                s2 = list(range(start2, start2 + 10))
                s3 = list(range(start3, start3 + 11))
                if max(s3) <= 512:
                    candidates.append(s1 + s2 + s3)

    # Evaluate candidates
    for cand in candidates:
        if len(cand) != 32 or len(set(cand)) != 32:
            continue
        if min(cand) < 1 or max(cand) > 512:
            continue
        B = list_to_bitset(cand)
        sc = score_bitset(B)
        if sc > best_score:
            best_score = sc
            best_set = cand[:]

    # Simulated Annealing with 2-element swap moves
    current_set = best_set[:]
    current_B = list_to_bitset(current_set)
    current_score = best_score
    
    max_iters = 500000
    current_temp = 10.0
    cooling = 0.99995
    
    iteration = 0
    while iteration < max_iters:
        if time.time() > deadline:
            break
        
        # Move: Either swap one element or swap two elements
        move_type = rng.random()
        
        if move_type < 0.7:
            # Single element move
            idx = rng.randint(0, 31)
            old_val = current_set[idx]
            new_val = rng.randint(1, 512)
            
            if new_val == old_val or new_val in current_set:
                iteration += 1
                continue
            
            new_B = current_B ^ (1 << (old_val - 1)) | (1 << (new_val - 1))
            new_score = score_bitset(new_B)
            
            delta = new_score - current_score
            if delta > 0 or rng.random() < math.exp(delta / max(current_temp, 1e-10)):
                current_set[idx] = new_val
                current_B = new_B
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = current_set[:]
            else:
                pass
        else:
            # Two element swap move
            i, j = rng.sample(range(32), 2)
            old_val_i = current_set[i]
            old_val_j = current_set[j]
            
            new_val_i = rng.randint(1, 512)
            new_val_j = rng.randint(1, 512)
            
            # Check for duplicates
            if new_val_i in current_set or new_val_j in current_set:
                # Allow if they are the ones being replaced
                if new_val_i != old_val_i and new_val_i != old_val_j:
                    if new_val_i in current_set:
                        iteration += 1
                        continue
                if new_val_j != old_val_i and new_val_j != old_val_j:
                    if new_val_j in current_set:
                        iteration += 1
                        continue
            
            if new_val_i == new_val_j:
                iteration += 1
                continue
            
            # Build new bitset
            new_B = current_B
            new_B ^= (1 << (old_val_i - 1))
            new_B ^= (1 << (old_val_j - 1))
            new_B |= (1 << (new_val_i - 1))
            new_B |= (1 << (new_val_j - 1))
            
            # Verify validity
            new_list = bitset_to_list(new_B)
            if len(new_list) != 32 or len(set(new_list)) != 32:
                iteration += 1
                continue
            if min(new_list) < 1 or max(new_list) > 512:
                iteration += 1
                continue
            
            new_score = score_bitset(new_B)
            
            delta = new_score - current_score
            if delta > 0 or rng.random() < math.exp(delta / max(current_temp, 1e-10)):
                current_set[i] = new_val_i
                current_set[j] = new_val_j
                current_B = new_B
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = current_set[:]
            else:
                pass
        
        iteration += 1
        if iteration % 200 == 0:
            current_temp *= cooling

    # Final Polish: Local Search (Greedy Hill Climbing)
    greedy_set = best_set[:]
    greedy_B = list_to_bitset(greedy_set)
    greedy_score = best_score
    
    for _ in range(20000):
        if time.time() > deadline - 5:
            break
        idx = rng.randint(0, 31)
        old_val = greedy_set[idx]
        new_val = rng.randint(1, 512)
        if new_val == old_val or new_val in greedy_set:
            continue
        
        new_B = greedy_B ^ (1 << (old_val - 1)) | (1 << (new_val - 1))
        new_score = score_bitset(new_B)
        
        if new_score > greedy_score:
            greedy_set[idx] = new_val
            greedy_B = new_B
            greedy_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = greedy_set[:]

    return sorted(best_set)

if __name__ == "__main__":
    result = construct()
    assert len(result) == 32
    assert len(set(result)) == 32
    assert all(1 <= x <= 512 for x in result)
    # Verify score
    S_set = set(result)
    diffs = set()
    arr = sorted(result)
    for i in range(32):
        a = arr[i]
        for j in range(i+1, 32):
            c = arr[j]
            if (a + c) % 2 == 0:
                mid = (a + c) // 2
                if mid in S_set:
                    d = (c - a) // 2
                    if d > 0:
                        diffs.add(d)
    print(f"Score: {len(diffs)}")
    print(f"Set: {arr}")
