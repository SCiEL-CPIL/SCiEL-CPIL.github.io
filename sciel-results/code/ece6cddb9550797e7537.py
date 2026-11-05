import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Bitset representation
    # Value v in 1..512 maps to bit v-1 in an integer B
    
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

    rng = random.Random(42)
    
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
                
    # 4. "Attractive" initialization: Start with a random set, then apply greedy attractive moves
    base_random = rng.sample(range(1, 513), 32)
    base_B = list_to_bitset(base_random)
    base_score = score_bitset(base_B)
    
    # Quick greedy improvement on the base random set
    for _ in range(100):
        improved = False
        for i in range(32):
            u = base_random[i]
            for j in range(i+1, 32):
                v = base_random[j]
                # Generate candidates
                cands = set()
                c1 = 2*u - v
                c2 = 2*v - u
                if (u + v) % 2 == 0:
                    c3 = (u + v) // 2
                    cands.add(c3)
                cands.add(c1)
                cands.add(c2)
                
                for w in cands:
                    if 1 <= w <= 512 and w not in base_random:
                        # Try replacing a random element k with w
                        k = rng.randint(0, 31)
                        old_val = base_random[k]
                        new_B = base_B ^ (1 << (old_val - 1)) | (1 << (w - 1))
                        new_score = score_bitset(new_B)
                        if new_score > base_score:
                            base_random[k] = w
                            base_B = new_B
                            base_score = new_score
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
            
    candidates.append(base_random)

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

    # Simulated Annealing with Attractive Moves
    current_set = best_set[:]
    current_B = list_to_bitset(current_set)
    current_score = best_score
    
    current_temp = 10.0
    cooling = 0.99995
    
    iteration = 0
    # We will run until deadline
    while time.time() < deadline - 5:
        # Move selection
        if rng.random() < 0.5:
            # Random Move
            idx = rng.randint(0, 31)
            old_val = current_set[idx]
            new_val = rng.randint(1, 512)
            
            if new_val == old_val or new_val in current_set:
                continue
            
            new_B = current_B ^ (1 << (old_val - 1)) | (1 << (new_val - 1))
        else:
            # Attractive Move
            # Pick two random elements u, v
            i, j = rng.sample(range(32), 2)
            u = current_set[i]
            v = current_set[j]
            
            cands = set()
            c1 = 2*u - v
            c2 = 2*v - u
            if (u + v) % 2 == 0:
                c3 = (u + v) // 2
                cands.add(c3)
            cands.add(c1)
            cands.add(c2)
            
            valid_cands = [w for w in cands if 1 <= w <= 512 and w not in current_set]
            
            if not valid_cands:
                continue
            
            new_val = rng.choice(valid_cands)
            # Pick a random element to replace
            idx = rng.randint(0, 31)
            old_val = current_set[idx]
            
            if new_val == old_val:
                continue
            
            new_B = current_B ^ (1 << (old_val - 1)) | (1 << (new_val - 1))
            
        new_score = score_bitset(new_B)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or (current_temp > 1e-10 and rng.random() < math.exp(delta / current_temp)):
            # Accept
            current_set[idx] = new_val
            current_B = new_B
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_set = current_set[:]
        
        iteration += 1
        if iteration % 100 == 0:
            current_temp *= cooling
            
    # Final Polish: Greedy Local Search
    # Try to improve by replacing any element with any attractive candidate
    greedy_set = best_set[:]
    greedy_B = list_to_bitset(greedy_set)
    greedy_score = best_score
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for i in range(32):
            if time.time() > deadline - 1:
                break
            u = greedy_set[i]
            for j in range(i+1, 32):
                if time.time() > deadline - 1:
                    break
                v = greedy_set[j]
                
                cands = set()
                c1 = 2*u - v
                c2 = 2*v - u
                if (u + v) % 2 == 0:
                    c3 = (u + v) // 2
                    cands.add(c3)
                cands.add(c1)
                cands.add(c2)
                
                for w in cands:
                    if 1 <= w <= 512 and w not in greedy_set:
                        # Try replacing element i or j or any other?
                        # Replacing an element that is part of few APs is ideal, but expensive.
                        # Let's try replacing i and j specifically, as they are involved in the AP logic.
                        # Also try a random element.
                        for k in [i, j, rng.randint(0, 31)]:
                            old_val = greedy_set[k]
                            if w == old_val:
                                continue
                            new_B = greedy_B ^ (1 << (old_val - 1)) | (1 << (w - 1))
                            new_score = score_bitset(new_B)
                            if new_score > greedy_score:
                                greedy_set[k] = w
                                greedy_B = new_B
                                greedy_score = new_score
                                improved = True
                                if new_score > best_score:
                                    best_score = new_score
                                    best_set = greedy_set[:]
                                break # Break k loop
                if improved:
                    break
            if improved:
                break
                
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
