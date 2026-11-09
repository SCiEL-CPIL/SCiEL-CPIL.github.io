import time
import random
import sys

def _score_from_bitmask(mask):
    """Count distinct d > 0 such that a, a+d, a+2d all in set represented by bitmask."""
    # mask is a 512-bit integer where bit i represents value i+1
    diffs = 0
    # For each possible difference d, check if there exists a such that bits a, a+d, a+2d are all set
    # a ranges from 1 to 512-2d
    # We can use bit operations: for a given d, we need (mask & (mask >> d) & (mask >> (2*d))) != 0
    # But we need to be careful about the bit positions
    # Let's use a different approach: iterate over d and use bit operations
    for d in range(1, 257):  # max d such that 1 + 2d <= 512 => d <= 255
        # Check if there exists a such that bits at positions a-1, a+d-1, a+2d-1 are all set
        # (mask >> (d-1)) & (mask >> (2*d-1)) & mask should have a common set bit
        # Actually, let's think differently.
        # For a 3-AP (a, a+d, a+2d), we need bits at positions a-1, a+d-1, a+2d-1
        # This is equivalent to: mask has bit i, and mask has bit i+d, and mask has bit i+2d for some i
        # Using bit shifts: (mask & (mask >> d) & (mask >> (2*d))) != 0
        # But this checks if there's a position i where all three are set
        # Wait, that's not quite right. Let me reconsider.
        # If mask has bit i, then mask >> d has bit i-d (if i>=d)
        # So (mask & (mask >> d)) has bit i where both mask[i] and mask[i+d] are set
        # Then (mask & (mask >> d) & (mask >> (2*d))) has bit i where mask[i], mask[i+d], mask[i+2d] are all set
        # So we just need to check if this is non-zero
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            diffs += 1
    return diffs

def _bitmask_to_set(mask):
    """Convert bitmask to sorted list of values."""
    result = []
    for i in range(512):
        if mask & (1 << i):
            result.append(i + 1)
    return result

def _set_to_bitmask(S):
    """Convert set to bitmask."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return mask

def _score_from_set(S):
    """Count distinct d > 0 such that a, a+d, a+2d in S for some a."""
    mask = _set_to_bitmask(S)
    return _score_from_bitmask(mask)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 32
    MAX_VAL = 512
    
    def random_set():
        return set(random.sample(range(1, MAX_VAL + 1), N))
    
    # Start with structured initializations
    initializations = []
    
    # 1. Dense set
    initializations.append(set(range(1, 33)))
    
    # 2. Arithmetic progression with various step sizes
    for step in [2, 3, 4, 5, 8, 10, 16]:
        if 1 + (N - 1) * step <= MAX_VAL:
            initializations.append(set(1 + i * step for i in range(N)))
    
    # 3. Union of two APs
    # A = {a + i * d1 for i in range(n1)}, B = {b + j * d2 for j in range(n2)}
    for d1 in [2, 3, 4, 5, 8]:
        for d2 in [3, 4, 5, 8, 10, 16]:
            if d1 == d2:
                continue
            for n1 in range(8, N - 7):
                n2 = N - n1
                # Try to find offsets that keep values in range
                best_offset_a = None
                best_offset_b = None
                for a in range(1, 50):
                    for b in range(1, 50):
                        A = set(a + i * d1 for i in range(n1))
                        B = set(b + j * d2 for j in range(n2))
                        if A & B:
                            continue
                        if max(A) > MAX_VAL or max(B) > MAX_VAL:
                            continue
                        if min(A) < 1 or min(B) < 1:
                            continue
                        S = A | B
                        if len(S) == N:
                            best_offset_a = a
                            best_offset_b = b
                            break
                    if best_offset_a is not None:
                        break
                if best_offset_a is not None:
                    A = set(best_offset_a + i * d1 for i in range(n1))
                    B = set(best_offset_b + j * d2 for j in range(n2))
                    if len(A | B) == N:
                        initializations.append(A | B)
    
    # 4. Random sets
    for _ in range(10):
        initializations.append(random_set())
    
    # Evaluate all initializations
    best_set = None
    best_score = -1
    for S in initializations:
        if len(S) != N:
            continue
        score = _score_from_set(S)
        if score > best_score:
            best_score = score
            best_set = set(S)
    
    current_set = best_set
    current_score = best_score
    
    # Local search with targeted moves
    def local_search(S, max_iters=20000):
        S = set(S)
        score = _score_from_set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            # Try replacing each element with a targeted value
            for x in list(S):
                # Try values that complete 3-APs
                # For each pair (a, c) in S, check if midpoint m = (a+c)/2 is not in S
                # If we add m, we get a new 3-AP
                # Or for each a, d, check if a+d and a+2d are in S but a is not
                targeted_values = set()
                
                # Find midpoints of pairs
                for a in S:
                    for c in S:
                        if a < c and (c - a) % 2 == 0:
                            m = (a + c) // 2
                            if m not in S and 1 <= m <= MAX_VAL:
                                targeted_values.add(m)
                
                # Find values that would complete APs as first term
                for d in range(1, 256):
                    for a in S:
                        if a + 2 * d <= MAX_VAL:
                            if (a + d) in S and (a + 2 * d) in S:
                                targeted_values.add(a)
                
                # Find values that would complete APs as second term
                for d in range(1, 256):
                    for a in S:
                        if a - d >= 1 and a + d <= MAX_VAL:
                            if (a - d) in S and (a + d) in S:
                                targeted_values.add(a)
                
                # Find values that would complete APs as third term
                for d in range(1, 256):
                    for a in S:
                        if a - 2 * d >= 1:
                            if (a - d) in S and (a - 2 * d) in S:
                                targeted_values.add(a)
                
                # Try replacing x with targeted values
                for y in targeted_values:
                    if y == x or y in S:
                        continue
                    new_S = S - {x} | {y}
                    new_score = _score_from_set(new_S)
                    if new_score > score:
                        S = new_S
                        score = new_score
                        improved = True
                        break
                if improved:
                    break
                
                # If no targeted improvement, try random
                for _ in range(3):
                    y = random.randint(1, MAX_VAL)
                    if y in S:
                        continue
                    new_S = S - {x} | {y}
                    new_score = _score_from_set(new_S)
                    if new_score > score:
                        S = new_S
                        score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        return S, score
    
    # Run local search
    while time.time() < deadline - 10.0:
        new_set, new_score = local_search(current_set, max_iters=10000)
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
            current_set = new_set
        else:
            # Restart with a new random or structured set
            if random.random() < 0.5:
                current_set = random_set()
            else:
                # Try a different structured initialization
                step = random.choice([2, 3, 4, 5, 8, 10, 16])
                if 1 + (N - 1) * step <= MAX_VAL:
                    current_set = set(1 + i * step for i in range(N))
                else:
                    current_set = random_set()
    
    # Final polish
    if time.time() < deadline - 5.0:
        final_set, final_score = local_search(best_set, max_iters=5000)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
    
    return sorted(best_set)
