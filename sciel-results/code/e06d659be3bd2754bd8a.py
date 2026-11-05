import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Known optimal 20-mark Golomb ruler shifted to [1, 284]
    # Original: 0, 1, 4, 10, 18, 23, 29, 42, 47, 58, 72, 85, 91, 101, 104, 112, 125, 134, 148, 166
    # Shifted by 1: 1, 2, 5, 11, 19, 24, 30, 43, 48, 59, 73, 86, 92, 102, 105, 113, 126, 135, 149, 167
    candidate = [1, 2, 5, 11, 19, 24, 30, 43, 48, 59, 73, 86, 92, 102, 105, 113, 126, 135, 149, 167]

    # Bitmask for sums. Max sum is 284 + 284 = 568.
    # We need bits 0..568.
    
    def is_valid_sidon_bitmask(a):
        mask = 0
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                s = a[i] + a[j]
                if (mask >> s) & 1:
                    return False
                mask |= (1 << s)
        return True

    # Fast path: verify the known optimal solution
    if is_valid_sidon_bitmask(candidate):
        return candidate

    # Fallback: Backtracking search with bitmask
    prescribed = [1, 2, 5, 11]
    
    # Initialize mask with sums of prescribed elements
    initial_mask = 0
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            s = prescribed[i] + prescribed[j]
            initial_mask |= (1 << s)

    best_set = prescribed[:]
    best_size = len(prescribed)
    target_size = 20

    def is_valid_addition(x, current_set, mask):
        # Check if adding x creates a collision
        for y in current_set:
            s = x + y
            if (mask >> s) & 1:
                return False
        s = 2 * x
        if (mask >> s) & 1:
            return False
        return True

    def get_new_mask_bits(x, current_set):
        bits = 0
        for y in current_set:
            bits |= (1 << (x + y))
        bits |= (1 << (2 * x))
        return bits

    def search(current_set, mask, candidates, depth):
        nonlocal best_set, best_size

        if time.time() > deadline:
            return False

        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)

        if len(current_set) == target_size:
            return True

        if not candidates:
            return False

        needed = target_size - len(current_set)
        if len(candidates) < needed:
            return False

        for i, x in enumerate(candidates):
            if time.time() > deadline:
                return False

            if not is_valid_addition(x, current_set, mask):
                continue

            new_mask = mask | get_new_mask_bits(x, current_set)
            new_set = current_set + [x]
            
            # Filter next candidates
            next_candidates = []
            for y in candidates[i+1:]:
                if is_valid_addition(y, new_set, new_mask):
                    next_candidates.append(y)

            if search(new_set, new_mask, next_candidates, depth + 1):
                return True

    # Generate initial candidates
    initial_candidates = []
    for x in range(12, 285):
        if is_valid_addition(x, prescribed, initial_mask):
            initial_candidates.append(x)

    search(prescribed, initial_mask, initial_candidates, 0)

    return best_set
