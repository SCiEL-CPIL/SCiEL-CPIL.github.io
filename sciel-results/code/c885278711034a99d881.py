import time
import random

def _check_sums_valid(x, S, sums):
    """Check if x can be added to set S without creating duplicate pair sums."""
    if x + x in sums:
        return False
    for a in S:
        if x + a in sums:
            return False
    return True

def _check_diffs_valid(x, S, other_diffs):
    """Check if x can be added to set S without creating differences that conflict with other_diffs."""
    for a in S:
        d1 = x - a
        d2 = a - x
        if d1 in other_diffs or d2 in other_diffs:
            return False
    return True

def _add_to_set(x, S, sums, diffs):
    """Add x to set S, updating sums and diffs."""
    S.append(x)
    sums.add(x + x)
    for a in S[:-1]:
        sums.add(x + a)
        diffs.add(x - a)
        diffs.add(a - x)

def _local_improve(A, B, sums_A, sums_B, diffs_A, diffs_B, start_time, timeout):
    """Try to improve the solution by swapping elements between sets or adding new elements."""
    # Try to add any remaining candidates
    used = set(A) | set(B)
    remaining = [x for x in range(1, 257) if x not in used]
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() - start_time > timeout:
            break
        
        valid_A = _check_sums_valid(x, A, sums_A) and _check_diffs_valid(x, A, diffs_B)
        valid_B = _check_sums_valid(x, B, sums_B) and _check_diffs_valid(x, B, diffs_A)
        
        if valid_A and not valid_B:
            _add_to_set(x, A, sums_A, diffs_A)
        elif valid_B and not valid_A:
            _add_to_set(x, B, sums_B, diffs_B)
        elif valid_A and valid_B:
            # Choose the set with fewer elements
            if len(A) <= len(B):
                _add_to_set(x, A, sums_A, diffs_A)
            else:
                _add_to_set(x, B, sums_B, diffs_B)
    
    # Try swapping elements between A and B to see if it allows more additions
    # This is a simple heuristic - try a few random swaps
    for _ in range(10):
        if time.time() - start_time > timeout:
            break
        
        if not A or not B:
            break
        
        # Try swapping a random element from A with one from B
        i = random.randint(0, len(A) - 1)
        j = random.randint(0, len(B) - 1)
        a_val = A[i]
        b_val = B[j]
        
        # Check if swap is valid
        A_new = A[:i] + A[i+1:]
        B_new = B[:j] + B[j+1:]
        
        sums_A_new = set()
        sums_B_new = set()
        diffs_A_new = set()
        diffs_B_new = set()
        
        for a in A_new:
            sums_A_new.add(a + a)
        for i2 in range(len(A_new)):
            for j2 in range(i2+1, len(A_new)):
                sums_A_new.add(A_new[i2] + A_new[j2])
                d = A_new[i2] - A_new[j2]
                diffs_A_new.add(d)
                diffs_A_new.add(-d)
        
        for b in B_new:
            sums_B_new.add(b + b)
        for i2 in range(len(B_new)):
            for j2 in range(i2+1, len(B_new)):
                sums_B_new.add(B_new[i2] + B_new[j2])
                d = B_new[i2] - B_new[j2]
                diffs_B_new.add(d)
                diffs_B_new.add(-d)
        
        # Check if a_val can be added to B_new and b_val can be added to A_new
        can_add_a_to_B = _check_sums_valid(a_val, B_new, sums_B_new) and _check_diffs_valid(a_val, B_new, diffs_A_new)
        can_add_b_to_A = _check_sums_valid(b_val, A_new, sums_A_new) and _check_diffs_valid(b_val, A_new, diffs_B_new)
        
        if can_add_a_to_B and can_add_b_to_A:
            # Perform the swap
            A_new.append(a_val)
            B_new.append(b_val)
            
            # Update sums and diffs for the new elements
            for a in A_new[:-1]:
                sums_A_new.add(a_val + a)
                d = a_val - a
                diffs_A_new.add(d)
                diffs_A_new.add(-d)
            
            for b in B_new[:-1]:
                sums_B_new.add(b_val + b)
                d = b_val - b
                diffs_B_new.add(d)
                diffs_B_new.add(-d)
            
            # Only accept if it allows more additions
            old_size = len(A) + len(B)
            new_size = len(A_new) + len(B_new)
            
            if new_size >= old_size:
                A, B = A_new, B_new
                sums_A, sums_B = sums_A_new, sums_B_new
                diffs_A, diffs_B = diffs_A_new, diffs_B_new
    
    return A, B, sums_A, sums_B, diffs_A, diffs_B

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    best_size = 0
    best_A = []
    best_B = []
    
    # We will try multiple random restarts
    max_restarts = 5000 # Adjust based on speed
    
    for _ in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Candidates are 1..256, with a bias toward smaller numbers
        # Smaller numbers tend to pack better in Sidon sets
        candidates = list(range(1, 257))
        
        # Use a weighted shuffle: smaller numbers have higher probability
        # This creates a bias toward smaller values while maintaining randomness
        weights = [1.0 / (x + 1) for x in candidates]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Shuffle with bias
        for i in range(len(candidates) - 1, 0, -1):
            j = random.choices(range(i + 1), weights=normalized_weights[:i + 1])[0]
            candidates[i], candidates[j] = candidates[j], candidates[i]
        
        for x in candidates:
            if time.time() - start_time > timeout:
                break
            
            # Try adding to A
            valid_A = _check_sums_valid(x, A, sums_A) and _check_diffs_valid(x, A, diffs_B)
            
            # Try adding to B
            valid_B = _check_sums_valid(x, B, sums_B) and _check_diffs_valid(x, B, diffs_A)
            
            # Decision logic:
            # If both valid, choose the set with fewer elements to balance
            if valid_A and valid_B:
                if len(A) < len(B):
                    side = 'A'
                elif len(B) < len(A):
                    side = 'B'
                else:
                    side = random.choice(['A', 'B'])
            elif valid_A:
                side = 'A'
            elif valid_B:
                side = 'B'
            else:
                continue # Cannot add x to either
            
            if side == 'A':
                _add_to_set(x, A, sums_A, diffs_A)
            else:
                _add_to_set(x, B, sums_B, diffs_B)
        
        # Local improvement phase
        A, B, sums_A, sums_B, diffs_A, diffs_B = _local_improve(
            A, B, sums_A, sums_B, diffs_A, diffs_B, start_time, timeout
        )
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]

    return [best_A, best_B]
