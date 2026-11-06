import time

def construct():
    # Deadline guard, though this algorithm is O(2^10 * 10) ~ trivial
    start_time = time.time()
    deadline = start_time + 80

    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    # We store them grouped by size to allow incremental checking
    subsets_by_size = {k: [] for k in range(1, 11)}
    
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        residue = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                residue = (residue + inv[x - 1]) % 101
        subsets_by_size[size].append((residue, mask))

    assignment = {}
    
    # Check K from 1 to 10
    for K in range(1, 11):
        if time.time() > deadline:
            # Fallback: just return the first valid assignment found so far?
            # Or raise? The prompt says return best result.
            # If we haven't found a full cover, we must return something valid.
            # But we know a cover exists for K=10.
            # If we run out of time, we might not have checked K=10 yet?
            # No, the loop is tiny. This guard is just for safety.
            pass
            
        for residue, mask in subsets_by_size[K]:
            if residue not in assignment:
                assignment[residue] = mask
        
        if len(assignment) == 101:
            # We found a cover with max size K
            result = []
            for r in range(101):
                mask = assignment[r]
                terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
                result.append(terms)
            return result
            
    # If loop finishes without return (should not happen), return a default valid solution
    # Using all 10 denominators for every row is valid but max size 10
    # But we should have returned earlier.
    # Just in case, construct a valid solution with max size 10
    # Actually, if K=10, we definitely have a cover because we have 1023 subsets.
    # So the loop must return.
    return [[1,2,3,4,5,6,7,8,9,10]] * 101
