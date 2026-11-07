def _is_valid_set(nums):
    n = len(nums)
    if n == 0:
        return True
    
    if len(set(nums)) != n:
        return False
    
    sums = {0}
    prev_len = 1
    for num in nums:
        # Generate new sums
        new_sums = set()
        for s in sums:
            new_sums.add(s + num)
        sums.update(new_sums)
        
        # Check for collisions
        # If no collisions, len(sums) should be 2 * prev_len
        if len(sums) != 2 * prev_len:
            return False
        prev_len = len(sums)
        
    sums.discard(0)
    
    # Check divisibility
    max_sum = max(sums)
    for s in sums:
        if s * 2 > max_sum:
            continue
        for k in range(2, max_sum // s + 1):
            if s * k in sums:
                return False
    return True
