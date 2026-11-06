import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    required = [1, 2, 5, 11]
    N = 284
    
    # Initial sums for required elements
    # Sums a+b for a<=b
    init_sums = set()
    for i in range(len(required)):
        for j in range(i, len(required)):
            init_sums.add(required[i] + required[j])
            
    init_mask = 0
    for s in init_sums:
        init_mask |= (1 << s)
        
    best_set = required[:]
    best_size = len(required)
    
    # Precompute sum bits for all pairs (x, y) where x <= y
    # sum_bit[x][y] = 1 << (x+y)
    # This saves addition and shift in the inner loop
    sum_bits = [[0]*(N+1) for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(x, N+1):
            sum_bits[x][y] = 1 << (x+y)
            
    def search(current_S, current_mask, current_size, last_x):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = current_S[:]
            
        # Pruning
        # Calculate free sum slots
        # Total possible sums are in range [2, 2N]
        # Bit 0 and 1 are unused.
        # Max sum is 2N.
        # Number of bits is 2N+1.
        # Free bits count is hard to get quickly from mask without popcount.
        # Python int has no native popcount in older versions, but bit_count() in 3.10+.
        # Let's assume 3.10+. If not, bin(mask).count("1").
        
        used_sums = current_mask.bit_count()
        total_possible_sums = (2 * N) - 2 + 1 # 2 to 2N inclusive is 2N-1 values?
        # Sums range from 1+1=2 to 284+284=568.
        # Count = 568 - 2 + 1 = 567.
        free_sums = 567 - used_sums
        
        if free_sums <= 0:
            return
            
        # Estimate max additional elements k
        # New sums needed for k elements added to current set of size S:
        # k * S (cross) + k(k+1)/2 (internal)
        # We need k * S + k(k+1)/2 <= free_sums
        
        S_size = current_size
        # Solve k^2/2 + k(S_size/2 + 1/2) - free <= 0
        # Approx k <= sqrt(2 * free)
        max_k = int((2 * free_sums) ** 0.5) + 2
        
        if current_size + max_k <= best_size:
            return
            
        # Try adding next element
        # Start from last_x + 1
        # To improve search, we might want to try elements that are "sparse"
        # But for now, simple iteration.
        
        # Optimization: If we are close to deadline, only try promising branches?
        
        for x in range(last_x + 1, N + 1):
            # Check validity of x
            # 1. 2x sum
            # 2. x + y for all y in current_S
            
            valid = True
            new_mask = current_mask
            new_mask |= sum_bits[x][x]
            
            if new_mask & current_mask:
                continue
                
            for y in current_S:
                new_mask |= sum_bits[x][y] if x >= y else sum_bits[y][x]
                if new_mask & current_mask:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Recurse
            search(current_S + [x], new_mask, current_size + 1, x)
            
            if time.time() > deadline:
                return
                
    search(required, init_mask, len(required), 11)
    
    return best_set
