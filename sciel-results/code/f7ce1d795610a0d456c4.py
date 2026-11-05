import time
import random

def construct():
    N = 512
    # Bit i represents element i+1
    
    # Precompute conflict masks
    # conflicts[x_idx] is a list of masks. 
    # A mask M in conflicts[x_idx] represents a set of 3 elements {a, b, c} 
    # such that {x, a, b, c} forms a 4-AP.
    # If (S & M) == M, then adding x to S creates a 4-AP.
    
    conflicts = [[] for _ in range(N)]
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    # 1 <= a, a+3d <= N
    # d >= 1
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Convert to 0-indexed bits
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            c_bit = 1 << (c - 1)
            e_bit = 1 << (e - 1)
            
            # For each element in the AP, the other 3 form a conflict mask
            # Element a (index a-1)
            mask_bce = b_bit | c_bit | e_bit
            conflicts[a - 1].append(mask_bce)
            
            # Element b (index b-1)
            mask_ace = a_bit | c_bit | e_bit
            conflicts[b - 1].append(mask_ace)
            
            # Element c (index c-1)
            mask_abe = a_bit | b_bit | e_bit
            conflicts[c - 1].append(mask_abe)
            
            # Element e (index e-1)
            mask_abc = a_bit | b_bit | c_bit
            conflicts[e - 1].append(mask_abc)

    def can_add(x_idx, S):
        """Check if element x_idx can be added to set S without creating a 4-AP."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def count_bits(S):
        return bin(S).count('1')

    def greedy_fill(S, order):
        """Add elements from order to S if they don't create a 4-AP."""
        for x_idx in order:
            if not (S & (1 << x_idx)):
                if can_add(x_idx, S):
                    S |= (1 << x_idx)
        return S

    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s buffer for 60s limit

    best_S = 0
    best_size = 0
    current_S = 0
    current_size = 0

    # 1. Initial Greedy Constructions
    # Try a few random orderings to get a good starting point
    base_order = list(range(N))
    
    # Always try natural order first
    S_try = greedy_fill(0, base_order)
    size_try = count_bits(S_try)
    if size_try > best_size:
        best_size = size_try
        best_S = S_try
        current_S = S_try
        current_size = size_try

    # Try random orderings
    for _ in range(50):
        random.shuffle(base_order)
        S_try = greedy_fill(0, base_order)
        size_try = count_bits(S_try)
        if size_try > best_size:
            best_size = size_try
            best_S = S_try
            current_S = S_try
            current_size = size_try

    # 2. Iterated Local Search (ILS)
    # Strategy: 
    # 1. Perturb current_S by removing k random elements.
    # 2. Run multiple greedy fills with different random orders on the perturbed set.
    # 3. Keep the best result from the fills.
    # 4. If it improves global best, update.
    # 5. Set current_S to the best result found in this iteration.
    
    # To speed up, we pre-generate a few random orders? No, shuffling is fast.
    # We will try 'num_fills' different orders for each perturbation.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Determine perturbation size k
        # Smaller k explores finer details, larger k escapes deeper local optima.
        # Let's vary k.
        k = random.randint(2, 6)
        
        # Get list of elements in current_S
        elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
        
        if len(elements_in_set) < k:
            k = len(elements_in_set)
            if k == 0:
                continue
        
        if k > 0:
            removed = random.sample(elements_in_set, k)
            S_temp = current_S
            for idx in removed:
                S_temp &= ~(1 << idx)
            
            # Run multiple greedy fills to find the best local optimum
            best_local_S = S_temp
            best_local_size = count_bits(S_temp)
            
            # Number of fills to try. 
            # Each fill is ~1-2ms. 
            # If we do 10 fills, that's ~10-20ms per iteration.
            # 55s / 0.02s = ~2750 iterations.
            # This seems reasonable.
            num_fills = 10
            
            for _ in range(num_fills):
                order = list(range(N))
                random.shuffle(order)
                S_filled = greedy_fill(S_temp, order)
                size_filled = count_bits(S_filled)
                
                if size_filled > best_local_size:
                    best_local_size = size_filled
                    best_local_S = S_filled
            
            # Update global best
            if best_local_size > best_size:
                best_size = best_local_size
                best_S = best_local_S
            
            # Move to the new local optimum
            current_S = best_local_S
            current_size = best_local_size

    # Return the best set found
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
