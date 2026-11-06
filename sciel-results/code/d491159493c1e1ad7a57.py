import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    # For each x, store pairs (a, b) such that {a, b, x} is a 3-AP
    # a, b < x is not required, just distinct elements.
    # To avoid double counting in gain, we can just count all pairs.
    # Actually, for gain of adding x, we count pairs (a,b) in S such that a,b,x form a 3-AP.
    # This is symmetric.
    
    neighbors_3ap = [[] for _ in range(N + 1)]
    neighbors_4ap = [[] for _ in range(N + 1)]
    
    # 3-APs: a, a+d, a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            
            # For x1, the other two are x2, x3
            neighbors_3ap[x1].append((x2, x3))
            # For x2, the other two are x1, x3
            neighbors_3ap[x2].append((x1, x3))
            # For x3, the other two are x1, x2
            neighbors_3ap[x3].append((x1, x2))
            
    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            
            # For x1, others are x2, x3, x4
            neighbors_4ap[x1].append((x2, x3, x4))
            # For x2, others are x1, x3, x4
            neighbors_4ap[x2].append((x1, x3, x4))
            # For x3, others are x1, x2, x4
            neighbors_3ap[x3].append((x1, x2, x4)) # Wait, typo in variable name above? No, this is 4-AP neighbor
            neighbors_4ap[x3].append((x1, x2, x4))
            # For x4, others are x1, x2, x3
            neighbors_4ap[x4].append((x1, x2, x3))
            
    def count_3aps(S):
        """Count 3-APs in bitmask S."""
        count = 0
        # Iterate through all 3-APs? Or iterate through elements?
        # Iterating through elements and counting pairs is O(N * deg)
        # Let's just iterate through precomputed 3-AP list for simplicity if needed, 
        # but for delta we use neighbors.
        # For total count, we can sum gains if we built it up, or just recalculate.
        # Recalculating is O(num_3aps). num_3aps is ~ N^2/4 = 16000. Fast enough.
        
        # Actually, we can just use the neighbors structure.
        # Each 3-AP is counted 3 times (once for each element).
        # So count = (sum over x in S of (number of pairs in S for x)) / 3
        total = 0
        for x in range(1, N + 1):
            if (S >> x) & 1:
                for a, b in neighbors_3ap[x]:
                    if ((S >> a) & 1) and ((S >> b) & 1):
                        total += 1
        return total // 3

    def is_valid_add(S, x):
        """Check if adding x to S creates a 4-AP."""
        for a, b, c in neighbors_4ap[x]:
            if ((S >> a) & 1) and ((S >> b) & 1) and ((S >> c) & 1):
                return False
        return True

    def gain_add(S, x):
        """Calculate gain in 3-AP count if we add x to S."""
        gain = 0
        for a, b in neighbors_3ap[x]:
            if ((S >> a) & 1) and ((S >> b) & 1):
                gain += 1
        return gain

    def loss_remove(S, x):
        """Calculate loss in 3-AP count if we remove x from S."""
        loss = 0
        for a, b in neighbors_3ap[x]:
            if ((S >> a) & 1) and ((S >> b) & 1):
                loss += 1
        return loss

    def build_base_set():
        """Build the classic 4-AP-free set: numbers with no '1' in base-3."""
        S = 0
        for x in range(1, N + 1):
            valid = True
            n = x
            while n > 0:
                if n % 3 == 1:
                    valid = False
                    break
                n //= 3
            if valid:
                S |= (1 << x)
        return S

    def greedy_extend(S):
        """Extend set by adding valid elements with high 3-AP gain."""
        changed = True
        while changed:
            changed = False
            best_x = -1
            best_gain = -1
            for x in range(1, N + 1):
                if not ((S >> x) & 1):
                    if is_valid_add(S, x):
                        g = gain_add(S, x)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
            if best_x != -1:
                S |= (1 << best_x)
                changed = True
        return S

    def local_search(S, max_iter=1000):
        """Local search with swaps and additions."""
        S = S
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
                
            # 1. Try additions
            best_add_x = -1
            best_add_gain = -1
            for x in range(1, N + 1):
                if not ((S >> x) & 1):
                    if is_valid_add(S, x):
                        g = gain_add(S, x)
                        if g > best_add_gain:
                            best_add_gain = g
                            best_add_x = x
            if best_add_x != -1:
                S |= (1 << best_add_x)
                improved = True
                continue

            # 2. Try swaps
            best_swap = None
            best_swap_delta = -1
            for x_out in range(1, N + 1):
                if not ((S >> x_out) & 1):
                    continue
                loss = loss_remove(S, x_out)
                S_without_out = S & ~(1 << x_out)
                
                for x_in in range(1, N + 1):
                    if (S_without_out >> x_in) & 1:
                        continue
                    if not is_valid_add(S_without_out, x_in):
                        continue
                    gain = gain_add(S_without_out, x_in)
                    delta = gain - loss
                    if delta > best_swap_delta:
                        best_swap_delta = delta
                        best_swap = (x_out, x_in)
            
            if best_swap and best_swap_delta > 0:
                x_out, x_in = best_swap
                S = (S & ~(1 << x_out)) | (1 << x_in)
                improved = True
                continue
                
        return S

    def perturb(S, k=5):
        """Remove k random elements to escape local optima."""
        S = S
        elements = [x for x in range(1, N + 1) if (S >> x) & 1]
        if len(elements) <= k:
            return S
        to_remove = random.sample(elements, k)
        for x in to_remove:
            S &= ~(1 << x)
        return S

    # Main logic
    best_S = 0
    best_score = -1
    
    # Initialize with base set
    S_init = build_base_set()
    S_init = greedy_extend(S_init)
    S_init = local_search(S_init, max_iter=200)
    score_init = count_3aps(S_init)
    if score_init > best_score:
        best_score = score_init
        best_S = S_init
    
    # Multi-start local search with perturbations
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Start from a perturbed version of the best solution found so far
        # Or start from a random valid set
        if i % 2 == 0:
            S_curr = perturb(best_S, k=3)
        else:
            # Random start: Greedy add random valid elements
            S_curr = 0
            # Ensure we don't timeout in greedy
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                added = False
                # Shuffle elements to randomize order
                elems = list(range(1, N + 1))
                random.shuffle(elems)
                for x in elems:
                    if not ((S_curr >> x) & 1):
                        if is_valid_add(S_curr, x):
                            S_curr |= (1 << x)
                            added = True
                            break
                if not added:
                    break
        
        S_curr = local_search(S_curr, max_iter=100)
        score_curr = count_3aps(S_curr)
        
        if score_curr > best_score:
            best_score = score_curr
            best_S = S_curr

    # Convert bitmask to sorted list
    result = []
    for x in range(1, N + 1):
        if (best_S >> x) & 1:
            result.append(x)
            
    return result
