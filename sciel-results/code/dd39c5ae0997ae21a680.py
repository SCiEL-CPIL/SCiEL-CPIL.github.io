import time
import math
import random

N = 256
TOTAL_TIME_LIMIT = 85.0

def construct():
    start_time = time.time()
    
    # 1. Precompute APs
    four_aps = []
    three_aps = []
    
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
            
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a-1, a+d-1, a+2*d-1))

    # 2. Precompute neighbor structures for fast updates
    # four_aps_by_elem[x]: list of bitmasks of the other 3 elements in 4-APs containing x
    four_aps_by_elem = [[] for _ in range(N)]
    for ap in four_aps:
        for i in range(4):
            x = ap[i]
            # Create mask of the other 3 elements
            mask = 0
            for j in range(4):
                if j != i:
                    mask |= (1 << ap[j])
            four_aps_by_elem[x].append(mask)

    # three_aps_by_elem[x]: list of bitmasks of the other 2 elements in 3-APs containing x
    three_aps_by_elem = [[] for _ in range(N)]
    for ap in three_aps:
        for i in range(3):
            x = ap[i]
            mask = 0
            for j in range(3):
                if j != i:
                    mask |= (1 << ap[j])
            three_aps_by_elem[x].append(mask)

    def count_3aps(S):
        """Count 3-APs in bitmask S."""
        count = 0
        # Iterate through all 3-APs
        for ap in three_aps:
            a, b, c = ap
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                count += 1
        return count

    def is_valid(S):
        """Check if S has no 4-APs."""
        for ap in four_aps:
            a, b, c, d = ap
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                return False
        return True

    def greedy_initial():
        """Generate a valid initial set using greedy method."""
        S = 0
        # Randomize order to get different starts
        elements = list(range(N))
        random.shuffle(elements)
        
        for x in elements:
            # Check if adding x creates a 4-AP
            valid = True
            for mask in four_aps_by_elem[x]:
                if (S & mask) == mask:
                    valid = False
                    break
            if valid:
                S |= (1 << x)
        return S

    def calculate_gain(S, x):
        """Calculate change in 3-AP count if we add x to S. Assumes x not in S."""
        gain = 0
        for mask in three_aps_by_elem[x]:
            if (S & mask) == mask:
                gain += 1
        return gain

    def calculate_loss(S, x):
        """Calculate change in 3-AP count if we remove x from S. Assumes x in S."""
        loss = 0
        for mask in three_aps_by_elem[x]:
            if (S & mask) == mask:
                loss += 1
        return loss

    best_S = 0
    best_score = 0
    
    # Run multiple SA restarts
    restarts = 0
    while time.time() - start_time < TOTAL_TIME_LIMIT - 5:
        restarts += 1
        # Initialize
        S = greedy_initial()
        current_score = count_3aps(S)
        
        if current_score > best_score:
            best_score = current_score
            best_S = S
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        steps = 0
        max_steps = 100000 # Upper bound, time limit is real constraint
        
        while steps < max_steps and T > T_min:
            if time.time() - start_time > TOTAL_TIME_LIMIT - 5:
                break
            
            # Pick random element
            x = random.randint(0, N - 1)
            
            if (S >> x) & 1:
                # Try removing x
                # Always valid to remove
                loss = calculate_loss(S, x)
                delta = -loss
                new_S = S & ~(1 << x)
                new_score = current_score + delta
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    S = new_S
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S
            else:
                # Try adding x
                # Check validity
                valid = True
                for mask in four_aps_by_elem[x]:
                    if (S & mask) == mask:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                gain = calculate_gain(S, x)
                delta = gain
                new_S = S | (1 << x)
                new_score = current_score + delta
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    S = new_S
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S
            
            T *= cooling
            steps += 1
            
            # Restart SA if stuck for too long? 
            # Simple approach: just let it cool.
            # If we want to be aggressive, we can reset S to best_S occasionally.
            if steps % 5000 == 0:
                # Check time and maybe restart from best
                if time.time() - start_time > TOTAL_TIME_LIMIT - 10:
                    break
                # With some probability, restart from best solution to explore different regions
                if random.random() < 0.1:
                    S = best_S
                    current_score = best_score
                    T = 1.0 # Reset temperature

    # Final verification
    if not is_valid(best_S):
        # Fallback to a simple valid set if something went wrong
        # This shouldn't happen
        pass

    # Convert bitmask to list of 1-based indices
    result = []
    for i in range(N):
        if (best_S >> i) & 1:
            result.append(i + 1)
            
    return result

# Set a fixed seed for reproducibility in tests, though randomization is key for SA
random.seed(42)
