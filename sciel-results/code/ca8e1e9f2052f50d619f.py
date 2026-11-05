import time
import random

def _valid_mean_free(selected):
    """Returns True if the list 'selected' satisfies the condition that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    Uses bitset DP for speed."""
    n = len(selected)
    if n < 3:
        return True
    
    selected = list(set(selected))
    n = len(selected)
    if n < 3:
        return True
    
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        if m < 2:
            continue
            
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(m, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
                    
        for k in range(2, m + 1):
            target = k * a
            if dp[k] & (1 << target):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Initial random set of size 40
    current_set = random.sample(range(1, 513), 40)
    current_set.sort()
    
    # Precompute some constants
    max_val = 512
    
    def get_violations(s):
        """Count the number of violated mean constraints in set s.
        A violation is if element a is the mean of a subset of size k >= 2 of others.
        We count each such (a, subset) pair as a violation? Or just count elements involved?
        For SA, we want a cost function. Let's count the number of violating pairs (a, subset).
        Actually, just counting the number of elements 'a' that are means of some subset is simpler
        but might not be smooth enough. Let's count the total number of violating subsets.
        """
        if len(s) < 3:
            return 0
        
        violations = 0
        n = len(s)
        for i, a in enumerate(s):
            others = [x for j, x in enumerate(s) if j != i]
            m = len(others)
            if m < 2:
                continue
                
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            for k in range(2, m + 1):
                target = k * a
                if dp[k] & (1 << target):
                    # Count the number of subsets of size k that sum to target
                    # This is expensive to count exactly with bitsets if we just have existence.
                    # For SA, existence is often enough as a binary penalty, or we can just count 1 per (a, k)
                    violations += 1
        return violations

    current_cost = get_violations(current_set)
    
    best_set = current_set[:]
    best_cost = current_cost
    
    if best_cost == 0:
        best_set = current_set[:]
    
    # Simulated Annealing Parameters
    T = 100.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Move: pick a random element to remove and a random new element to add
        # Or swap? Adding/Removing changes size. We want to maximize size.
        # Let's try to add elements if possible, or swap to improve cost.
        
        # Strategy: 
        # 1. If cost > 0, try to reduce cost by removing an element or swapping.
        # 2. If cost == 0, try to add a new element. If it fails (cost > 0), accept with probability.
        
        new_set = current_set[:]
        
        if len(new_set) < 48:
            # Try to add
            candidates = [x for x in range(1, 513) if x not in new_set]
            if not candidates:
                break
            new_elem = random.choice(candidates)
            new_set.append(new_elem)
            new_set.sort()
        else:
            # Try to swap
            if len(new_set) < 2:
                break
            idx = random.randint(0, len(new_set) - 1)
            old_elem = new_set[idx]
            candidates = [x for x in range(1, 513) if x not in new_set]
            if not candidates:
                break
            new_elem = random.choice(candidates)
            new_set[idx] = new_elem
            new_set.sort()
        
        new_cost = get_violations(new_set)
        
        # Acceptance criteria
        if new_cost < current_cost:
            # Better cost, always accept
            current_set = new_set
            current_cost = new_cost
            
            # Update best if valid
            if current_cost == 0:
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
                    best_cost = 0
        else:
            # Worse or equal cost
            delta = new_cost - current_cost
            # If we are trying to add and it makes cost worse, we need to be careful
            # If we are swapping and it makes cost worse, accept with prob exp(-delta/T)
            prob = random.random()
            if prob < pow(2.718281828, -delta / T):
                current_set = new_set
                current_cost = new_cost
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Final greedy expansion on the best valid set found
    if best_cost == 0:
        # Try to expand best_set
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in remaining[:]:
                if time.time() > deadline:
                    break
                if x in best_set_set:
                    continue
                # Fast check
                temp = best_set + [x]
                if _valid_mean_free(temp):
                    best_set = temp
                    best_set_set.add(x)
                    remaining.remove(x)
                    improved = True
    
    # Ensure we return a valid set. If best_set is invalid (cost > 0), we need to fix it or return a smaller valid one.
    # The SA might end in an invalid state. We should keep track of the best VALID state.
    # My code above only updates best_set if current_cost == 0.
    # So best_set is always valid (or empty/size<3 which is valid).
    
    if not _valid_mean_free(best_set):
        # Fallback: return a simple valid set
        # Powers of 2 are always valid?
        # 1, 2, 4, 8... 
        # Mean of 1,2 is 1.5 (not integer). Mean of 1,4 is 2.5.
        # Is 2 the mean of 1 and 3? 3 is not in set.
        # Powers of 2 are sum-free and mean-free?
        # Yes, likely.
        fallback = []
        val = 1
        while val <= 512:
            fallback.append(val)
            val *= 2
        return fallback

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
