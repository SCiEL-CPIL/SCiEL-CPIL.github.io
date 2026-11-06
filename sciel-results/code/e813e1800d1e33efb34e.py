"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    def is_valid_mean_check(a, others):
        """Check if a is the mean of any subset of size >= 2 from others using bitset DP."""
        n = len(others)
        if n < 2:
            return False
        
        # For each k from 2 to n, check if any k-subset of others sums to k*a
        # Use bitset DP: dp[j] is a bitset where bit s is set if sum s is achievable with j elements
        # Max possible sum with n elements each <= 512: 512*48 = 24576
        max_sum = 512 * n
        
        for k in range(2, n + 1):
            target = k * a
            if target > max_sum:
                break
            
            # DP: dp[j] = bitset of achievable sums using exactly j elements
            # dp[0] has bit 0 set (sum 0 with 0 elements)
            dp = [0] * (k + 1)
            dp[0] = 1  # bit 0 set
            
            for x in others:
                # Process in reverse to avoid using x multiple times
                for j in range(min(k, len([y for y in others[:0]])), -1, -1):
                    pass  # placeholder
            
            # Better approach: iterate elements, update dp from high j to low j
            dp = [0] * (k + 1)
            dp[0] = 1  # only sum 0 achievable with 0 elements
            
            for x in others:
                for j in range(k, 0, -1):
                    # dp[j] |= dp[j-1] shifted left by x
                    dp[j] |= (dp[j - 1] << x)
            
            # Check if target is achievable with exactly k elements
            if (dp[k] >> target) & 1:
                return True
        
        return False
    
    def is_valid_addition(a, selected):
        """Check if adding a to selected keeps the property."""
        others = list(selected)
        # Check: is a the mean of any subset of >= 2 others?
        if is_valid_mean_check(a, others):
            return False
        # Check: for each existing element b, is b the mean of any subset of >= 2 from (selected + [a]) \ {b}?
        for b in selected:
            new_others = [x for x in selected if x != b] + [a]
            if is_valid_mean_check(b, new_others):
                return False
        return True
    
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if is_valid_addition(x, selected):
            selected.append(x)
        if len(selected) >= 48:
            break
    
    return selected
# EVOLVE-BLOCK-END
