import random
import itertools

def construct():
    # We want to maximize the size.
    # Theoretical max size is limited by distinct subset sums.
    # 2^N - 1 <= Sum(a_i)
    # If a_i ~ 200, N=10 -> 1023 <= 2000 (OK)
    # N=11 -> 2047 <= 2200 (Tight)
    
    # We will try to find a valid set of size 11, then 10, etc.
    
    def is_valid(S):
        # S is a tuple of ints
        n = len(S)
        sums = set()
        # Generate all non-empty subset sums
        # For efficiency, we can use DP or just itertools if N is small
        # N=11, 2^11 = 2048 subsets. Fast.
        for r in range(1, n + 1):
            for combo in itertools.combinations(S, r):
                s = sum(combo)
                if s in sums:
                    return False
                sums.add(s)
        
        # Check divisibility
        # Convert to list to sort
        sum_list = list(sums)
        sum_list.sort()
        for i in range(len(sum_list)):
            a = sum_list[i]
            for j in range(i + 1, len(sum_list)):
                b = sum_list[j]
                if b % a == 0:
                    return False
        return True

    # Try sizes from 11 down to 5
    for target_size in range(11, 4, -1):
        # Try a few random attempts
        for _ in range(1000):
            # Pick random distinct integers from 1..256
            # Bias towards larger numbers to increase sum range?
            # Actually, larger numbers increase the chance of divisibility?
            # No, larger numbers make the sums larger, but the ratio is similar.
            # Let's just pick from 1..256.
            S = random.sample(range(1, 257), target_size)
            if is_valid(S):
                return S
        
    # Fallback: Return a known valid set (e.g., size 3 or 4)
    # The seed returned [31, 32, 34].
    # Let's check if [31, 32, 34] is valid.
    # Sums: 31, 32, 34, 63, 65, 66, 97.
    # Distinct? Yes.
    # Divisibility?
    # 31 divides 63? No. 31 divides 65? No. 31 divides 66? No. 31 divides 97? No.
    # 32 divides 63? No. 32 divides 65? No. 32 divides 66? No. 32 divides 97? No.
    # 34 divides 63? No. 34 divides 65? No. 34 divides 66? No. 34 divides 97? No.
    # 63 divides 97? No.
    # 65 divides 97? No.
    # 66 divides 97? No.
    # So [31, 32, 34] is valid.
    
    return [31, 32, 34]
