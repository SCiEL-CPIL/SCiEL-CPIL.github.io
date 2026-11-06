"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid(S):
    """Check if set S satisfies the condition: for all a<b<=c in S, a does not divide b+c."""
    S = sorted(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            for k in range(j, n):
                c = S[k]
                if (b + c) % a == 0:
                    return False
    return True

def _greedy_lower_bound():
    """Try to find a large valid set using a greedy approach from the top down."""
    # Start with the upper 1/3 which we know is valid
    S = set(range(171, 257))
    # Try to add elements from the lower part, checking validity
    # We'll try adding elements in decreasing order of "conflict potential"
    # Actually, let's just try all elements from 1 to 170 and see if we can add them
    # But this might be slow if we re-check the whole set each time.
    
    # A better greedy: Start with empty set, add elements one by one if valid.
    # Or start with the large valid set and try to swap/add.
    
    # Let's try a simple greedy: iterate x from 1 to 256, add x if S U {x} is valid.
    # This is O(n^4) in worst case but n=256, so 256^4 is too big.
    # However, we can optimize the check.
    
    # Let's stick to the deterministic upper 1/3 for now, as it's fast and proven.
    return list(S)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # The upper 1/3 construction is known to be valid and has size 86.
    # Let's verify it quickly.
    S_upper_1_3 = list(range(171, 257))
    
    # Is there a better construction?
    # The paper "Large sets with no divisor of the sum of two elements" by Erdos, Sárközy, etc.
    # states that the maximum size is at least n/3.
    # Can we do better?
    
    # Let's try to include more elements from the lower range.
    # The condition is: for a in S, for all b,c in S with a < b <= c, a does not divide b+c.
    
    # If we include small numbers, they impose many constraints.
    # For example, if 1 is in S, then no pair b,c can sum to a multiple of 1 (which is always true).
    # Wait, 1 divides everything. So if 1 is in S, then for any b,c in S with 1 < b <= c,
    # 1 must NOT divide b+c. But 1 always divides b+c.
    # So if 1 is in S, there can be NO b,c in S such that 1 < b <= c.
    # This means S can contain at most 2 elements: {1, x}.
    # So 1 cannot be in a large set.
    
    # What about 2?
    # If 2 is in S, then for any b,c in S with 2 < b <= c, 2 must NOT divide b+c.
    # So b+c must be odd. This means b and c must have different parity.
    # So all elements in S \ {2} must be of different parity? No, that's impossible for large sets.
    # Actually, if S contains 2, then all pairs {b,c} with b,c > 2 must sum to an odd number.
    # This implies that all elements in S \ {2} must have the same parity?
    # If b and c are both odd, b+c is even (divisible by 2). Bad.
    # If b and c are both even, b+c is even (divisible by 2). Bad.
    # So we cannot have two elements of the same parity in S \ {2}.
    # So S \ {2} can contain at most one odd and one even?
    # No, if we have two odds, their sum is even. If we have two evens, their sum is even.
    # So we can have at most one odd and at most one even in S \ {2}.
    # So S can have at most {2, odd, even} = 3 elements.
    # So 2 cannot be in a large set.
    
    # In general, small numbers are very restrictive.
    # The upper 1/3 construction avoids this by only using large numbers.
    
    # Can we extend the upper 1/3?
    # The current set is [171, 256].
    # Min element is 171.
    # Max sum is 256+256 = 512.
    # Multiples of 171 in this range: 342, 513.
    # 342 is not reachable by sum of two numbers > 171?
    # Min sum is 172+172 = 344.
    # So 342 is not a sum of two elements > 171.
    # 513 is not reachable (max 512).
    # So 171 is safe.
    
    # What if we lower the threshold?
    # Let min element be m.
    # We need that no multiple of m in [2(m+1), 512] can be formed by sum of two elements > m.
    # The smallest sum is 2(m+1).
    # The largest sum is 512.
    # Multiples of m in this range: k*m for k >= 2.
    # We need k*m < 2(m+1) for all k >= 2?
    # For k=2: 2m < 2m+2, which is always true.
    # So the first multiple 2m is always less than the min sum 2(m+1).
    # Next multiple 3m.
    # We need 3m > 512 or 3m < 2(m+1).
    # 3m < 2m+2 => m < 2. Impossible for large m.
    # So we need 3m > 512 => m > 170.66.
    # So m >= 171.
    
    # This confirms that the upper 1/3 [171, 256] is the natural boundary for this simple construction.
    
    # Can we do better by being more selective?
    # For example, if we include 170, we must avoid sums that are multiples of 170.
    # Multiples of 170: 340, 510.
    # 340 is not reachable (min sum 172+172=344).
    # 510 is reachable. Pairs summing to 510: (254, 256), (255, 255).
    # So if we include 170, we must exclude 254, 255, 256?
    # If we exclude 255, then (255,255) is gone.
    # If we exclude 254, then (254,256) is gone.
    # So we gain 170, lose 254 and 255 (or 256). Net loss.
    
    # So it seems hard to beat 86.
    
    # Let's just return the upper 1/3.
    return S_upper_1_3
# EVOLVE-BLOCK-END
