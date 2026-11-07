import time

def construct():
    N = 128
    # Start with a known valid core
    S = set(range(64, N + 1))
    
    # Helper to check if adding x to S creates a violation
    def is_valid_addition(x, S):
        # Check all pairs a, b in S
        # If a+b in S AND x+a in S AND x+b in S, violation
        # Note: x+a != x, x+b != x
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                if a + b in S:
                    if x + a in S and x + b in S:
                        return False
        return True

    # Greedy addition from high to low
    # This tends to keep sums high, reducing conflicts
    for x in range(63, 0, -1):
        if x not in S:
            if is_valid_addition(x, S):
                S.add(x)
                
    return sorted(S)
