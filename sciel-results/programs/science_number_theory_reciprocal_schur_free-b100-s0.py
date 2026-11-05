import time
import random

def construct():
    deadline = time.time() + 80  # safety margin
    
    def can_add(a, S):
        """
        Check if adding 'a' to valid set 'S' keeps it valid.
        S must be a set for O(1) lookup.
        """
        # Case 1: a is the smallest (x)
        # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
        a2 = a * a
        for b in S:
            if b <= a:
                continue
            d = b - a
            if a2 % d != 0:
                continue
            c = a + a2 // d
            if c > a and c != b and c in S:
                return False
        
        # Case 2: a is the middle (y)
        # 1/x = 1/a + 1/c => c = a*x / (a-x)
        # x < a < c
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            c = num // denom
            if c > a and c in S:
                return False
        
        # Case 3: a is the largest (z)
        # 1/x = 1/y + 1/a => y = a*x / (a-x)
        # x < y < a
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            y = num // denom
            if x < y < a and y in S:
                return False
        
        return True

    # Start with a known valid set: all numbers > 90
    # 1/91 > 1/180 + 1/180 ? No, 1/91 approx 0.011, 2/180 approx 0.011.
    # Actually, if a,b,c > 90, then 1/a < 1/90.
    # 1/b + 1/c < 2/90 = 1/45.
    # Is 1/a = 1/b + 1/c possible?
    # If a,b,c are all in (90, 180], then 1/a >= 1/180, 1/b+1/c >= 2/180 = 1/90.
    # Wait. 1/a is the LHS. 1/b+1/c is RHS.
    # If a,b,c > 90, then 1/a < 1/90.
    # And 1/b + 1/c < 1/90 + 1/90 = 2/90 = 1/45.
    # This doesn't immediately rule it out.
    # However, for a,b,c > N/2, it is known that no such triples exist.
    # Proof: If a < b < c, then 1/a > 1/b + 1/c > 1/c + 1/c = 2/c.
    # So a < c/2.
    # If c <= 180, then a < 90.
    # So if all elements are >= 90, no triple can exist with a < b < c where a is the smallest.
    # Because a would have to be < 90.
    # Thus, {91, ..., 180} is valid.
    
    S = set(range(91, 181))
    
    # Greedily add smaller numbers
    for a in range(1, 91):
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
    
    # Local search
    random.seed(42)
    best_S = set(S)
    best_size = len(S)
    
    # List of excluded elements
    all_nums = set(range(1, 181))
    
    # Hill climbing
    # We try to add elements or swap.
    # To speed up, we can maintain a list of candidates.
    
    for _ in range(50000): # Limit iterations to ensure time
        if time.time() > deadline:
            break
        
        excluded = all_nums - S
        if not excluded:
            break
        
        # Try to add a random excluded element
        a = random.choice(list(excluded))
        if can_add(a, S):
            S.add(a)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
        else:
            # Try to remove a random element from S and add a
            if S:
                b = random.choice(list(S))
                S.remove(b)
                if can_add(a, S):
                    S.add(a)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    # Revert
                    S.add(b)
    
    return sorted(best_S)
