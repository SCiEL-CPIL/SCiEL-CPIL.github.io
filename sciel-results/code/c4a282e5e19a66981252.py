import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: 5s before 90s kill
    
    # Precompute all triples of distinct elements from 1..192
    # For each triple (a, b, c), store the sum and the triple
    # We'll use this to quickly check if adding a new element x to set S
    # violates the condition for any existing element in S.
    
    # Actually, let's think about efficiency.
    # Condition: For every a in S, and every triple (b, c, d) of distinct elements in S\{a},
    # a must NOT divide (b + c + d).
    
    # When we add a new element x to S, we need to check:
    # 1. For each a in S: does a divide x + b + c for any pair (b, c) from S\{a}?
    #    (Since x, b, c are 3 distinct OTHER elements for a)
    # 2. For x: does x divide b + c + d for any triple (b, c, d) from S?
    
    # Let's precompute for each possible new element x and current set S,
    # but that's too expensive. Instead, let's be smart about the greedy order.
    
    # Key insight: If we pick large numbers first, they are less likely to divide
    # small sums. Also, for a large number a, the only way a divides (b+c+d) 
    # where b,c,d < a is if b+c+d >= a and b+c+d is a multiple of a.
    # Since max(b+c+d) <= 3*191 = 573, and a >= 96, the only multiples possible
    # are a itself, 2a, 3a. So for a >= 96, b+c+d must equal a, 2a, or 3a.
    
    # Let's use a two-phase approach:
    # Phase 1: Build a core from [96, 192] using greedy (descending)
    # Phase 2: Add smaller numbers greedily, checking validity efficiently
    
    def is_valid_full(S):
        """Full validity check for set S."""
        S_list = sorted(S)
        n = len(S_list)
        if n < 4:
            return True
        for i, a in enumerate(S_list):
            others = [v for j, v in enumerate(S_list) if j != i]
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
        return True
    
    def is_valid_addition(x, S_set):
        """Check if adding x to S_set maintains validity."""
        S_list = sorted(S_set)
        
        # Check 1: For each a in S_set, does a divide x + b + c for any distinct b, c in S_set\{a}?
        # We need to check all pairs (b, c) from S_set\{a} where b != c
        # sum = x + b + c, check if sum % a == 0
        
        for a in S_list:
            if a == x:
                continue
            # Get all elements in S_set except a
            others = [v for v in S_list if v != a]
            # Check all pairs from others
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    b, c = others[i], others[j]
                    total = x + b + c
                    if total % a == 0:
                        return False
        
        # Check 2: Does x divide b + c + d for any triple (b, c, d) from S_set?
        if len(S_list) >= 3:
            for t in combinations(S_list, 3):
                if sum(t) % x == 0:
                    return False
        
        return True
    
    best_set = []
    best_size = 0
    
    # Phase 1: Greedy from large to small
    S = set()
    for x in range(192, 0, -1):
        if is_valid_addition(x, S):
            S.add(x)
    
    if len(S) > best_size:
        best_set = list(S)
        best_size = len(S)
    
    # Phase 2: Try multiple randomized restarts with different orderings
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Create a shuffled order with bias toward larger numbers
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        # Sort so larger numbers come first with some randomness
        # Use a weighted sort: key = -value + random noise
        S = set()
        for x in candidates:
            if is_valid_addition(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)
    
    # Phase 3: Local search to try to improve
    if time.time() < deadline - 5:
        current_set = set(best_set)
        improved = True
        no_improve_streak = 0
        
        while improved and time.time() < deadline - 2:
            improved = False
            no_improve_streak += 1
            if no_improve_streak > 50:
                break
            
            # Try to add any missing element
            added = False
            for x in range(192, 0, -1):
                if x not in current_set:
                    if is_valid_addition(x, current_set):
                        current_set.add(x)
                        added = True
                        improved = True
                        break
            
            if not improved:
                # Try a swap: remove one, add another
                if len(current_set) >= 2:
                    swapped = False
                    # Try several random swaps
                    for _ in range(50):
                        if time.time() > deadline - 2:
                            break
                        u = random.choice(list(current_set))
                        v = random.randint(1, 192)
                        if v in current_set:
                            continue
                        new_set = current_set - {u} | {v}
                        if is_valid_full(new_set):
                            current_set = new_set
                            swapped = True
                            # After swap, try to add more
                            for x in range(192, 0, -1):
                                if x not in current_set:
                                    if is_valid_addition(x, current_set):
                                        current_set.add(x)
                                        improved = True
                                        break
                            break
                    
                    if not swapped and not improved:
                        no_improve_streak += 10
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(best_set)
    
    return best_set
