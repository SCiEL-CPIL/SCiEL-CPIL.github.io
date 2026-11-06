import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 256
    
    def is_valid_with(S, x):
        """
        Check if x can be added to set S.
        S is a set of ints.
        """
        S_list = list(S)
        
        # Case 1: x is a
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
            
        # Case 3: x is c
        for a in S_list:
            if a >= x:
                continue
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def greedy(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_with(S, x):
                S.add(x)
        return S

    def polish(S):
        """
        Add any valid elements from [1, n] not in S.
        """
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if time.time() > deadline - 2.0:
                    return S
                if x not in S:
                    if is_valid_with(S, x):
                        S.add(x)
                        changed = True
        return S

    best_S = set()
    
    # 1. Initial Greedy Strategies
    strategies = [
        list(range(n, 0, -1)), # Descending
        list(range(1, n + 1)), # Ascending
    ]
    
    # Add some random strategies
    for i in range(10):
        random.seed(i)
        order = list(range(1, n + 1))
        random.shuffle(order)
        strategies.append(order)
        
    for order in strategies:
        if time.time() > deadline - 10.0:
            break
        S = greedy(order)
        S = polish(S)
        if len(S) > len(best_S):
            best_S = S
            
    # 2. Local Search (Swaps)
    # Try to improve best_S by swapping elements
    current_S = set(best_S)
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        improved = False
        # Try a few random swaps
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            if len(current_S) == 0:
                break
            u = random.choice(list(current_S))
            v = random.choice([x for x in range(1, n + 1) if x not in current_S])
            
            S_new = (current_S - {u}) | {v}
            
            # Check if S_new is valid
            # Note: S_new is a set of size |current_S|.
            # We need to check if S_new itself is a valid set?
            # No, the condition is on the set itself.
            # If we swap u with v, we need to ensure the new set is valid.
            # This requires checking all triples in S_new.
            # This is O(|S|^3) which is expensive.
            
            # However, we can just run polish on S_new?
            # No, polish assumes S_new is valid and tries to add.
            # If S_new is invalid, polish will fail to add anything, but S_new itself is invalid.
            # So we must check if S_new is a valid set.
            
            # Efficient check for validity of S_new:
            # Since S_new differs from current_S by one element,
            # we only need to check triples involving v.
            # Triples involving u are removed.
            # Triples not involving u or v are unchanged (and were valid in current_S).
            
            # So, check if v can be added to (current_S - {u}).
            # This is exactly is_valid_with(current_S - {u}, v).
            
            S_base = current_S - {u}
            if is_valid_with(S_base, v):
                # S_new is valid
                S_new = S_base | {v}
                S_polished = polish(S_new)
                if len(S_polished) > len(current_S):
                    current_S = S_polished
                    improved = True
                    no_improve_count = 0
                    break
                elif len(S_polished) == len(current_S):
                    # Same size, but different composition.
                    # Might be worth keeping to escape local optimum?
                    # Let's accept with low probability or if it allows future swaps?
                    # For now, let's just track it.
                    # If we don't accept, we might get stuck.
                    # Let's accept if random() < 0.1
                    if random.random() < 0.1:
                        current_S = S_polished
                        improved = True # Count as progress to reset no_improve?
                        # Actually, if size is same, it's not progress.
                        # But it might lead to progress later.
                        no_improve_count = 0 # Reset to allow more search
                        break
            # If not improved, continue trying swaps
        
        if not improved:
            no_improve_count += 1
            
    return sorted(current_S)
