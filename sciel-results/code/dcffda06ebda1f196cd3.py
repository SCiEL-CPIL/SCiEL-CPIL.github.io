import time
import random

def is_valid(S):
    S = sorted(S)
    k = len(S)
    prods = set()
    for i in range(k):
        p = 1
        for j in range(i, k):
            p *= S[j]
            if p in prods:
                return False
            prods.add(p)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Greedy Initialization
    S = []
    products = set()
    for x in range(2, 129):
        tails = [x]
        prod = x
        for a in reversed(S):
            prod *= a
            tails.append(prod)
        
        if not products.intersection(tails):
            S.append(x)
            products.update(tails)
            
    best_S = S[:]
    best_len = len(S)
    
    # 2. Local Search
    # We use a simple hill climbing with random restarts/shuffles if stuck.
    # To prevent getting stuck in a local optimum of same size, we allow 
    # 1-out-1-in moves but only keep them if they lead to a 1-in later.
    # Simpler: Just try to increase size. If stuck for a while, restart greedy 
    # with a different random order? No, greedy is deterministic.
    # We can randomize the greedy order.
    
    # Let's try multiple greedy starts with random permutations of [2, 128]?
    # No, the constraint depends on values, not order of addition?
    # Wait, the "tails" calculation depends on the current sequence.
    # If we add numbers in different order, the set of "current products" is the same 
    # (all subsequence products of the set), but the "tails" (products ending at the new element) 
    # are different?
    # No, the condition for adding $x$ is:
    # $\{ x \cdot \prod_{y \in T} y \mid T \text{ is a suffix of } S \} \cap \text{Products}(S) = \emptyset$.
    # The set of suffixes depends on the order of $S$.
    # But $S$ is always sorted in the final output.
    # In the greedy loop, $S$ is built in increasing order.
    # So the order is fixed.
    
    # So we rely on local search.
    
    current_S = S[:]
    
    while time.time() < deadline:
        improved = False
        
        # Try 1-in
        candidates = [x for x in range(2, 129) if x not in current_S]
        if candidates:
            random.shuffle(candidates)
            for y in candidates:
                new_S = current_S + [y]
                if is_valid(new_S):
                    current_S = new_S
                    if len(current_S) > best_len:
                        best_S = current_S[:]
                        best_len = len(current_S)
                    improved = True
                    break
        
        # If no 1-in, try 1-out-2-in
        if not improved and len(current_S) > 0:
            x = random.choice(current_S)
            S_temp = [v for v in current_S if v != x]
            candidates = [x for x in range(2, 129) if x not in S_temp]
            if len(candidates) >= 2:
                # Try random pairs
                # Limit attempts to avoid spending all time on one bad swap
                attempts = 0
                max_attempts = 50
                while attempts < max_attempts:
                    y, z = random.sample(candidates, 2)
                    new_S = S_temp + [y, z]
                    if is_valid(new_S):
                        current_S = new_S
                        if len(current_S) > best_len:
                            best_S = current_S[:]
                            best_len = len(current_S)
                        improved = True
                        break
                    attempts += 1
        
        # If no improvement, try 1-out-1-in to shuffle
        if not improved and len(current_S) > 0:
            x = random.choice(current_S)
            S_temp = [v for v in current_S if v != x]
            candidates = [x for x in range(2, 129) if x not in S_temp]
            if candidates:
                random.shuffle(candidates)
                # Try a few
                for y in candidates[:20]:
                    new_S = S_temp + [y]
                    if is_valid(new_S):
                        current_S = new_S
                        # Do not update best if length same
                        improved = True
                        break
                        
        # If completely stuck, maybe random restart greedy?
        # Or just continue.
        
    return best_S
