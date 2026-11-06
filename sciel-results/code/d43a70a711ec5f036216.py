import random
import itertools
import time

def is_sidon_4(a, b, c, d):
    # Check unique sums for i<=j
    # Sums: 2a, a+b, a+c, a+d, 2b, b+c, b+d, 2c, c+d, 2d
    # Using a set is fast enough for 10 elements
    sums = {
        2*a, a+b, a+c, a+d,
        2*b, b+c, b+d,
        2*c, c+d,
        2*d
    }
    return len(sums) == 10

def count_sidon_4(S):
    count = 0
    for subset in itertools.combinations(S, 4):
        if is_sidon_4(*subset):
            count += 1
    return count

def construct():
    start_time = time.time()
    best_S = list(range(12))
    best_count = count_sidon_4(best_S)
    
    if best_count == 0:
        return best_S
        
    # Local search
    for restart in range(100):
        if time.time() - start_time > 55:
            break
            
        S = list(range(12))
        # Randomize slightly to escape local minima
        for i in range(12):
            if random.random() < 0.5:
                S[i] = random.randint(0, 128)
        
        # Ensure distinct
        while len(set(S)) < 12:
            idx = random.randint(0, 11)
            S[idx] = random.randint(0, 128)
            
        current_count = count_sidon_4(S)
        
        for step in range(1000):
            if time.time() - start_time > 55:
                break
                
            if current_count == 0:
                return S
                
            # Find a sidon subset to break
            # To save time, we can just pick a random 4-subset and check
            # Or find one. Finding one is O(495).
            sidon_subsets = []
            for sub in itertools.combinations(S, 4):
                if is_sidon_4(*sub):
                    sidon_subsets.append(sub)
                    # We can break early if we just want one, 
                    # but for count we need all. 
                    # Here we just need one to target.
                    break
            
            if not sidon_subsets:
                return S
                
            target_subset = sidon_subsets[0]
            
            # Try to move one element of target_subset
            moved = False
            # Shuffle indices to try different ones
            indices = [i for i in range(12) if S[i] in target_subset]
            random.shuffle(indices)
            
            for idx in indices:
                old_val = S[idx]
                # Try several new values
                for _ in range(30):
                    new_val = random.randint(0, 128)
                    if new_val == old_val:
                        continue
                    if new_val in S:
                        continue
                    
                    S[idx] = new_val
                    new_count = count_sidon_4(S)
                    
                    if new_count < current_count:
                        current_count = new_count
                        moved = True
                        break
                    else:
                        S[idx] = old_val
                        
                if moved:
                    break
            
            if not moved:
                # Stuck, randomize a bit
                idx = random.randint(0, 11)
                S[idx] = random.randint(0, 128)
                while len(set(S)) < 12:
                    S[idx] = random.randint(0, 128)
                current_count = count_sidon_4(S)
        
        if current_count < best_count:
            best_count = current_count
            best_S = S[:]
            
    return best_S
