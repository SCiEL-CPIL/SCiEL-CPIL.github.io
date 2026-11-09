import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We want to find a set S of 32 distinct integers in 1..512
    # that maximizes the number of distinct d > 0 such that
    # a, a+d, a+2d are in S.
    
    def score(S):
        S_set = set(S)
        diffs = set()
        list_S = list(S)
        n = len(list_S)
        for i in range(n):
            for j in range(i+1, n):
                a = list_S[i]
                c = list_S[j]
                if (a + c) % 2 == 0:
                    b = (a + c) // 2
                    if b in S_set:
                        d = (c - a) // 2
                        if d > 0:
                            diffs.add(d)
        return len(diffs)

    best_score = -1
    best_set = None
    all_numbers = list(range(1, 513))
    
    # Try a few restarts to escape local optima
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Initialize with a random set
        current_set = set(random.sample(all_numbers, 32))
        current_score = score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # Simulated Annealing
        T = 5.0
        min_T = 0.1
        cooling = 0.999
        no_improve = 0
        max_no_improve = 5000
        
        while T > min_T and time.time() < deadline:
            # Pick a random element to remove
            to_remove = random.choice(list(current_set))
            # Pick a random element to add (not in set)
            # To avoid slow sampling, pick from all_numbers and retry if in set
            # Or maintain a list of available numbers.
            # Since 512 is small, picking randomly and checking is fine.
            new_val = random.randint(1, 512)
            attempts = 0
            while new_val in current_set and attempts < 100:
                new_val = random.randint(1, 512)
                attempts += 1
            if new_val in current_set:
                continue
                
            # Perform move
            current_set.remove(to_remove)
            current_set.add(new_val)
            
            new_score = score(current_set)
            
            # Acceptance criterion
            if new_score >= current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_set.remove(new_val)
                current_set.add(to_remove)
                no_improve += 1
                
            T *= cooling
            
            if no_improve > max_no_improve:
                break

    # Ensure we return a valid list
    if best_set is None:
        return list(range(1, 33))
        
    return list(best_set)
