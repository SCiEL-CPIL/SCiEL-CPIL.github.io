import time
import random

def construct():
    t_start = time.time()
    deadline = t_start + 85.0 # Safety margin for 90s limit
    N = 4096
    MAX_SIZE = 80
    
    def is_valid_addition(x, S, sums):
        """
        Check if x can be added to set S without creating duplicate triple sums.
        S is a list of elements.
        sums is a set of all current triple sums.
        """
        # 1. 3x
        s_3x = 3 * x
        if s_3x in sums:
            return False
        
        # 2. 2x + a
        for a in S:
            s_2xa = 2 * x + a
            if s_2xa in sums:
                return False
        
        # 3. x + a + b
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                s_xab = x + a + b
                if s_xab in sums:
                    return False
        
        return True

    def calculate_sums(S):
        """Calculate all triple sums for set S."""
        sums = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            # 3a
            sums.add(3*a)
            # 2a + b
            for j in range(i, n):
                b = S[j]
                sums.add(2*a + b)
                # a + b + c
                for k in range(j, n):
                    c = S[k]
                    sums.add(a + b + c)
        return sums

    def greedy_initial():
        """Generate a valid set using greedy approach."""
        S = []
        sums = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            if is_valid_addition(x, S, sums):
                S.append(x)
                # Update sums
                # Instead of recalculating all, we can add new sums
                # But for simplicity and correctness in initialization, 
                # recalculating is O(K^3) which is fine for K~16.
                sums = calculate_sums(S)
        
        return S, sums

    def local_search(S, sums):
        """Improve the set using local search."""
        best_S = list(S)
        best_sums = sums
        best_size = len(S)
        
        # Pre-compute a list of all numbers for random selection
        all_numbers = list(range(1, N + 1))
        
        while time.time() < deadline:
            current_S = list(S)
            current_sums = sums
            improved = False
            
            # 1. Try to add a random element
            # To speed up, we can keep a list of unused elements
            # But since N=4096, picking random is fine.
            # We need to know which are used.
            used_set = set(current_S)
            
            # Try adding
            for _ in range(50): # Try 50 random additions
                if not current_S: break
                x = random.choice(all_numbers)
                if x in used_set:
                    continue
                if is_valid_addition(x, current_S, current_sums):
                    current_S.append(x)
                    used_set.add(x)
                    # Update sums incrementally for speed? 
                    # For now, recalculate if we accept, or check validity first.
                    # is_valid_addition already checked against current_sums.
                    # We need to update current_sums with new sums involving x.
                    new_sums = set()
                    new_sums.add(3*x)
                    for a in current_S[:-1]: # Exclude x itself from 'a' for 2x+a? No, S is current_S before append.
                        # Wait, current_S was appended. Let's use the pre-append list for logic.
                        pass
                    
                    # Let's just recalculate sums for simplicity in the inner loop if we accept?
                    # No, that's slow.
                    # Let's update sums manually.
                    # S_old = current_S[:-1]
                    S_old = current_S[:-1]
                    new_sums = {3*x}
                    for a in S_old:
                        new_sums.add(2*x + a)
                    for i in range(len(S_old)):
                        a = S_old[i]
                        for j in range(i, len(S_old)):
                            b = S_old[j]
                            new_sums.add(x + a + b)
                    
                    current_sums.update(new_sums)
                    improved = True
                    break # Accept first valid addition and restart loop? Or continue?
            
            if improved:
                S = current_S
                sums = current_sums
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                    best_sums = sums
                continue
            
            # 2. If no addition, try a swap
            if len(S) > 0:
                # Pick a random element to remove
                u = random.choice(S)
                S_temp = [x for x in S if x != u]
                # Remove sums involving u? 
                # This is hard to do incrementally without a map from sum to triples.
                # Recalculating sums for S_temp is O(K^3) ~ 16^3 = 4096 ops. Very fast.
                sums_temp = calculate_sums(S_temp)
                
                # Try to add a random element v
                used_set = set(S_temp)
                swapped = False
                for _ in range(20):
                    v = random.choice(all_numbers)
                    if v in used_set:
                        continue
                    if is_valid_addition(v, S_temp, sums_temp):
                        S_temp.append(v)
                        # Recalculate sums for S_temp
                        sums_temp = calculate_sums(S_temp)
                        S = S_temp
                        sums = sums_temp
                        swapped = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = list(S)
                            best_sums = sums
                        break
                
                if not swapped:
                    # Try to remove u to allow future additions?
                    # Or just accept the removal if it leads to a better state later?
                    # For now, keep S as is if no swap found.
                    pass
        
        return best_S

    # Initial greedy
    S, sums = greedy_initial()
    
    # Local search
    best_S = local_search(S, sums)
    
    # Final check and return
    # Ensure the returned set is valid
    final_sums = calculate_sums(best_S)
    if len(final_sums) == len(best_S)*3 + len(best_S)*(len(best_S)-1)//2 + len(best_S)*(len(best_S)-1)*(len(best_S)-2)//6:
         # This formula is for number of unique triples.
         # Total triples with repetition from K elements is C(K+2, 3).
         # If len(final_sums) == C(K+2, 3), it is valid.
         pass
    else:
         # Fallback to greedy if something went wrong
         S, sums = greedy_initial()
         best_S = S

    return best_S

