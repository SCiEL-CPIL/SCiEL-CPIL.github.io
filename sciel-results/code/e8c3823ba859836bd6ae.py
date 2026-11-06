import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for 60s limit
    
    n = 4096
    best_set = []
    best_size = 0
    
    def check_validity(S):
        """Verify S is a valid B_3[1] set. Used for final check or debugging."""
        sums = set()
        for a, b, c in combinations_with_replacement(S, 3):
            s = a + b + c
            if s in sums:
                return False
            sums.add(s)
        return True

    def run_greedy(order):
        """
        Greedily add elements from 'order' to a set S.
        Returns the final set S.
        """
        S = []
        T = set()  # Set of all triple sums from S
        
        for x in order:
            # Generate new sums involving x
            new_sums = set()
            
            # 1. x + x + x
            new_sums.add(3 * x)
            
            # 2. x + x + s for s in S
            for s in S:
                new_sums.add(2 * x + s)
                
            # 3. x + s1 + s2 for s1 <= s2 in S
            for i in range(len(S)):
                s1 = S[i]
                new_sums.add(x + 2 * s1)
                for j in range(i + 1, len(S)):
                    s2 = S[j]
                    new_sums.add(x + s1 + s2)
            
            # Check if new sums are valid
            # 1. No duplicates within new_sums
            if len(new_sums) != len(new_sums): # This is always true for a set, so check list vs set if we used list. 
                # Since we used a set, this check is redundant unless we generated a list.
                pass
            
            # Actually, we need to ensure new_sums are distinct from each other.
            # Using a set `new_sums` automatically handles distinctness among new sums.
            # But wait, if we added duplicates to the set, the size would be smaller than the count of generated sums.
            # Let's count generated sums.
            count_generated = 1 + len(S) + len(S)*(len(S)+1)//2
            if len(new_sums) != count_generated:
                # Collision among new sums
                continue
                
            # 2. No overlap with existing sums T
            if new_sums & T:
                continue
                
            # Valid, add x
            S.append(x)
            T.update(new_sums)
            
        return S

    # Generate candidate pool
    # We can use the full range [1, 4096]
    candidates = list(range(1, n + 1))
    
    # Define various orders
    orders = []
    
    # 1. Ascending
    orders.append(candidates)
    
    # 2. Descending
    orders.append(list(reversed(candidates)))
    
    # 3. Random shuffles
    # We will generate these on the fly in the loop to save memory, 
    # but we need to ensure we try different ones.
    
    # 4. Biased: Start with small, then random? Or start with large?
    # Let's just use random shuffles for the bulk of the trials.
    
    # Run the deterministic ones first
    for order in orders:
        if time.time() - start_time > time_limit:
            break
        S = run_greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Now run random trials
    # Estimate how many trials we can run.
    # Each trial takes roughly 0.05s - 0.1s depending on set size.
    # We have ~55s.
    
    while time.time() - start_time < time_limit - 2.0: # Keep 2s buffer
        random_order = candidates.copy()
        random.shuffle(random_order)
        S = run_greedy(random_order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        # Optional: Try a biased random order
        # e.g., sort by (random.random(), x) or something
        # For now, pure random shuffle is a good baseline.
        
    # Final verification (optional, but good for safety if we want to be sure)
    # The greedy check is strict, so best_set should be valid.
    # However, if we want to be absolutely sure, we can check.
    # Given the time, we skip full verification unless best_size is 0.
    
    return best_set
