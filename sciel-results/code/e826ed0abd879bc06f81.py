import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Precompute all triples
    all_triples = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                all_triples.append((a, b, c))
    
    # Map triple to index for quick lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    N = len(all_triples)
    
    # Precompute for each triple, the list of other 7 triples in each K(2,2,2) containing it
    # Actually, we just need to check if the set of 7 is present.
    # Let's precompute the 7-tuples of indices for each K(2,2,2) configuration that contains triple i.
    # Or simpler: for each triple i, and each choice of u', v', w', the 7 other triples are fixed.
    
    best_m = 105
    best_set = set()
    # Initialize with the star
    for t in all_triples:
        if 0 in t:
            best_set.add(t)
            
    # We will try to improve by adding random triples.
    # We can also try to remove some star triples to allow more others?
    # The star is a valid configuration.
    # Let's try a greedy random addition from a random permutation.
    
    for _ in range(100): # Max 100 restarts, but check time
        if time.time() > deadline - 5:
            break
            
        random.shuffle(all_triples)
        current_set = set()
        # Start with a random subset? Or empty?
        # Starting empty and adding greedily often works well for independence-like problems.
        # But the star is a strong baseline.
        # Let's try starting with the star and adding others.
        # Or just start empty and see if we can beat 105.
        
        current_set = set()
        
        for t in all_triples:
            if time.time() > deadline - 5:
                break
                
            # Check if adding t violates K(2,2,2)
            valid = True
            a, b, c = t
            # u, v, w are a, b, c
            # We need to check all u', v', w'
            # The 7 other triples are:
            # (u, v, w'), (u, v', w), (u, v', w'), (u', v, w), (u', v, w'), (u', v', w), (u', v', w')
            
            for u_prime in range(16):
                if u_prime in (a, b, c): continue
                for v_prime in range(16):
                    if v_prime in (a, b, c, u_prime): continue
                    for w_prime in range(16):
                        if w_prime in (a, b, c, u_prime, v_prime): continue
                        
                        # The 7 other triples
                        t1 = (a, b, w_prime)
                        t2 = (a, v_prime, c)
                        t3 = (a, v_prime, w_prime)
                        t4 = (u_prime, b, c)
                        t5 = (u_prime, b, w_prime)
                        t6 = (u_prime, v_prime, c)
                        t7 = (u_prime, v_prime, w_prime)
                        
                        # Check if all are in current_set
                        if (t1 in current_set and t2 in current_set and t3 in current_set and
                            t4 in current_set and t5 in current_set and t6 in current_set and
                            t7 in current_set):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
            
            if valid:
                current_set.add(t)
        
        if len(current_set) > best_m:
            best_m = len(current_set)
            best_set = current_set
            
    # Convert set back to list of lists
    return [list(t) for t in best_set]
