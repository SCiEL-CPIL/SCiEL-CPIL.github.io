import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # All possible triples
    all_triples = list(combinations(range(16), 3))
    
    best_set = set()
    
    def creates_conflict(E, T):
        """
        Returns True if adding T to E creates a K(2,2,2).
        This is equivalent to checking if there exist a', b', c' such that
        the 7 other transversals of {a,a'}x{b,b'}x{c,c'} are in E.
        """
        a, b, c = T
        # To speed up, we can precompute neighbors? 
        # For now, direct check.
        for a_prime in range(16):
            if a_prime == a: continue
            for b_prime in range(16):
                if b_prime == b: continue
                # Check {a', b, c} in E
                if (a_prime, b, c) not in E and (a_prime, c, b) not in E and (b, a_prime, c) not in E and (b, c, a_prime) not in E and (c, a_prime, b) not in E and (c, b, a_prime) not in E:
                    # We store as sorted tuples in E, so just check sorted tuple
                    if tuple(sorted((a_prime, b, c))) not in E: continue
                # Check {a, b', c} in E
                if tuple(sorted((a, b_prime, c))) not in E: continue
                for c_prime in range(16):
                    if c_prime == c: continue
                    # Check {a, b, c'} in E
                    if tuple(sorted((a, b, c_prime))) not in E: continue
                    # Check {a', b', c} in E
                    if tuple(sorted((a_prime, b_prime, c))) not in E: continue
                    # Check {a', b, c'} in E
                    if tuple(sorted((a_prime, b, c_prime))) not in E: continue
                    # Check {a, b', c'} in E
                    if tuple(sorted((a, b_prime, c_prime))) not in E: continue
                    # Check {a', b', c'} in E
                    if tuple(sorted((a_prime, b_prime, c_prime))) not in E: continue
                    return True
        return False

    def sanitize(E):
        """Remove triples that are part of a K(2,2,2)."""
        changed = True
        while changed:
            changed = False
            # Convert to list to avoid modification during iteration
            for T in list(E):
                if creates_conflict(E, T):
                    E.discard(T)
                    changed = True
                    break # Restart loop to ensure consistency

    def greedy_fill(E):
        """Add any triple that doesn't create a conflict."""
        changed = True
        while changed:
            changed = False
            for T in all_triples:
                if T not in E:
                    if not creates_conflict(E, T):
                        E.add(T)
                        changed = True

    # Strategy: Multiple restarts with different initial densities
    # The trivial solution is 105. We expect to do much better.
    
    for init_size in [300, 350, 400, 450, 500]:
        if time.time() - start_time > timeout:
            break
            
        for restart in range(20):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization
            E = set(random.sample(all_triples, init_size))
            
            # Sanitize
            sanitize(E)
            
            # Fill
            greedy_fill(E)
            
            # Record best
            if len(E) > len(best_set):
                best_set = set(E)
                
            # If we have very little time left, stop
            if time.time() - start_time > timeout - 5:
                break

    # Final polish: Try to improve best_set with more aggressive local search?
    # The greedy fill is quite strong. 
    # Let's try a few "kicks" on the best set found.
    if time.time() - start_time < timeout - 10:
        for _ in range(50):
            if time.time() - start_time > timeout - 5:
                break
            E = set(best_set)
            # Remove random 10-20 triples
            if len(E) > 10:
                to_remove = random.sample(list(E), min(20, len(E)//5))
                for t in to_remove:
                    E.discard(t)
            
            sanitize(E)
            greedy_fill(E)
            
            if len(E) > len(best_set):
                best_set = set(E)

    # Return as list of lists
    return [list(t) for t in best_set]
