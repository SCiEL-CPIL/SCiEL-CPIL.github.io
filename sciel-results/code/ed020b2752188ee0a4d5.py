"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 55  # safety margin before 60s kill
    
    def reciprocal_sum(S):
        """Compute exact reciprocal sum as Fraction."""
        total = Fraction(0, 1)
        for x in S:
            total += Fraction(1, x)
        return total
    
    def valid_addition(S, candidate):
        """
        Check if adding candidate to set S creates a 3-term AP.
        S is a set for O(1) lookups.
        """
        if candidate in S:
            return False
        
        # 1. Candidate is the middle term: a, candidate, c
        # a + c = 2 * candidate => c = 2 * candidate - a
        for a in S:
            c = 2 * candidate - a
            if c in S and c != a:
                return False
        
        # 2. Candidate is the first term: candidate, b, c
        # candidate + c = 2 * b => c = 2 * b - candidate
        for b in S:
            c = 2 * b - candidate
            if c in S and c != b:
                return False
        
        # 3. Candidate is the last term: a, b, candidate
        # a + candidate = 2 * b => a = 2 * b - candidate
        for b in S:
            a = 2 * b - candidate
            if a in S and a != b:
                return False
        
        return True
    
    def greedy_initial(order=None):
        """Greedy: add elements in given order (or 1..256) if valid."""
        if order is None:
            order = list(range(1, 257))
        selected = set()
        for x in order:
            if valid_addition(selected, x):
                selected.add(x)
        return selected
    
    def local_search(S, max_iter=100):
        """
        Local search with swaps and additions.
        S is a set.
        """
        S = set(S)
        best_sum = reciprocal_sum(S)
        best_S = set(S)
        
        for _ in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            improved = False
            
            # Try single addition
            # We prioritize adding small numbers (high reciprocal)
            candidates_to_add = [x for x in range(1, 257) if x not in S]
            # Sort by value descending to find improvements faster?
            # Or just iterate. Since we want max sum, smaller x is better.
            # But we must check validity.
            for cand in sorted(candidates_to_add, key=lambda x: -1/x):
                if valid_addition(S, cand):
                    new_S = S.union([cand])
                    new_sum = best_sum + Fraction(1, cand)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_S = new_S
                        S = new_S
                        improved = True
                        break # Restart loop to re-evaluate additions with new S
            
            # Try swaps: remove one, add one
            if not improved:
                # To save time, we can try swapping larger elements with smaller ones
                # Or just iterate all elements in S and all candidates not in S
                # This is O(N * N * N) roughly.
                # Optimization: Only try swapping if we can potentially improve.
                # If we remove x and add y, gain is 1/y - 1/x.
                # So we only care if y < x.
                
                # Create list of elements in S
                S_list = list(S)
                for elem in S_list:
                    # Potential candidates to add are those smaller than elem (to improve sum)
                    # But we can also swap to a larger number if it allows adding other smaller numbers later?
                    # For local search, simple 1-swap usually improves by replacing a large x with a smaller y.
                    # Let's check all y not in S.
                    for cand in range(1, 257):
                        if cand in S:
                            continue
                        if cand >= elem:
                            continue # No immediate improvement in reciprocal sum
                    
                    # This inner loop is inefficient if we don't break.
                    # Let's just pick the best candidate for this elem?
                    # No, we need to check validity.
                    
                    # Efficient approach:
                    # For a fixed elem to remove, find the smallest cand not in S \ {elem} that is valid.
                    # If found, and 1/cand > 1/elem, it's an improvement.
                    
                    S_temp = S - {elem}
                    best_cand = None
                    for cand in range(1, elem): # Only smaller candidates improve sum
                        if cand in S_temp:
                            continue
                        if valid_addition(S_temp, cand):
                            best_cand = cand
                            break # Smallest cand is best
                    
                    if best_cand is not None:
                        new_S = S_temp.union([best_cand])
                        new_sum = best_sum - Fraction(1, elem) + Fraction(1, best_cand)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_S = new_S
                            S = new_S
                            improved = True
                            break # Restart loop
                
            if not improved:
                break
        
        return best_S

    # Phase 1: Greedy initial solution
    # Try different orderings for greedy to get diverse starts
    S1 = greedy_initial(order=list(range(1, 257)))
    S2 = greedy_initial(order=list(range(256, 0, -1)))
    S3 = greedy_initial(order=sorted(range(1, 257), key=lambda x: random.random()))
    
    best_S = max([S1, S2, S3], key=lambda s: reciprocal_sum(s))
    best_sum = reciprocal_sum(best_S)
    
    # Phase 2: Local search on the best initial
    best_S = local_search(best_S, max_iter=20)
    best_sum = reciprocal_sum(best_S)
    
    # Phase 3: Random restarts
    restarts = 0
    while (time.time() - start_time) < DEADLINE - 2:
        restarts += 1
        # Create a perturbed version of the current best
        S_perturbed = set(best_S)
        
        # Remove a few random elements
        num_remove = random.randint(1, 5)
        to_remove = random.sample(list(S_perturbed), min(num_remove, len(S_perturbed)))
        for r in to_remove:
            S_perturbed.remove(r)
        
        # Re-add greedily with random order
        order = list(range(1, 257))
        random.shuffle(order)
        S_new = greedy_initial(order=order)
        
        # Merge with perturbed set? Or just use the new greedy result?
        # Let's try to add elements from S_perturbed to S_new?
        # Simpler: Just run local search on the new greedy result.
        
        S_new = local_search(S_new, max_iter=10)
        
        new_sum = reciprocal_sum(S_new)
        if new_sum > best_sum:
            best_sum = new_sum
            best_S = S_new
    
    # Final polish
    best_S = local_search(best_S, max_iter=50)
    
    return list(best_S)
# EVOLVE-BLOCK-END
