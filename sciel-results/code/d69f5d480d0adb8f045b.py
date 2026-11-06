import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    N = 284
    
    def is_valid_addition(S, sums, x):
        """Check if x can be added to S without creating duplicate sums."""
        # Check x + y for y in S
        for y in S:
            if x + y in sums:
                return False
        # Check 2*x
        if 2 * x in sums:
            return False
        return True

    def get_new_sums(S, x):
        """Get the new sums created by adding x to S."""
        new_sums = set()
        for y in S:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    def compute_sums(S):
        """Compute all pairwise sums of S."""
        sums = set()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    def greedy_fill(S, sums, candidates, order='random'):
        """
        Greedily add candidates to S.
        candidates: list of integers to consider
        order: 'random' or 'sorted'
        Returns updated S and sums.
        """
        S = list(S)
        S_set = set(S)
        
        # Determine order
        if order == 'random':
            random.shuffle(candidates)
        elif order == 'sorted':
            candidates = sorted(candidates)
        elif order == 'sorted_desc':
            candidates = sorted(candidates, reverse=True)
            
        for x in candidates:
            if x in S_set:
                continue
            if is_valid_addition(S, sums, x):
                S.append(x)
                S_set.add(x)
                new_sums = get_new_sums(S, x)
                sums |= new_sums
                
        return S, sums

    def local_search_improve(S, sums, max_iters=1000):
        """
        Try to increase size by adding elements.
        If stuck, try swaps to find a better configuration that allows addition.
        """
        S = list(S)
        S_set = set(S)
        current_size = len(S)
        
        # First, try to add any remaining valid element
        # We can do this efficiently by iterating through all possible x
        # and checking validity.
        
        # To speed up, we can keep a list of "blocked" elements or just scan.
        # Scanning 1..284 is fast.
        
        improved = True
        while improved and len(S) < 20: # Max possible is 20
            improved = False
            # Try to add any element
            added = False
            # Randomize the order of candidates to break symmetry
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_addition(S, sums, x):
                    S.append(x)
                    S_set.add(x)
                    sums |= get_new_sums(S, x)
                    current_size += 1
                    improved = True
                    added = True
                    break # Restart scan as sums changed
            
            if not added:
                # No single addition possible. Try a swap?
                # Swap a non-fixed element with a candidate.
                # This doesn't increase size, but might allow a future addition.
                # However, pure local search on size only accepts size-increasing moves.
                # To escape, we need a kick. This is handled by the outer loop.
                break
                
        return S, sums

    def kick_and_refill(S, sums, kick_size=2):
        """
        Remove 'kick_size' random non-fixed elements and greedily refill.
        Returns new S and sums.
        """
        S = list(S)
        S_set = set(S)
        non_fixed = [x for x in S if x not in fixed_set]
        
        if len(non_fixed) < kick_size:
            return S, sums
            
        # Remove random elements
        to_remove = set(random.sample(non_fixed, kick_size))
        S = [x for x in S if x not in to_remove]
        S_set = set(S)
        sums = compute_sums(S)
        
        # Refill greedily
        # Use a few different orders to see which yields a larger set
        best_S, best_sums = S, sums
        best_size = len(S)
        
        for order in ['random', 'sorted', 'sorted_desc']:
            trial_S, trial_sums = S, sums
            trial_S, trial_sums = greedy_fill(trial_S, trial_sums, list(range(1, N + 1)), order)
            if len(trial_S) > best_size:
                best_size = len(trial_S)
                best_S = trial_S
                best_sums = trial_sums
                
        return best_S, best_sums

    # Main Search Loop
    best_S = []
    best_sums = set()
    best_size = 0
    
    # Initial Greedy
    initial_S = list(fixed_elements)
    initial_sums = compute_sums(initial_S)
    best_S, best_sums = greedy_fill(initial_S, initial_sums, list(range(1, N + 1)), 'random')
    best_size = len(best_S)
    
    # Iterated Local Search
    while time.time() < deadline:
        # If we found 20, we can stop or try to verify? 20 is max.
        if best_size >= 20:
            break
            
        # Choose a kick size. Smaller kicks are more likely to preserve good structure.
        # Larger kicks explore more.
        if random.random() < 0.7:
            kick_size = 1
        elif random.random() < 0.5:
            kick_size = 2
        else:
            kick_size = 3
            
        # Perform kick from current best
        new_S, new_sums = kick_and_refill(best_S, best_sums, kick_size)
        
        # Local search to maximize size of the new candidate
        new_S, new_sums = local_search_improve(new_S, new_sums)
        
        if len(new_S) > best_size:
            best_size = len(new_S)
            best_S = new_S
            best_sums = new_sums
            
            # If we improved, we can try to improve further immediately
            # by doing a few more local search steps or kicks
            # But the loop handles this.
            
        # Occasionally, restart from scratch to explore different regions
        if random.random() < 0.05:
            initial_S = list(fixed_elements)
            initial_sums = compute_sums(initial_S)
            new_S, new_sums = greedy_fill(initial_S, initial_sums, list(range(1, N + 1)), 'random')
            new_S, new_sums = local_search_improve(new_S, new_sums)
            if len(new_S) > best_size:
                best_size = len(new_S)
                best_S = new_S
                best_sums = new_sums

    return best_S

# Quick self-test
if __name__ == "__main__":
    # Set seed for reproducibility in testing
    random.seed(42)
    res = construct()
    print(f"Found size: {len(res)}")
    print(f"Elements: {sorted(res)}")
    
    # Verify validity
    sums = set()
    valid = True
    S = sorted(res)
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                valid = False
                print(f"Conflict: {S[i]} + {S[j]} = {s}")
            sums.add(s)
            
    # Check fixed elements
    for f in [1, 2, 5, 11]:
        if f not in res:
            valid = False
            print(f"Missing fixed element: {f}")
            
    # Check range
    for x in res:
        if x < 1 or x > 284:
            valid = False
            print(f"Out of range: {x}")
            
    print(f"Valid: {valid}")
