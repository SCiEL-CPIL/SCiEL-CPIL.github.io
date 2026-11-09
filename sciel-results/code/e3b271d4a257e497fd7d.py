import time
import random

N = 1000
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    # Data structures
    S = []
    in_S = [False] * (N + 1)
    counts = [0] * (MAX_SUM + 1)
    
    def add_element(x):
        """Add x to S, updating counts. Assumes x is not in S and addition is valid."""
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
                
    def remove_element(x):
        """Remove x from S, updating counts. Assumes x is in S."""
        # Remove x from S list
        # Note: S.remove is O(n). For better performance, we could use a set for membership 
        # and a list for iteration, but we need to keep them in sync.
        # Given N=1000, O(n) removal is acceptable if not too frequent.
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1
                
    def try_add(x):
        """Try to add x to S. Returns True if successful."""
        if in_S[x]:
            return False
        if counts[2 * x] >= 2:
            return False
        for y in S:
            s = x + y
            if counts[s] >= 2:
                return False
        add_element(x)
        return True

    def find_pairs_for_sum(s):
        """Find all pairs (a,b) in S such that a+b=s and a<=b."""
        pairs = []
        # Iterate through S
        # Optimization: Since S is small (max ~700-800?), linear scan is fine.
        # We can break early if a > s/2
        for i in range(len(S)):
            a = S[i]
            if a > s // 2:
                break # Assuming S is sorted? We don't keep it sorted. 
                # So we can't break. We must check all.
            b = s - a
            if b < a:
                continue
            if b > N:
                continue
            if in_S[b]:
                pairs.append((a, b))
        return pairs

    def get_blockers(c):
        """Get elements in S that block adding c."""
        blockers = set()
        
        # Check 2*c
        if counts[2 * c] == 2:
            pairs = find_pairs_for_sum(2 * c)
            for a, b in pairs:
                blockers.add(a)
                blockers.add(b)
                
        # Check c + y
        for y in S:
            s = c + y
            if counts[s] == 2:
                pairs = find_pairs_for_sum(s)
                for a, b in pairs:
                    blockers.add(a)
                    blockers.add(b)
                    
        return blockers

    def is_valid_swap(a, c):
        """
        Check if swapping a (in S) with c (not in S) is valid.
        a is removed, c is added.
        """
        if a == c:
            return False
        if not in_S[a] or in_S[c]:
            return False
            
        # We need to check the net effect on counts.
        # Affected sums:
        # Removed: 2*a, a+y for y in S, y!=a
        # Added: 2*c, c+y for y in S, y!=a
        
        # We can simulate the changes on a temporary dict or just check the final counts.
        # Let's build a dict of deltas.
        deltas = {}
        
        # Subtract a's contributions
        deltas[2 * a] = deltas.get(2 * a, 0) - 1
        for y in S:
            if y != a:
                s = a + y
                deltas[s] = deltas.get(s, 0) - 1
                
        # Add c's contributions
        deltas[2 * c] = deltas.get(2 * c, 0) + 1
        for y in S:
            if y != a:
                s = c + y
                deltas[s] = deltas.get(s, 0) + 1
                
        # Check if any count exceeds 2
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
            if new_count < 0:
                return False # Should not happen if logic is correct
        
        return True

    def perform_swap(a, c):
        """Perform the swap a->c. Assumes is_valid_swap(a,c) is True."""
        remove_element(a)
        add_element(c)

    def greedy_build(order):
        """Greedy build given a specific order."""
        for x in order:
            if time.time() - start_time > time_limit:
                return
            try_add(x)

    # --- Step 1: Initialization ---
    # Try a few greedy strategies
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [random.randint(1, N) for _ in range(N)], # Random
    ]
    
    # To ensure we keep the best result, we should store the state.
    # However, greedy is fast. Let's just run one good one first.
    # Descending often works well for sumsets? Or ascending?
    # Let's try ascending first.
    greedy_build(list(range(1, N + 1)))
    
    best_size = len(S)
    best_S = S.copy()
    best_in_S = in_S.copy()
    best_counts = counts.copy()

    # --- Step 2: Local Search ---
    # We will run local search on the current state.
    # If we find a larger set, we update best.
    
    # To allow multiple attempts/restarts, we might want to save/restore state.
    # But copying the whole state (S, in_S, counts) is expensive.
    # Instead, we can just run the search on the current state.
    # If we get stuck, we can perturb.
    
    rng = random.Random(42)
    no_improve_count = 0
    last_improve_time = start_time
    
    while time.time() - start_time < time_limit:
        # Check for stagnation
        if time.time() - last_improve_time > 5.0:
            # Perturb: remove 5 random elements
            if len(S) > 10:
                for _ in range(5):
                    if S:
                        a = rng.choice(S)
                        remove_element(a)
            last_improve_time = time.time()
            no_improve_count = 0
            continue
            
        # Pick a random candidate c not in S
        c = rng.randint(1, N)
        if in_S[c]:
            continue
            
        # Try to add c
        if try_add(c):
            no_improve_count = 0
            last_improve_time = time.time()
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_in_S = in_S.copy()
                best_counts = counts.copy()
            continue
            
        # If add failed, try targeted swap
        blockers = get_blockers(c)
        
        if blockers:
            # Try to swap with a random blocker
            # Shuffle blockers to avoid bias
            blocker_list = list(blockers)
            rng.shuffle(blocker_list)
            
            swapped = False
            for a in blocker_list:
                if is_valid_swap(a, c):
                    perform_swap(a, c)
                    swapped = True
                    no_improve_count = 0
                    last_improve_time = time.time()
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                        best_in_S = in_S.copy()
                        best_counts = counts.copy()
                    break
            
            if not swapped:
                no_improve_count += 1
        else:
            # No blockers? That means counts were < 2 for all sums c touches.
            # But try_add failed. This implies counts[s] >= 2 for some s, 
            # but find_pairs_for_sum might have failed to find pairs? 
            # Or counts[s] > 2? No, we maintain counts <= 2.
            # If try_add failed, it means counts[s] >= 2 for some s.
            # If get_blockers returns empty, it means for all s with counts[s]==2, 
            # find_pairs_for_sum(s) returned empty.
            # This can happen if the pairs involve elements that are no longer in S?
            # No, counts are updated.
            # It can happen if we missed a pair in find_pairs_for_sum.
            # Let's just treat it as no move.
            no_improve_count += 1
            
    # Restore best state? 
    # The current S might be smaller than best_S if we perturbed and didn't recover.
    # We should return the best found.
    # However, restoring the state is tricky if we don't have a snapshot.
    # We have best_S, best_in_S, best_counts.
    # We can just return best_S.
    
    return best_S
