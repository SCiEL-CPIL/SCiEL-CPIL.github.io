import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # 1. Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append((mask, combo))
        
    N = len(all_sets)
    masks = [m for m, _ in all_sets]
    combos = [c for _, c in all_sets]
    
    # 2. Precompute intersection sizes matrix I[i][j]
    # N is 1820, N^2 is ~3.3M. This is fast.
    I = [[0] * N for _ in range(N)]
    for i in range(N):
        mi = masks[i]
        for j in range(i, N):
            inter = bin(mi & masks[j]).count('1')
            I[i][j] = inter
            I[j][i] = inter

    # Helper to check if adding c to chosen_indices creates a sunflower
    # Returns True if conflict, False otherwise
    def check_conflict(c, chosen_indices):
        # Group chosen indices by their intersection size with c
        groups = {}
        for idx in chosen_indices:
            s = I[idx][c]
            if s not in groups:
                groups[s] = []
            groups[s].append(idx)
        
        for s, group in groups.items():
            if len(group) < 2:
                continue
            # Check if any pair in group has intersection size s
            # We can optimize this by just checking pairs
            # If group is large, this is O(L^2)
            # But L is usually small for specific s
            for i in range(len(group)):
                a = group[i]
                for j in range(i + 1, len(group)):
                    b = group[j]
                    if I[a][b] == s:
                        return True
        return False

    def is_valid_family(indices):
        # Check if the whole family is valid
        # This is O(K^3) which is too slow for large K
        # We rely on incremental checks during construction
        pass

    best_family_indices = []
    random.seed(42)
    
    # We will use a loop of "Greedy + Perturbation"
    # This is an Iterated Local Search (ILS)
    
    while time.time() < deadline:
        # 1. Greedy Phase
        # Pick sets in a random order, but with a slight bias?
        # Just random shuffle is a good baseline.
        indices = list(range(N))
        random.shuffle(indices)
        
        chosen_indices = []
        chosen_set = set()
        
        for idx in indices:
            if not check_conflict(idx, chosen_indices):
                chosen_indices.append(idx)
                chosen_set.add(idx)
                
        # Update best
        if len(chosen_indices) > len(best_family_indices):
            best_family_indices = chosen_indices[:]
        
        # 2. Local Search / Perturbation Phase
        # Try to improve the current family
        improved = True
        # Limit the number of local search steps to avoid getting stuck in one run
        local_steps = 0
        max_local_steps = 5000
        
        while improved and local_steps < max_local_steps and time.time() < deadline:
            improved = False
            local_steps += 1
            
            # Try to add a set
            # To speed this up, we can pick a random candidate from the complement
            # But to be thorough, we might want to try all?
            # N is 1820. K is maybe 100-200.
            # Checking all N candidates takes N * K^2 time.
            # 1820 * 200^2 is too slow.
            # We pick a random sample of candidates.
            
            # Get unchosen indices
            # This is O(N)
            unchosen = [i for i in range(N) if i not in chosen_set]
            
            if not unchosen:
                break
                
            # Pick a random candidate
            # If we pick randomly, we might miss the one that fits.
            # Better: Pick a few random candidates.
            # Or, iterate through a random subset of unchosen.
            
            # Let's try 10 random candidates per step
            random.shuffle(unchosen)
            candidates = unchosen[:20]
            
            added = False
            for c in candidates:
                if not check_conflict(c, chosen_indices):
                    # Add it
                    chosen_indices.append(c)
                    chosen_set.add(c)
                    improved = True
                    added = True
                    break # Restart improvement loop
                    
            if added:
                continue
            
            # If no set could be added, perform a Perturbation
            # Remove a small number of random sets from the family
            # This allows us to escape local optima where no single addition works
            # but a different configuration might allow a larger family
            
            # How many to remove?
            # Maybe 1-3 sets.
            num_remove = random.randint(1, 3)
            if len(chosen_indices) > num_remove:
                # Pick random sets to remove
                to_remove = random.sample(chosen_indices, num_remove)
                for r in to_remove:
                    chosen_indices.remove(r)
                    chosen_set.remove(r)
                # We don't set improved = True here because size decreased,
                # but we want to trigger the "add" phase again to refill
                # However, the loop condition `while improved` will stop.
                # We should structure the loop to allow "refill" after perturbation.
                
                # Instead of breaking, we just continue the outer loop?
                # No, we are in the inner loop.
                # Let's force the inner loop to continue by setting a flag
                # or just breaking and letting the outer loop restart?
                # Breaking and restarting the whole Greedy+LocalSearch is safer.
                break
        
        # If we broke due to perturbation, the family is smaller.
        # We can try to refill it greedily again in the next iteration of the outer loop?
        # Or just continue the outer loop which restarts from scratch?
        # Restarting from scratch (random shuffle) is a strong restart.
        # Perturbation is a soft restart.
        
        # For simplicity and robustness, let's just do Random Restarts with Greedy.
        # The "Perturbation" idea is good, but implementing it correctly within the time limit
        # without complex state management is risky.
        
        # However, pure random restarts might be slow to find the global max.
        # Let's add a "Greedy Refill" after perturbation.
        
        # Actually, the code above breaks out of the inner loop.
        # The outer loop then starts a NEW greedy phase from scratch.
        # This is effectively Random Restarts.
        
        # To make it better, we should try to **improve** the current best found so far.
        # Let's modify the strategy:
        # 1. Keep `best_family_indices`.
        # 2. In each iteration, start from a copy of `best_family_indices`?
        #    No, that might be too constrained.
        # 3. Start from scratch, but use a **Smarter Greedy**:
        #    Instead of random order, pick the set that minimizes conflicts?
        #    Or, pick the set that has the "highest potential"?
        
        # Let's try a **Greedy with Backtracking** (limited depth)? No, too slow.
        
        # Let's stick to **Iterated Local Search** but with a better perturbation:
        # Instead of restarting from scratch, we take the `best_family_indices` found so far,
        # perturb it, and try to improve it.
        
    # The above logic is a bit mixed. Let's refine.
    
    # Revised Strategy:
    # 1. Find an initial family via Random Greedy.
    # 2. Local Search: Try to add sets. If stuck, Perturb (remove k sets) and Refill.
    # 3. Keep the best family found.
    
    best_family = []
    
    for restart in range(100): # Max 100 restarts or until time
        if time.time() > deadline:
            break
            
        # 1. Initialize
        # Start with a random subset? Or empty?
        # Let's start with empty and Greedy Add.
        chosen_indices = []
        chosen_set = set()
        
        # Random Greedy
        indices = list(range(N))
        random.shuffle(indices)
        for idx in indices:
            if not check_conflict(idx, chosen_indices):
                chosen_indices.append(idx)
                chosen_set.add(idx)
                
        # 2. Local Search (Add/Refill)
        improved = True
        stall_count = 0
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add
            unchosen = [i for i in range(N) if i not in chosen_set]
            if unchosen:
                random.shuffle(unchosen)
                # Try a batch of candidates
                for c in unchosen[:50]:
                    if not check_conflict(c, chosen_indices):
                        chosen_indices.append(c)
                        chosen_set.add(c)
                        improved = True
                        break
                        
            if improved:
                stall_count = 0
            else:
                stall_count += 1
                
            # If we couldn't add anything, perturb
            if stall_count > 5:
                # Remove 1-5 random sets
                k = random.randint(1, 5)
                if len(chosen_indices) > k:
                    to_remove = random.sample(chosen_indices, k)
                    for r in to_remove:
                        chosen_indices.remove(r)
                        chosen_set.remove(r)
                    # After removing, we should try to refill in the next loop iteration
                    # Set improved to False to continue, but we need to allow the loop to run
                    # The loop condition is `while improved`.
                    # So we need to force it to continue.
                    # Let's change the loop structure.
                    pass
        
        # Update best
        if len(chosen_indices) > len(best_family):
            best_family = chosen_indices[:]

    return [sorted(combos[i]) for i in best_family]

# The previous code structure was slightly flawed in the loop control.
# Let's write a cleaner version.

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append((mask, combo))
        
    N = len(all_sets)
    masks = [m for m, _ in all_sets]
    combos = [c for _, c in all_sets]
    
    I = [[0] * N for _ in range(N)]
    for i in range(N):
        mi = masks[i]
        for j in range(i, N):
            inter = bin(mi & masks[j]).count('1')
            I[i][j] = inter
            I[j][i] = inter

    def check_conflict(c, chosen_indices):
        groups = {}
        for idx in chosen_indices:
            s = I[idx][c]
            if s not in groups:
                groups[s] = []
            groups[s].append(idx)
        
        for s, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                a = group[i]
                for j in range(i + 1, len(group)):
                    b = group[j]
                    if I[a][b] == s:
                        return True
        return False

    best_family = []
    random.seed(42)
    
    # Number of main iterations
    # Each iteration does a Greedy + Local Search
    while time.time() < deadline:
        chosen_indices = []
        chosen_set = set()
        
        # Greedy Init
        indices = list(range(N))
        random.shuffle(indices)
        for idx in indices:
            if not check_conflict(idx, chosen_indices):
                chosen_indices.append(idx)
                chosen_set.add(idx)
                
        # Local Search
        # We will try to add sets. If we can't, we perturb.
        max_stall = 10
        stall = 0
        
        while time.time() < deadline:
            # Try to add
            unchosen = [i for i in range(N) if i not in chosen_set]
            if not unchosen:
                break
                
            random.shuffle(unchosen)
            added = False
            # Try up to 20 random candidates
            for c in unchosen[:20]:
                if not check_conflict(c, chosen_indices):
                    chosen_indices.append(c)
                    chosen_set.add(c)
                    added = True
                    stall = 0
                    break
                    
            if not added:
                stall += 1
                if stall > max_stall:
                    # Perturb: remove 1-3 sets
                    k = random.randint(1, 3)
                    if len(chosen_indices) > k:
                        to_remove = random.sample(chosen_indices, k)
                        for r in to_remove:
                            chosen_indices.remove(r)
                            chosen_set.remove(r)
                        stall = 0
                    else:
                        break # Family too small to perturb meaningfully, or empty
                        
        if len(chosen_indices) > len(best_family):
            best_family = chosen_indices[:]

    return [sorted(combos[i]) for i in best_family]
