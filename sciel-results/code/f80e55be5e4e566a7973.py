import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    max_target = 250
    
    # Pre-generate all 4-subsets as bitmasks
    all_sets = []
    all_set_list = [] # list of lists for output
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        all_set_list.append(combo)
        
    total = len(all_sets)
    
    # Helper to check if adding mask m to family F (list of masks) is valid
    # F is a list of masks.
    # We need to check if there exist i, j in F such that F[i] & F[j] == F[i] & m == F[j] & m
    def is_valid_add(m, F):
        # Optimization: Group F by intersection? 
        # For each pair (i, j), compute inter = F[i] & F[j].
        # If m contains inter, and (m & (F[i] ^ inter)) == 0 and (m & (F[j] ^ inter)) == 0, then conflict.
        # Wait, A \cap B = I. A' = A \ I. B' = B \ I.
        # Condition: I \subseteq m AND (m \ I) \cap A' = \emptyset AND (m \ I) \cap B' = \emptyset.
        # m \ I = m & ~I.
        # A' = F[i] & ~I.
        # So: (m & ~I) & (F[i] & ~I) == 0 => (m & F[i] & ~I) == 0.
        # But we already know I = F[i] & F[j]. So F[i] & ~I is the part of F[i] not in F[j].
        # The condition is simply:
        # 1. (m & I) == I  (i.e., I is subset of m)
        # 2. (m & (F[i] ^ I)) == 0
        # 3. (m & (F[j] ^ I)) == 0
        
        # Actually, simpler:
        # Conflict if (F[i] & F[j]) == (F[i] & m) == (F[j] & m).
        
        n_f = len(F)
        for i in range(n_f):
            fi = F[i]
            for j in range(i + 1, n_f):
                fj = F[j]
                inter_ij = fi & fj
                inter_im = fi & m
                inter_jm = fj & m
                if inter_ij == inter_im == inter_jm:
                    return False
        return True

    # Initial construction: Lexicographic Greedy (same as seed)
    best_family = []
    for idx in range(total):
        m = all_sets[idx]
        if is_valid_add(m, best_family):
            best_family.append(m)
            
    best_size = len(best_family)
    
    # Local Search
    rng = random.Random(42)
    
    # To speed up is_valid_add, we can maintain a data structure, but for n=200, O(N^2) is 40k checks.
    # 40k simple bit ops is fast in Python? Maybe ~0.01s.
    # If we do 1000 swaps, that's 10s. We have 55s.
    
    current_family = best_family[:]
    
    # Precompute "conflict pairs" for current family?
    # Let's just run a simple hill climbing with random restarts or swaps.
    
    def try_improve(family, time_limit):
        F = family[:]
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add a random set
            # Or try to swap
            
            # Strategy 1: Try adding all sets not in F
            # This is slow. 
            # Strategy 2: Random Swaps
            for _ in range(1000): # iterations per pass
                if time.time() > time_limit:
                    break
                
                # Pick a random index in F
                if not F:
                    break
                idx_remove = rng.randint(0, len(F) - 1)
                m_remove = F[idx_remove]
                
                # Pick a random candidate to add
                # It's better to pick candidates that are "close" or just random
                m_add = all_sets[rng.randint(0, total - 1)]
                
                if m_add == m_remove:
                    continue
                
                # Check if m_add is already in F
                if m_add in F:
                    continue
                
                # Try the swap: (F \ {m_remove}) U {m_add}
                new_F = F[:]
                new_F[idx_remove] = m_add
                
                if is_valid_add(m_add, [x for x in new_F if x != m_add]):
                    # Valid swap
                    F = new_F
                    improved = True
                    if len(F) > best_size:
                        best_size = len(F)
                        best_family = F[:]
                else:
                    # Try just adding m_add (if size allows? No, we want to maintain or increase size)
                    # Actually, if we just add, size increases by 1.
                    # Let's try adding first, then swapping.
                    pass
            
            # Try to add any valid set
            # This is O(N * Total) which is 200 * 1820 = 364k checks.
            # Each check is O(N^2) = 40k. Total 14 billion ops. Too slow.
            
            # So we must rely on random sampling for additions/swaps.
            
            # Let's try a "repair" phase: 
            # If we can't improve, try removing a set that blocks many others?
            # Hard to compute.
            
            # Let's just do random restarts of the local search if stuck.
            
        return F

    # Run local search
    # Since is_valid_add is O(N^2), and N~200, it's ~40k ops.
    # Python does ~10M simple ops/sec. So 40k is 0.004s.
    # We can do ~1000 checks per second.
    # In 50s, we can do 50,000 checks.
    
    # Let's structure the loop to maximize checks.
    
    current_F = best_family[:]
    last_improve = time.time()
    
    while time.time() < deadline:
        # Random restart if stuck for too long?
        # Or just continue.
        
        # Pick a random move
        # 1. Add random set
        # 2. Swap random set
        
        m_add = all_sets[rng.randint(0, total - 1)]
        
        # Check if m_add is in current_F
        if m_add in current_F:
            continue
            
        # Try to add m_add
        if is_valid_add(m_add, current_F):
            current_F.append(m_add)
            last_improve = time.time()
            if len(current_F) > best_size:
                best_size = len(current_F)
                best_family = current_F[:]
                if best_size >= max_target:
                    break
        else:
            # Try to swap
            # Pick a random element from current_F to remove
            if current_F:
                idx_rm = rng.randint(0, len(current_F) - 1)
                m_rm = current_F[idx_rm]
                
                # Check if (current_F \ {m_rm}) U {m_add} is valid
                # This requires checking m_add against all pairs in current_F \ {m_rm}
                # And checking that removing m_rm doesn't break anything (it doesn't, as we are removing a member)
                # Wait, removing a member cannot create a sunflower. It only removes potential sunflowers.
                # So we only need to check if m_add conflicts with the remaining set.
                
                # Optimization: 
                # We need to check if m_add forms a sunflower with any pair in current_F \ {m_rm}.
                
                # We can reuse the is_valid_add logic but on the modified list.
                # To avoid list copy, we can pass a flag or just copy (list of 200 ints is cheap).
                
                temp_F = current_F[:]
                temp_F.pop(idx_rm)
                
                if is_valid_add(m_add, temp_F):
                    current_F = temp_F
                    current_F.append(m_add)
                    last_improve = time.time()
                    if len(current_F) > best_size:
                        best_size = len(current_F)
                        best_family = current_F[:]
                        if best_size >= max_target:
                            break
                else:
                    # Try swapping with a different removal?
                    # Or just accept failure for this step.
                    pass
                    
    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
