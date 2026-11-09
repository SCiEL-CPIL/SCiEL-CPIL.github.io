import time
import random
import numpy as np

def construct():
    N = 51975
    # Get all divisors > 1 of N
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    # All divisors of an odd number are odd, so no need to filter for oddness explicitly, 
    # but the problem statement says "distinct odd moduli", and N is odd.
    
    # Precompute indices for each divisor to speed up coverage checks
    # indices[m] will be a list of arrays, or we can just use slicing r::m
    # Slicing is very fast in numpy.
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_score = N  # Number of uncovered residues
    best_classes = []
    
    def get_indices(m, r):
        return np.arange(r, N, m)

    def calculate_gain(uncovered, m, r):
        # Returns number of currently uncovered elements in class (m, r)
        # uncovered is a boolean array of length N
        # We want to count True values in uncovered at positions r, r+m, ...
        # This is sum(uncovered[r::m])
        return np.sum(uncovered[r::m])

    def greedy_init(divs, deadline):
        # Select a random subset of divisors to consider? 
        # Or just shuffle all divisors and pick the best one at each step.
        # With ~48 divisors, checking all at each step is cheap.
        
        uncovered = np.ones(N, dtype=bool)
        classes = [] # List of (m, r)
        used_moduli = set()
        
        while time.time() < deadline:
            best_gain = 0
            best_m = -1
            best_r = -1
            
            # Try all unused moduli
            for m in divisors:
                if m in used_moduli:
                    continue
                
                # Find best residue for this modulus
                # We can optimize this by checking residues in a random order or 
                # just iterating all. m is at most 51975, so at most 51975 checks.
                # 48 moduli * 51975 residues * sum ops... might be slow if done naively.
                # But sum(uncovered[r::m]) is fast.
                
                # Optimization: If m is large, the number of elements is small.
                # If m is small, the number of elements is large.
                
                # Let's just iterate all residues for now, it's O(N) per modulus.
                # Total work per step: sum(m for m in unused) ~ N * num_moduli? No.
                # Sum of (N/m) for all m is N * sum(1/m). This is manageable.
                
                for r in range(m):
                    gain = np.sum(uncovered[r::m])
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
            
            if best_gain <= 0:
                break
            
            # Apply best choice
            used_moduli.add(best_m)
            classes.append((best_m, best_r))
            uncovered[best_r::best_m] = False
            
        return classes, np.sum(uncovered)

    def local_search(classes, score, deadline):
        # classes: list of (m, r)
        # score: number of uncovered residues
        # We maintain a coverage count array to efficiently update
        # cov[x] = number of classes covering x
        
        cov = np.zeros(N, dtype=np.int8)
        for m, r in classes:
            cov[r::m] += 1
        
        # Uncovered mask
        uncovered = (cov == 0)
        
        # Map modulus to index in classes list for quick lookup
        mod_to_idx = {}
        for i, (m, r) in enumerate(classes):
            mod_to_idx[m] = i
            
        used_moduli = set(mod_to_idx.keys())
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try 1-swap: change residue of an existing modulus
            # 2. Try 1-add: add a new modulus
            # 3. Try 1-remove: remove an existing modulus
            
            best_delta = 0 # We want to minimize uncovered, so we look for negative delta (improvement)
            best_move_type = None
            best_params = None
            
            # --- 1-Swap ---
            for i, (m, r) in enumerate(classes):
                # Calculate current contribution
                # If we change r to r', the change in coverage is:
                # -1 for elements in old class not in new class
                # +1 for elements in new class not in old class
                # Net change in uncovered count:
                # Elements becoming uncovered: cov[x] becomes 0 (was 1) -> x in old, not in new, cov[x]==1
                # Elements becoming covered: cov[x] becomes 1 (was 0) -> x in new, not in old, cov[x]==0
                
                # Let's just calculate the new score for a few candidate residues?
                # Or iterate all?
                # Iterating all residues for all moduli might be too slow.
                # Let's sample residues or use a heuristic.
                # Actually, N is small. Let's try all residues for the moduli with largest m first?
                # Or just all.
                
                for r_new in range(m):
                    if r_new == r:
                        continue
                    
                    # Calculate delta
                    # old_mask = (r::m)
                    # new_mask = (r_new::m)
                    
                    # Elements in old but not new:
                    # We can compute this by checking cov.
                    # This is expensive to do for every r_new.
                    
                    # Alternative:
                    # Precompute for each modulus m, the effect of changing residue?
                    # Too complex.
                    
                    # Let's stick to a simpler heuristic:
                    # Only try a few random residues or the residue that covers the most currently uncovered.
                    # But that's what greedy did.
                    
                    # Let's try all residues for now, but break early if time is tight.
                    if time.time() > deadline:
                        return classes, score
                    
                    # Efficient delta calculation:
                    # gain_new = np.sum(uncovered[r_new::m])  # These will become covered
                    # loss_old = 0
                    # for x in range(r, N, m):
                    #     if x not in new class and cov[x] == 1:
                    #         loss_old += 1
                    # 
                    # "x not in new class" is equivalent to (x - r_new) % m != 0
                    # This is still O(N/m).
                    
                    # Let's use the mask approach but vectorized.
                    # This might be slow in Python loop over r_new.
                    # Let's just do a limited search: try top K residues by gain.
                    
                    pass # Placeholder for now, will implement below
                    
            # --- 1-Add ---
            # Try adding a new modulus
            for m in divisors:
                if m in used_moduli:
                    continue
                # Find best residue for m
                # gain = np.sum(uncovered[r::m])
                # We want to maximize gain.
                # We can find the max gain for this m.
                # max_gain = max(np.sum(uncovered[r::m]) for r in range(m))
                # This is O(N) per m.
                
                # Optimization: We can compute this faster?
                # For a fixed m, the values are periodic.
                
                best_r_add = 0
                max_gain_add = 0
                for r in range(m):
                    g = np.sum(uncovered[r::m])
                    if g > max_gain_add:
                        max_gain_add = g
                        best_r_add = r
                
                if max_gain_add > 0:
                    # Delta is -max_gain_add (uncovered decreases)
                    if -max_gain_add < best_delta:
                        best_delta = -max_gain_add
                        best_move_type = 'add'
                        best_params = (m, best_r_add)
                        
            # --- 1-Remove ---
            # Try removing an existing modulus
            # If we remove (m, r), uncovered increases by number of elements x where cov[x]==1 and x in (m,r)
            for i, (m, r) in enumerate(classes):
                # Count elements covered ONLY by this class
                # These are x in range(r, N, m) where cov[x] == 1
                # loss = np.sum((cov == 1) & (np.arange(N) % m == r)) ? No, cov is an array.
                # We can just sum cov[r::m] == 1
                loss = np.sum(cov[r::m] == 1)
                # Delta is +loss (uncovered increases)
                # We only care if this is an improvement (negative delta), so removal is never an improvement
                # unless it allows a better swap? No, pure removal increases uncovered.
                # So we skip removal in pure hill climbing for minimization of uncovered.
                # Wait, removing a class might allow adding another class that covers more?
                # That's a 2-move. We are doing 1-move.
                # So 1-remove is never beneficial for minimizing uncovered.
                
            # Apply best move if found
            if best_move_type is not None:
                if best_move_type == 'add':
                    m, r = best_params
                    classes.append((m, r))
                    used_moduli.add(m)
                    cov[r::m] += 1
                    score += best_delta
                    improved = True
                elif best_move_type == 'swap':
                    # Implement swap logic here if we decided to do it
                    pass
                
        return classes, score

    # The local search above was a sketch. Let's implement a more robust one.
    # Given the time constraints, a simpler but effective local search is:
    # 1. Greedy initialization.
    # 2. Iterate:
    #    a. For each modulus in classes, try changing its residue to the one that maximizes coverage of currently uncovered.
    #    b. For each unused modulus, try adding it with the residue that maximizes coverage of currently uncovered.
    #    c. Accept any move that improves the score.
    #    d. Repeat until no improvement.
    
    def improved_local_search(classes, deadline):
        # Rebuild coverage
        cov = np.zeros(N, dtype=np.int8)
        for m, r in classes:
            cov[r::m] += 1
        
        uncovered = (cov == 0)
        score = np.sum(uncovered)
        
        used_moduli = set(m for m, r in classes)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try swapping residues of existing moduli
            for i, (m, r) in enumerate(classes):
                # Find best residue for m
                best_r = r
                best_gain = np.sum(uncovered[r::m])
                
                # Try other residues?
                # To speed up, we can just check if the current one is optimal.
                # Or check a few random ones.
                # Let's check all if m is small, sample if m is large.
                if m < 100:
                    candidates = range(m)
                else:
                    candidates = random.sample(range(m), min(100, m))
                
                for r_new in candidates:
                    if r_new == r:
                        continue
                    gain = np.sum(uncovered[r_new::m])
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r_new
                
                if best_r != r:
                    # Apply change
                    # Remove old
                    cov[r::m] -= 1
                    # Add new
                    cov[best_r::m] += 1
                    # Update classes
                    classes[i] = (m, best_r)
                    # Update uncovered and score
                    uncovered = (cov == 0)
                    new_score = np.sum(uncovered)
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        # Revert if no improvement (shouldn't happen if gain logic is correct, but safe)
                        cov[best_r::m] -= 1
                        cov[r::m] += 1
                        classes[i] = (m, r)
                        uncovered = (cov == 0)
                        score = np.sum(uncovered)
                        
            # 2. Try adding new moduli
            for m in divisors:
                if m in used_moduli:
                    continue
                
                best_r = 0
                best_gain = 0
                if m < 100:
                    candidates = range(m)
                else:
                    candidates = random.sample(range(m), min(100, m))
                
                for r in candidates:
                    gain = np.sum(uncovered[r::m])
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                        
                if best_gain > 0:
                    # Add class
                    classes.append((m, best_r))
                    used_moduli.add(m)
                    cov[best_r::m] += 1
                    uncovered = (cov == 0)
                    new_score = np.sum(uncovered)
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        # Revert
                        classes.pop()
                        used_moduli.remove(m)
                        cov[best_r::m] -= 1
                        uncovered = (cov == 0)
                        score = np.sum(uncovered)
                        
        return classes, score

    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Shuffle divisors for greedy init to get different starting points
        shuffled_divs = divisors[:]
        random.shuffle(shuffled_divs)
        
        # Greedy init
        classes, score = greedy_init(shuffled_divs, deadline - 10)
        
        # Local search
        classes, score = improved_local_search(classes, deadline - 2)
        
        if score < best_score:
            best_score = score
            best_classes = classes[:]
            
    return [[m, r] for m, r in best_classes]
