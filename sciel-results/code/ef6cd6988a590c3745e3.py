import numpy as np
import time
import random

def construct():
    N = 512
    deadline = time.time() + 85.0
    
    # Precompute 4-APs
    # An AP is (a, a+d, a+2d, a+3d)
    # We store them as indices into the bitset (0-indexed for numpy, so subtract 1)
    # a in 1..N, d >= 1, a+3d <= N
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            # Indices in 0-based numpy array
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
    
    aps = np.array(aps, dtype=np.int32)
    
    def count_violations(S):
        """
        S: numpy array of bools, shape (N,)
        Returns number of 4-APs fully contained in S.
        """
        if len(aps) == 0:
            return 0
        # S[aps] gives a 2D array of shape (num_aps, 4)
        # We need to check if all 4 are True
        # Sum along axis 1 == 4
        return np.sum(np.sum(S[aps], axis=1) == 4)
    
    # Initial state: Greedy construction to get a good starting point
    # This is faster than starting from scratch in SA
    S = np.zeros(N, dtype=bool)
    
    # Simple greedy to initialize
    # We want to maximize size, so we add elements if they don't create violations
    # However, standard greedy might get stuck. 
    # Let's try a few random greedy passes to find a good start.
    best_start = None
    best_start_size = 0
    
    for _ in range(20):
        if time.time() > deadline - 5.0: break
        S_trial = np.zeros(N, dtype=bool)
        candidates = list(range(N))
        random.shuffle(candidates)
        for idx in candidates:
            S_trial[idx] = True
            if count_violations(S_trial) > 0:
                S_trial[idx] = False
        size = np.sum(S_trial)
        if size > best_start_size:
            best_start_size = size
            best_start = S_trial.copy()
            
    if best_start is None:
        best_start = np.zeros(N, dtype=bool)
        # Ensure non-empty
        best_start[0] = True
        best_start_size = 1
        
    S = best_start
    current_size = best_start_size
    current_violations = count_violations(S)
    
    # Simulated Annealing
    # Objective: Maximize size, Minimize violations.
    # We can use a penalty method: Score = size - M * violations
    # But we want 0 violations ideally. 
    # Let's use a two-phase approach or a penalty that increases.
    # Actually, standard SA for Max-Independent-Set-like problems:
    # Move: Flip a bit.
    # Delta Energy: 
    #   If adding x: 
    #     size_delta = +1
    #     violation_delta = (number of APs containing x that become complete)
    #   If removing x:
    #     size_delta = -1
    #     violation_delta = -(number of APs containing x that are currently complete)
    
    # To speed up violation_delta calculation, we can precompute for each index
    # the list of APs it belongs to.
    # But iterating through APs in Python is slow.
    # Let's stick to the vectorized count_violations for the whole set,
    # but that's O(N_APS) per move. N_APS is ~ 50k. 50k * 10000 moves = 500M ops.
    # NumPy can handle this, but Python overhead of calling count_violations might be high.
    
    # Alternative: Precompute for each index i, the mask of APs it touches.
    # This is too complex for quick implementation.
    
    # Let's optimize the SA loop.
    # We will use a fixed number of iterations or time-based.
    
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # We need to calculate the change in violations efficiently.
    # Let's precompute for each index, the list of AP indices it is part of.
    # ap_indices_for_idx[i] = list of row indices in `aps` where i is present.
    
    ap_indices_for_idx = [[] for _ in range(N)]
    for row_idx in range(len(aps)):
        for col in range(4):
            idx = aps[row_idx, col]
            ap_indices_for_idx[idx].append(row_idx)
    
    # Convert to numpy arrays for faster access? Or just lists.
    # Lists are fine if we use them carefully.
    
    best_S = S.copy()
    best_size = current_size
    
    start_time = time.time()
    
    while time.time() < deadline - 5.0:
        # Pick a random index to flip
        idx = random.randint(0, N - 1)
        
        if S[idx]:
            # Removing
            # Check how many violations are removed
            # A violation is removed if it was complete and contained idx.
            removed_violations = 0
            for ap_idx in ap_indices_for_idx[idx]:
                # Check if this AP is currently fully in S
                # We can check S[aps[ap_idx]]
                if np.sum(S[aps[ap_idx]]) == 4:
                    removed_violations += 1
            
            new_size = current_size - 1
            new_violations = current_violations - removed_violations
            
            # We want to accept if it improves size or reduces violations.
            # In SA, we accept worse moves with probability exp(-delta_E / T).
            # Let's define Energy E = -size + M * violations.
            # We want to minimize E.
            # Delta E = E_new - E_old
            # E_new = -(size-1) + M*(viol-rem) = -size + 1 + M*viol - M*rem
            # E_old = -size + M*viol
            # Delta E = 1 - M*rem
            
            # If rem > 0, Delta E is negative (improvement) if M is large enough.
            # If rem == 0, Delta E = 1 (worse size, same violations).
            
            # We prioritize size. Let M be very large?
            # Actually, we want 0 violations. 
            # If we have violations, we should remove them.
            # If we have 0 violations, we should add size.
            
            # Let's use a simpler heuristic:
            # If new_violations < current_violations: Accept.
            # If new_violations == current_violations and new_size > current_size: Accept.
            # Else: Accept with probability exp(-delta_score / T)
            
            # Score = size * 1000 - violations
            # Delta Score = (new_size - current_size)*1000 - (new_violations - current_violations)
            # Delta Score = -1000 - (-removed_violations) = -1000 + removed_violations
            
            # If removed_violations > 0, it's likely negative (good) if M is large?
            # No, removing an element decreases size.
            # If it removes a violation, it's good.
            
            delta_score = (new_size - current_size) * 1000 - (new_violations - current_violations)
            # delta_score = -1000 + removed_violations
            
            if delta_score > 0:
                accept = True
            else:
                # Accept if random < exp(delta_score / T)
                # delta_score is negative
                if random.random() < np.exp(delta_score / T):
                    accept = True
                else:
                    accept = False
            
            if accept:
                S[idx] = False
                current_size = new_size
                current_violations = new_violations
        else:
            # Adding
            # Check how many violations are created
            # A violation is created if the other 3 elements are already in S.
            created_violations = 0
            for ap_idx in ap_indices_for_idx[idx]:
                # Check if the other 3 are in S
                # Get the 4 indices
                ap = aps[ap_idx]
                # Count how many are in S, excluding idx
                # S[ap] has 4 elements. One is idx.
                # If sum(S[ap]) == 3, then adding idx makes it 4.
                if np.sum(S[ap]) == 3:
                    created_violations += 1
            
            new_size = current_size + 1
            new_violations = current_violations + created_violations
            
            delta_score = (new_size - current_size) * 1000 - (new_violations - current_violations)
            # delta_score = 1000 - created_violations
            
            if delta_score > 0:
                accept = True
            else:
                if random.random() < np.exp(delta_score / T):
                    accept = True
                else:
                    accept = False
            
            if accept:
                S[idx] = True
                current_size = new_size
                current_violations = new_violations
        
        if current_violations == 0 and current_size > best_size:
            best_size = current_size
            best_S = S.copy()
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final check: ensure best_S has 0 violations
    # It might not if we got stuck in a local optimum with violations.
    # If best_S has violations, we should try to fix them.
    # But the requirement is "containing no nonconstant 4-AP".
    # So we MUST return a set with 0 violations.
    
    # If best_S has violations, we need to remove elements to fix them.
    # Greedy removal:
    S_clean = best_S.copy()
    v = count_violations(S_clean)
    while v > 0:
        # Find an AP that is violated
        # Get all violated APs
        violated_aps = aps[np.sum(S_clean[aps], axis=1) == 4]
        if len(violated_aps) == 0:
            break
        # Pick one
        ap = violated_aps[0]
        # Remove one element from it. Prefer the one that breaks the most other APs?
        # For simplicity, remove the first one, or the one with highest degree in violated subgraph?
        # Let's just remove the first element of the first violated AP.
        # This is O(N) in worst case per removal, but removals should be few.
        idx_to_remove = ap[0]
        S_clean[idx_to_remove] = False
        v = count_violations(S_clean)
        
    # Verify
    if count_violations(S_clean) > 0:
        # Fallback: Return a known small set or empty? No, must be non-empty.
        # If we still have violations, something is wrong with the logic.
        # Let's force remove until clean.
        # This loop should terminate.
        pass
        
    result = [i + 1 for i in range(N) if S_clean[i]]
    if not result:
        result = [1]
        
    return result

