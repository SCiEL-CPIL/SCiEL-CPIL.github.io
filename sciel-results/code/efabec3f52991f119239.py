import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1 # 129
    
    # Precompute for each x in [0, 128], the indices in the target range [128, 256]
    # that are affected when x is added to or removed from A.
    # If x is in A, it contributes to sums x+z for all z in A.
    # When we add/remove x, the change in counts depends on the current A.
    # However, the *potential* sums x can form are fixed.
    # Let's precompute the list of target sums s such that s-x is in [0, 128].
    # s in [128, 256] and s-x in [0, 128] => x in [s-128, s] and x in [0, 128].
    
    # For fast delta updates, we will maintain:
    # A: boolean array of length 129
    # counts: array of length 257 (or just target range 129)
    
    # To speed up delta calculation for move "add x":
    # For each z in A, if x+z is in target range, counts[x+z] += 1.
    # Also counts[2x] += 1 (self sum, if 2x in range).
    # This is O(|A|) per move. |A| ~ 40-50. 50k steps * 50 = 2.5M ops. Very fast.
    
    # We will use Tabu Search to avoid cycling.
    # Tabu list: recent moves (x, type) where type is 'add' or 'remove'.
    # If we add x, we cannot remove x for T steps.
    
    best_A = None
    best_score = float('inf')
    
    # Strategy: Multiple restarts with different initializations.
    # Initialization 1: Random set of size 45.
    # Initialization 2: Structured set (e.g., arithmetic progression + noise).
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(100 + r)
        
        # Initialize A
        # Let's try a slightly larger set to ensure coverage, then prune.
        # Or start with a set known to cover well.
        # A simple set: {0, 1, ..., k} U {128-k, ..., 128} might cover well?
        # No, sums of small numbers are small. Sums of large numbers are large.
        # We need sums in [128, 256].
        # If A = [0, 128], sums are 0..256.
        # To cover 128..256, we need enough "middle" numbers.
        
        init_size = 48
        A_set = set()
        
        # Heuristic init: Pick numbers that spread out sums.
        # e.g. multiples of 3, plus some randoms.
        # Or just random.
        
        # Let's force 0 and 128 to be in A?
        # 0+x = x. If x in [128, 256], but x<=128, so 0 only covers 128..128? No.
        # 0 + 128 = 128.
        # 128 + 128 = 256.
        # So 0 and 128 cover the endpoints.
        
        A_set.add(0)
        A_set.add(128)
        
        while len(A_set) < init_size:
            # Bias towards middle?
            # random.randint(0, 128)
            # Let's just use uniform random for now.
            A_set.add(random.randint(0, 128))
            
        A = np.zeros(129, dtype=bool)
        for x in A_set:
            A[x] = True
            
        # Calculate initial counts
        A_conv = np.convolve(A, A)
        # A_conv[s] is ordered count.
        # Unordered count c[s]:
        # If s is even and A[s/2] is True, c[s] = (A_conv[s] + 1) // 2
        # Else c[s] = A_conv[s] // 2
        
        counts = np.zeros(257, dtype=np.int32)
        for s in range(128, 257):
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and A[half]:
                    counts[s] = (A_conv[s] + 1) // 2
                else:
                    counts[s] = A_conv[s] // 2
            else:
                counts[s] = A_conv[s] // 2
        
        def get_score(counts_arr):
            c_target = counts_arr[128:257]
            violations = np.sum(c_target == 0)
            peak = np.max(c_target)
            # We want to minimize peak, but violations are fatal.
            # Score = violations * 1000 + peak
            return violations * 1000 + peak
        
        current_score = get_score(counts)
        
        # Tabu Search
        tabu_list = {} # x -> iteration when it becomes non-tabu
        tabu_tenure = 20 # Keep last 20 moves tabu
        aspiration_threshold = 0 # If new score < best_score found so far, ignore tabu
        
        best_local_A = A.copy()
        best_local_score = current_score
        best_local_counts = counts.copy()
        
        no_improve_count = 0
        max_no_improve = 2000
        
        # Precompute list of active elements for fast iteration
        A_list = [x for x in range(129) if A[x]]
        
        it = 0
        while time.time() < deadline - 2.0:
            if no_improve_count > max_no_improve:
                break
                
            it += 1
            
            # Generate candidate moves
            # 1. Remove an element x from A
            # 2. Add an element y to A
            
            # To keep set size roughly constant, we might prefer swaps?
            # But add/remove allows size change.
            # If size grows too much, peak increases.
            # If size shrinks too much, violations increase.
            # Let's try random add or random remove.
            
            move_type = random.choice(['add', 'remove'])
            
            if move_type == 'remove':
                if len(A_list) <= 10: # Don't shrink too much
                    move_type = 'add'
            
            if move_type == 'add':
                # Pick y not in A
                y = random.randint(0, 128)
                while A[y]:
                    y = random.randint(0, 128)
                
                # Check tabu
                if y in tabu_list and tabu_list[y] > it:
                    continue
                
                # Calculate delta
                # Add y:
                # For each z in A_list:
                #   s = y + z
                #   if 128 <= s <= 256: counts[s] += 1
                # Self:
                #   s = 2*y
                #   if 128 <= s <= 256: counts[s] += 1
                
                delta = np.zeros(257, dtype=np.int32)
                for z in A_list:
                    s = y + z
                    if 128 <= s <= 256:
                        delta[s] += 1
                s_self = 2 * y
                if 128 <= s_self <= 256:
                    delta[s_self] += 1
                    
                new_counts = counts + delta
                new_score = get_score(new_counts)
                
                # Aspiration: if new_score < best_local_score, ignore tabu (already checked tabu above, but if it was tabu, we skipped. Let's re-check logic)
                # Actually, if it's tabu, we should skip UNLESS it improves best.
                # Let's re-evaluate tabu check after score calc?
                # No, standard TS: check tabu, if tabu and not aspiration, reject.
                
                if y in tabu_list and tabu_list[y] > it and new_score >= best_local_score:
                    continue
                    
                # Acceptance: TS accepts if it improves current, or is neutral, or even worsens slightly?
                # Standard TS: Accept if it improves current best.
                # Or accept if it improves current state.
                # Let's accept if new_score < current_score OR (new_score == current_score and random < 0.1)
                # Actually, to escape local minima, we should accept worse moves sometimes?
                # TS doesn't use probability, it uses tabu.
                # So if it's not tabu, we accept it? No, that leads to random walk.
                # TS accepts the move if it improves the current solution.
                # If no improving move is available (all are worse), we take the best non-tabu move.
                
                # Let's stick to: Accept if new_score < current_score.
                # If not, reject.
                # This is steepest descent with tabu.
                
                if new_score < current_score:
                    # Accept
                    A[y] = True
                    A_list.append(y)
                    counts = new_counts
                    current_score = new_score
                    tabu_list[y] = it + tabu_tenure # y is tabu for 'add' (so we can't remove it? No, tabu is on the move.
                    # If we added y, we shouldn't remove y for T steps.
                    # So tabu_list[y] = it + T.
                    
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_A = A.copy()
                        best_local_counts = counts.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
            else: # remove
                if not A_list:
                    continue
                x = random.choice(A_list)
                
                # Check tabu
                if x in tabu_list and tabu_list[x] > it:
                    continue
                
                # Calculate delta
                # Remove x:
                # For each z in A_list:
                #   if z == x: skip (handled by self)
                #   s = x + z
                #   if 128 <= s <= 256: counts[s] -= 1
                # Self:
                #   s = 2*x
                #   if 128 <= s <= 256: counts[s] -= 1
                
                delta = np.zeros(257, dtype=np.int32)
                for z in A_list:
                    if z == x:
                        continue
                    s = x + z
                    if 128 <= s <= 256:
                        delta[s] -= 1
                s_self = 2 * x
                if 128 <= s_self <= 256:
                    delta[s_self] -= 1
                    
                new_counts = counts + delta
                new_score = get_score(new_counts)
                
                if x in tabu_list and tabu_list[x] > it and new_score >= best_local_score:
                    continue
                
                if new_score < current_score:
                    # Accept
                    A[x] = False
                    A_list.remove(x)
                    counts = new_counts
                    current_score = new_score
                    tabu_list[x] = it + tabu_tenure # x is tabu for 'remove' (so we can't add it back? No, tabu is on the move.
                    # If we removed x, we shouldn't add x for T steps.
                    
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_A = A.copy()
                        best_local_counts = counts.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        # If we just made it worse? No, we only accept if new_score < current_score.
                        # So current_score always improves or stays same?
                        # Wait, if new_score < current_score, current_score decreases.
                        # So best_local_score should also decrease if this is the best so far.
                        
        # End while
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_A = best_local_A.copy()
            
        # Early exit
        if best_score <= 4: # If we get peak 4, it's very good.
            break
            
    if best_A is None:
        best_A = np.zeros(129, dtype=bool)
        best_A[0] = True
        best_A[128] = True
        
    result = [i for i in range(129) if best_A[i]]
    return result

if __name__ == "__main__":
    A = construct()
    print(A)
