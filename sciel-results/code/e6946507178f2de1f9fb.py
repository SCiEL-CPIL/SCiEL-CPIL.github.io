import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    num_sums = target_max + 1  # 257
    
    # Precompute delta vectors for self-pairs
    delta_self = np.zeros((129, num_sums), dtype=np.int32)
    for x in range(129):
        s = 2 * x
        if target_min <= s <= target_max:
            delta_self[x, s] = 1
    
    def calculate_N_conv(A_mask):
        """Calculate N vector using convolution: N[s] = count of unordered pairs (a,b) in A with a+b=s."""
        # Use FFT-based convolution for speed
        f = np.zeros(257, dtype=np.float64)
        f[np.where(A_mask)[0]] = 1.0
        # Convolution gives ordered pairs count
        conv = np.convolve(f, f, mode='full')
        # conv[k] = sum_{i+j=k} f[i]*f[j] = number of ordered pairs (i,j) with i+j=k
        # Unordered pairs: for i != j, each pair counted twice; for i == j, counted once
        # So unordered count = (conv[k] + delta_self_count) / 2 where delta_self_count is 1 if k is even and k/2 in A
        # Actually: conv[k] = sum_{i=0}^{k} f[i]*f[k-i]
        # This counts (i,j) and (j,i) separately for i != j, and (i,i) once
        # Unordered: count = (conv[k] + self_count) / 2 where self_count = 1 if k/2 in A and k even
        N = np.zeros(num_sums, dtype=np.int32)
        for s in range(num_sums):
            ordered = int(round(conv[s]))
            if s % 2 == 0:
                half = s // 2
                self_cnt = 1 if (0 <= half <= 128 and A_mask[half]) else 0
            else:
                self_cnt = 0
            unordered = (ordered + self_cnt) // 2
            N[s] = unordered
        return N
    
    def calculate_N_fast(A_mask):
        """Fast calculation using numpy operations."""
        A_indices = np.where(A_mask)[0]
        m = len(A_indices)
        if m == 0:
            return np.zeros(num_sums, dtype=np.int32)
        
        # Use broadcasting to compute all pair sums
        # This is O(m^2) but with numpy it's fast
        i_idx, j_idx = np.triu_indices(m)
        sums = A_indices[i_idx] + A_indices[j_idx]
        N = np.zeros(num_sums, dtype=np.int32)
        valid = (sums >= 0) & (sums < num_sums)
        if np.any(valid):
            np.add.at(N, sums[valid], 1)
        return N
    
    def get_objective(N):
        target_N = N[target_min:target_max+1]
        missing = np.sum(target_N == 0)
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = int(np.max(target_N))
        sum_sq = int(np.sum(target_N**2))
        return (0, peak, sum_sq)
    
    def structural_init():
        """Initialize using structural insights."""
        A_mask = np.zeros(129, dtype=bool)
        
        base = set()
        for x in range(96, 129):
            base.add(x)
        for x in range(48, 96, 2):
            base.add(x)
        for x in range(0, 48, 8):
            base.add(x)
        
        for x in base:
            A_mask[x] = True
        
        N = calculate_N_fast(A_mask)
        
        for _ in range(100):
            missing_sums = [s for s in range(target_min, target_max+1) if N[s] == 0]
            if not missing_sums:
                break
            
            A_indices = np.where(A_mask)[0]
            candidates = [x for x in range(129) if not A_mask[x]]
            
            best_x = None
            best_cover = 0
            
            for x in candidates:
                covered = 0
                for s in missing_sums:
                    y = s - x
                    if 0 <= y <= 128 and A_mask[y]:
                        covered += 1
                    if 2*x == s:
                        covered += 1
                if covered > best_cover:
                    best_cover = covered
                    best_x = x
            
            if best_x is None or best_cover == 0:
                break
            
            A_mask[best_x] = True
            N = calculate_N_fast(A_mask)
        
        return A_mask
    
    def local_search(A_mask, time_limit):
        """Fast local search with incremental updates."""
        A_mask = A_mask.copy()
        N = calculate_N_fast(A_mask)
        
        A_indices = set(np.where(A_mask)[0].tolist())
        not_A_indices = [x for x in range(129) if not A_mask[x]]
        
        current_obj = get_objective(N)
        best_obj = current_obj
        best_A = A_mask.copy()
        best_N = N.copy()
        
        no_improve = 0
        max_no_improve = 500
        step = 0
        
        while time.time() < time_limit:
            step += 1
            if not A_indices or not not_A_indices:
                break
            
            # Random swap
            a = random.choice(list(A_indices))
            b = random.choice(not_A_indices)
            
            # Remove a
            A_indices.discard(a)
            A_mask[a] = False
            for y in A_indices:
                s = a + y
                if 0 <= s < num_sums:
                    N[s] -= 1
            s_aa = 2 * a
            if 0 <= s_aa < num_sums:
                N[s_aa] -= 1
            
            # Add b
            A_indices.add(b)
            A_mask[b] = True
            for y in A_indices:
                if y != b:
                    s = b + y
                    if 0 <= s < num_sums:
                        N[s] += 1
            s_bb = 2 * b
            if 0 <= s_bb < num_sums:
                N[s_bb] += 1
            
            new_obj = get_objective(N)
            
            # Accept if better or equal
            accept = False
            if new_obj < best_obj:
                accept = True
            elif new_obj[0] == 0 and best_obj[0] == 0:
                if new_obj[1] < best_obj[1]:
                    accept = True
                elif new_obj[1] == best_obj[1] and new_obj[2] <= best_obj[2]:
                    accept = True
            
            if accept:
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = A_mask.copy()
                    best_N = N.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                A_mask[b] = False
                A_mask[a] = True
                A_indices.discard(b)
                A_indices.add(a)
                
                for y in A_indices:
                    if y != a:
                        s = b + y
                        if 0 <= s < num_sums:
                            N[s] -= 1
                s_bb = 2 * b
                if 0 <= s_bb < num_sums:
                    N[s_bb] -= 1
                
                for y in A_indices:
                    s = a + y
                    if 0 <= s < num_sums:
                        N[s] += 1
                s_aa = 2 * a
                if 0 <= s_aa < num_sums:
                    N[s_aa] += 1
                
                no_improve += 1
            
            if no_improve > max_no_improve:
                break
        
        return best_A, best_N, best_obj
    
    best_peak = float('inf')
    best_A_mask = None
    best_sum_sq = float('inf')
    
    # Main loop: multiple restarts with local search
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        A_mask = structural_init()
        
        if restart > 0:
            A_indices = list(np.where(A_mask)[0])
            for _ in range(random.randint(3, 8)):
                if A_indices:
                    a = random.choice(A_indices)
                    candidates = [x for x in range(129) if not A_mask[x]]
                    if candidates:
                        b = random.choice(candidates)
                        A_mask[a] = False
                        A_mask[b] = True
        
        N = calculate_N_fast(A_mask)
        obj = get_objective(N)
        
        if obj[0] == 0:
            peak = obj[1]
            sum_sq = obj[2]
            if peak < best_peak or (peak == best_peak and sum_sq < best_sum_sq):
                best_peak = peak
                best_sum_sq = sum_sq
                best_A_mask = A_mask.copy()
        
        time_left = deadline - time.time() - 3
        if time_left > 1:
            A_opt, N_opt, obj_opt = local_search(A_mask, time_left)
            
            if obj_opt[0] == 0:
                if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                    best_peak = obj_opt[1]
                    best_sum_sq = obj_opt[2]
                    best_A_mask = A_opt.copy()
    
    if best_A_mask is None:
        A_mask = np.zeros(129, dtype=bool)
        for x in range(64, 129):
            A_mask[x] = True
        N = calculate_N_fast(A_mask)
        obj = get_objective(N)
        if obj[0] == 0:
            best_peak = obj[1]
            best_sum_sq = obj[2]
            best_A_mask = A_mask.copy()
        else:
            return list(range(129))
        
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list
