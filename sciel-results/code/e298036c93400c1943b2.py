import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    def time_left():
        return deadline - time.time()
    
    def compute_score_fft(U):
        """Compute score using FFT for O(M log M) sumset/diffset size."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Sumset: U + U
        # Convolve indicator of U with itself
        n = len(U)
        max_sum = 2 * max_u
        # Size of FFT: need at least max_sum + 1
        fft_size = 1
        while fft_size <= max_sum:
            fft_size <<= 1
        
        # Create indicator array
        f = np.zeros(fft_size, dtype=np.float64)
        for x in U:
            f[x] = 1.0
        
        # FFT convolution
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=fft_size)
        
        # Count non-zero entries (with tolerance)
        n_sum = int(np.sum(conv > 0.5))
        
        # Difference set: U - U
        # This is convolution of U with reversed U
        # U - U = {a - b : a,b in U}
        # Range: from -max_u to max_u
        # We can compute by convolving U with reversed U
        # Let g[x] = 1 if x in U, h[x] = 1 if (max_u - x) in U
        # Then (g * h)[k] = sum_x g[x] h[k-x] = sum_x g[x] g[max_u - (k-x)] = sum_x g[x] g[max_u - k + x]
        # This gives counts of pairs (x, y) with x + y = max_u - k, i.e., y - x = k - max_u
        # So difference d = k - max_u, count is conv_diff[max_u - d]... let me think more carefully.
        
        # Actually, U - U range is [-max_u, max_u]
        # We want to count distinct values in U - U
        # Use: diff_count[d] for d in [-max_u, max_u]
        # diff_count[d] = |{a in U : a - d in U}| = |U ∩ (U + d)|
        # This is autocorrelation
        
        # Autocorrelation via FFT:
        # autocorr[k] = sum_x f[x] f[x + k] for k >= 0 (with appropriate handling)
        # Using FFT: autocorr = ifft(|fft(f)|^2)
        
        # But we need cross-correlation style. Let's just use the fact that
        # |U - U| = number of distinct differences.
        # We can compute this by: for each possible difference d, check if there exists a in U with a+d in U.
        # That's O(max_u * n) which is too slow for large max_u.
        
        # Better: use FFT-based method.
        # The set U-U has the same size as the set of all a-b for a,b in U.
        # We can compute the indicator of U-U by convolution:
        # Let f[x] = 1 if x in U else 0
        # Let g[x] = 1 if -x in U else 0 (i.e., g[x] = f[-x] but shifted)
        # Actually, U-U = {a-b}. Let's define h[x] = 1 if x in U else 0.
        # Then (h * h_reflected)[k] where h_reflected[x] = h[-x]... 
        # This is getting complicated. Let me just use a direct but optimized approach for diffset.
        
        # For the difference set, since we're dealing with relatively small sets (n <= ~50),
        # we can compute it directly but efficiently.
        # Actually, let's use the FFT approach properly.
        
        # U - U: we want to know which values d = a - b are achievable.
        # Define array f where f[x] = 1 if x in U.
        # The difference set indicator can be computed as:
        # diff_indicator[d] = 1 if exists a: f[a] = 1 and f[a-d] = 1
        # This is the cross-correlation of f with itself.
        # cross_corr[d] = sum_x f[x] f[x-d]
        # We can compute this via FFT: cross_corr = ifft(fft(f) * conj(fft(f)))
        # But this gives circular correlation. We need linear.
        
        # For linear cross-correlation of f with f (autocorrelation):
        # We zero-pad to size >= 2*max_u + 1
        # autocorr[k] for k in [0, 2*max_u] via ifft(|FFT(f_padded)|^2)
        # The full autocorrelation has values for lags -max_u to max_u.
        # autocorr_full[d] = sum_x f[x] f[x+d]
        # We want to know if autocorr_full[d] > 0 for each d.
        
        # Let's compute autocorrelation:
        fft_size2 = 1
        while fft_size2 <= 2 * max_u:
            fft_size2 <<= 1
        
        f2 = np.zeros(fft_size2, dtype=np.float64)
        for x in U:
            f2[x] = 1.0
        
        F2 = np.fft.rfft(f2)
        power = F2 * np.conj(F2)
        autocorr = np.fft.irfft(power, n=fft_size2)
        
        # autocorr[k] for k in [0, fft_size2/2] gives sum_x f2[x] f2[x+k] (with wraparound, but since we zero-padded, it's linear for k <= max_u)
        # Actually, the irfft gives us the full autocorrelation.
        # autocorr[0] = sum f2[x]^2 = n
        # autocorr[k] for k > 0 = sum_x f2[x] f2[x+k] (this is the correlation at lag k)
        # But we also need negative lags. Due to symmetry, autocorr[-k] = autocorr[k].
        # The distinct differences are d where autocorr[|d|] > 0, for d != 0, plus d=0.
        
        # Count distinct differences:
        # d = 0 is always in U-U (a-a=0)
        # For d > 0: d is in U-U iff autocorr[d] > 0 (since d = a-b means b = a-d, so a and a-d both in U)
        # Actually: d = a - b > 0 means a > b, a - b = d. So b = a - d.
        # autocorr[d] = sum_x f[x] f[x+d] = number of pairs (x, x+d) both in U.
        # So d > 0 is in U-U iff autocorr[d] >= 1.
        # Similarly, -d is in U-U iff d is in U-U (by symmetry).
        
        # So |U-U| = 1 + 2 * |{d > 0 : autocorr[d] >= 1}|
        
        # autocorr array: index 0 to fft_size2//2
        # We need to check indices 1 to max_u
        positive_diffs = int(np.sum(autocorr[1:max_u+1] > 0.5))
        n_diff = 1 + 2 * positive_diffs
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def compute_score_direct(U):
        """Direct computation for small sets."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def compute_score(U):
        """Choose best method based on set size and max value."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        n = len(U)
        
        # Use FFT for larger sets or larger max_u
        if n > 15 or max_u > 500:
            return compute_score_fft(U)
        else:
            return compute_score_direct(U)
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Update best if seed is better
    seed_score = compute_score(best_set)
    if seed_score > best_score:
        best_score = seed_score
    
    # Stage 1: Generate diverse initial candidates
    
    candidates = []
    
    # Geometric progressions: {0, 1, B, B^2, ..., B^(k-1)}
    for B in range(2, 50):
        for k in range(2, 15):
            if time_left() < 5:
                break
            U = [0]
            current_val = 1
            for i in range(1, k):
                U.append(current_val)
                current_val *= B
            U = sorted(U)
            candidates.append(list(U))
            
            # Variant without 1: {0, B, B^2, ..., B^(k-1)}
            U2 = [0]
            current_val = B
            for i in range(1, k):
                U2.append(current_val)
                current_val *= B
            U2 = sorted(U2)
            candidates.append(list(U2))
    
    if time_left() < 5:
        return best_set
    
    # Evaluate all candidates
    scored_candidates = []
    for cand in candidates:
        if time_left() < 5:
            break
        s = compute_score(cand)
        scored_candidates.append((s, cand))
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Sort by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Stage 2: Local search with FFT-based scoring
    # Refine top candidates
    
    # We'll do local search on the top few candidates
    num_to_refine = min(10, len(scored_candidates))
    
    for idx in range(num_to_refine):
        if time_left() < 3:
            break
        
        current_set = list(scored_candidates[idx][1])
        current_score = scored_candidates[idx][0]
        
        # Local search parameters
        max_iterations = 500
        iterations = 0
        no_improve_count = 0
        max_no_improve = 50
        
        while iterations < max_iterations and time_left() > 3 and no_improve_count < max_no_improve:
            iterations += 1
            n = len(current_set)
            max_val = max(current_set) if current_set else 0
            
            # Generate a set of candidate moves
            moves = []
            
            # 1. Try adding a new element
            if n < 60:
                # Try values in strategic locations
                # Near existing elements
                for x in current_set:
                    for delta in [1, 2, 3, 5, 10]:
                        new_val = x + delta
                        if 0 < new_val and new_val not in current_set:
                            moves.append(('add', new_val))
                        new_val2 = x - delta
                        if 0 < new_val2 and new_val2 not in current_set:
                            moves.append(('add', new_val2))
                
                # Try random values in range
                for _ in range(30):
                    new_val = random.randint(1, max_val * 3)
                    if new_val not in current_set:
                        moves.append(('add', new_val))
                
                # Try values based on gaps
                if n >= 2:
                    for i in range(n - 1):
                        gap_start = current_set[i]
                        gap_end = current_set[i + 1]
                        if gap_end - gap_start > 1:
                            # Try midpoint or other positions in gap
                            mid = (gap_start + gap_end) // 2
                            if mid not in current_set and mid > 0:
                                moves.append(('add', mid))
                            # Try a few random positions in gap
                            for _ in range(3):
                                rv = random.randint(gap_start + 1, gap_end - 1)
                                if rv not in current_set:
                                    moves.append(('add', rv))
            
            # 2. Try replacing an element
            if n >= 3:
                for i in range(n):
                    if current_set[i] == 0:
                        continue
                    # Try small perturbations
                    for delta in [-3, -2, -1, 1, 2, 3, 5, 10, -10, -5]:
                        new_val = current_set[i] + delta
                        if new_val < 0 or new_val == 0 or new_val in current_set:
                            continue
                        moves.append(('replace', i, new_val))
                    
                    # Try larger jumps
                    if max_val > 0:
                        for scale in [0.5, 1.5, 2.0]:
                            new_val = int(current_set[i] * scale)
                            if new_val > 0 and new_val not in current_set:
                                moves.append(('replace', i, new_val))
            
            # 3. Try removing an element (if size > 3)
            if n > 3 and random.random() < 0.3:
                for i in range(n):
                    if current_set[i] == 0:
                        continue
                    moves.append(('remove', i))
            
            if not moves:
                no_improve_count += 1
                continue
            
            # Shuffle moves for diversity
            random.shuffle(moves)
            
            improved = False
            # Try a limited number of moves per iteration
            num_moves_to_try = min(len(moves), 20)
            
            for move in moves[:num_moves_to_try]:
                if time_left() < 3:
                    break
                
                new_set = list(current_set)
                
                if move[0] == 'add':
                    new_val = move[1]
                    new_set.append(new_val)
                    new_set = sorted(set(new_set))
                elif move[0] == 'replace':
                    i, new_val = move[1], move[2]
                    new_set[i] = new_val
                    new_set = sorted(set(new_set))
                    if 0 not in new_set:
                        continue
                elif move[0] == 'remove':
                    i = move[1]
                    new_set.pop(i)
                    new_set = sorted(set(new_set))
                    if 0 not in new_set:
                        continue
                
                if len(new_set) != len(set(new_set)):
                    continue
                
                new_score = compute_score(new_set)
                
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    improved = True
                    no_improve_count = 0
                    break  # Restart with new set
            
            if not improved:
                no_improve_count += 1
    
    # Stage 3: Final polish - try to improve best_set further
    if time_left() > 5:
        current_set = list(best_set)
        current_score = best_score
        max_iterations = 200
        iterations = 0
        no_improve_count = 0
        max_no_improve = 30
        
        while iterations < max_iterations and time_left() > 2 and no_improve_count < max_no_improve:
            iterations += 1
            n = len(current_set)
            max_val = max(current_set) if current_set else 0
            
            moves = []
            
            # Add
            if n < 80:
                for x in current_set:
                    for delta in [1, 2, 3, 5]:
                        nv = x + delta
                        if 0 < nv and nv not in current_set:
                            moves.append(('add', nv))
                        nv2 = x - delta
                        if 0 < nv2 and nv2 not in current_set:
                            moves.append(('add', nv2))
                for _ in range(20):
                    nv = random.randint(1, max_val * 2)
                    if nv not in current_set:
                        moves.append(('add', nv))
            
            # Replace
            if n >= 3:
                for i in range(n):
                    if current_set[i] == 0:
                        continue
                    for delta in [-2, -1, 1, 2, 3, -3]:
                        nv = current_set[i] + delta
                        if nv > 0 and nv not in current_set:
                            moves.append(('replace', i, nv))
            
            # Remove
            if n > 3 and random.random() < 0.2:
                for i in range(n):
                    if current_set[i] != 0:
                        moves.append(('remove', i))
            
            if not moves:
                no_improve_count += 1
                continue
            
            random.shuffle(moves)
            improved = False
            for move in moves[:15]:
                if time_left() < 2:
                    break
                
                new_set = list(current_set)
                if move[0] == 'add':
                    new_set.append(move[1])
                    new_set = sorted(set(new_set))
                elif move[0] == 'replace':
                    new_set[move[1]] = move[2]
                    new_set = sorted(set(new_set))
                    if 0 not in new_set:
                        continue
                elif move[0] == 'remove':
                    new_set.pop(move[1])
                    new_set = sorted(set(new_set))
                    if 0 not in new_set:
                        continue
                
                if len(new_set) != len(set(new_set)):
                    continue
                
                new_score = compute_score(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
    
    return list(best_set)
