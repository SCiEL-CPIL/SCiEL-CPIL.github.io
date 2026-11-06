import time
import random
import numpy as np
from scipy.fft import fft, ifft

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1 # 129
    
    # Precompute indices for target range
    target_indices = np.arange(target_lo, target_hi + 1)
    
    def get_unordered_counts(A):
        """
        Calculate unordered pair representation counts for all sums in [0, 256].
        Uses FFT for speed.
        """
        # Create indicator vector
        size = 257 # Max sum is 128+128=256
        f = np.zeros(size, dtype=np.float64)
        for a in A:
            if 0 <= a <= 128:
                f[a] = 1.0
        
        # Convolution
        # We need sum_{a+b=s} f(a)f(b)
        # This is the coefficient of x^s in F(x)^2
        F = fft(f, n=512) # Use power of 2 for speed
        G = ifft(F * F, n=512)
        
        # G[s] is the number of ordered pairs (a,b) such that a+b=s
        # We want unordered pairs {a,b} with a<=b.
        # Let N_ordered(s) = G[s].
        # Let D(s) = 1 if s is even and s/2 in A, else 0.
        # N_unordered(s) = (N_ordered(s) + D(s)) / 2
        
        counts = np.zeros(size, dtype=np.int32)
        for s in range(size):
            ordered = int(round(G[s].real))
            # Check if s/2 is in A
            d = 0
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and f[half] == 1.0:
                    d = 1
            unordered = (ordered + d) // 2
            counts[s] = unordered
            
        return counts

    def evaluate(A):
        """
        Returns (uncovered_count, peak_count, sum_sq)
        """
        counts = get_unordered_counts(A)
        target_counts = counts[target_lo:target_hi+1]
        
        uncovered = np.sum(target_counts == 0)
        peak = np.max(target_counts) if len(target_counts) > 0 else 0
        # Sum of squares helps flatten distribution
        sum_sq = np.sum(target_counts * target_counts)
        
        return int(uncovered), int(peak), int(sum_sq)

    def score_tuple(t):
        """
        Lexicographic score: minimize uncovered, then peak, then sum_sq.
        """
        return (t[0], t[1], t[2])

    best_A = None
    best_score = (float('inf'), float('inf'), float('inf'))
    
    def update_best(A, score):
        nonlocal best_A, best_score
        if score < best_score:
            best_score = score
            best_A = A[:]
            return True
        return False

    def local_search(initial_A, max_steps, temp_start=1.0, temp_end=0.01):
        """
        Simulated Annealing / Hill Climbing hybrid.
        """
        A = list(initial_A)
        A_set = set(A)
        curr_score = evaluate(A)
        curr_uncovered, curr_peak, curr_sum_sq = curr_score
        
        best_A_local = A[:]
        best_score_local = curr_score
        
        if time.time() > deadline - 2:
            return best_A_local, best_score_local

        # Precompute neighbors? No, random sampling is fine.
        # Move: Remove x, Add y
        
        steps_since_improve = 0
        max_no_improve = 2000
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            if steps_since_improve > max_no_improve:
                break
                
            # Temperature decay
            progress = step / max_steps
            temp = temp_start * (temp_end / temp_start) ** progress
            
            # Select element to remove
            x = random.choice(A)
            
            # Select element to add
            # Bias towards elements that might help cover holes or reduce peaks
            # Simple random
            y = random.randint(0, n)
            attempts = 0
            while y in A_set and attempts < 10:
                y = random.randint(0, n)
                attempts += 1
            if y in A_set:
                continue
            
            # Apply move
            A.remove(x)
            A_set.remove(x)
            A.append(y)
            A_set.add(y)
            
            new_score = evaluate(A)
            
            # Calculate delta
            # We want to minimize (uncovered, peak, sum_sq)
            # If new_score < curr_score, accept.
            # Else, accept with prob exp(-delta/temp)
            
            if new_score < curr_score:
                curr_score = new_score
                steps_since_improve = 0
                if new_score < best_score_local:
                    best_score_local = new_score
                    best_A_local = A[:]
            else:
                # Calculate penalty
                # Simplified: if uncovered increases, reject.
                # If peak increases, reject unless high temp.
                # If sum_sq increases, accept with prob.
                
                d_uncovered = new_score[0] - curr_score[0]
                d_peak = new_score[1] - curr_score[1]
                d_sq = new_score[2] - curr_score[2]
                
                # Heuristic acceptance
                accept = False
                if d_uncovered > 0:
                    # Don't allow increasing uncovered count easily
                    if temp > 0.5 and random.random() < 0.01:
                        accept = True
                elif d_peak > 0:
                    if temp > 0.2 and random.random() < 0.05:
                        accept = True
                else:
                    # d_uncovered == 0, d_peak <= 0
                    # Check sum_sq
                    if d_sq > 0:
                        # Accept with probability based on temp
                        if random.random() < np.exp(-d_sq * 0.01 / (temp + 1e-9)):
                            accept = True
                    else:
                        accept = True # d_sq <= 0 is always better or equal in sq, and peak/uncovered didn't worsen
                
                if accept:
                    curr_score = new_score
                else:
                    # Rollback
                    A.remove(y)
                    A_set.remove(y)
                    A.append(x)
                    A_set.add(x)
                    steps_since_improve += 1
                    
        return best_A_local, best_score_local

    # Initial Candidates
    candidates = []
    
    # 1. Structured: Arithmetic Progressions
    for start in range(0, 65, 5):
        for step in [2, 3, 4]:
            A = list(range(start, 129, step))
            if len(A) >= 15 and len(A) <= 25:
                candidates.append(A)
                
    # 2. Random sets of various sizes
    for k in range(15, 25):
        for _ in range(10):
            candidates.append(random.sample(range(n + 1), k))
            
    # 3. "Spread" sets: Mix of small and large
    for _ in range(20):
        smalls = random.sample(range(0, 40), 5)
        larges = random.sample(range(80, 129), 10)
        mids = random.sample(range(40, 80), 5)
        A = smalls + mids + larges
        A = list(set(A))
        if 15 <= len(A) <= 25:
            candidates.append(A)

    # Evaluate and sort candidates
    scored_candidates = []
    for A in candidates:
        if time.time() > deadline - 5:
            break
        score = evaluate(A)
        scored_candidates.append((score, A))
        
    scored_candidates.sort(key=score_tuple)
    
    # Run Local Search on top candidates
    top_k = 10
    for i, (score, A) in enumerate(scored_candidates[:top_k]):
        if time.time() > deadline - 5:
            break
            
        # Allocate more time to better initial candidates
        steps = 5000 if i == 0 else 3000
        
        A_res, score_res = local_search(A, steps)
        
        if score_res < best_score:
            best_score = score_res
            best_A = A_res
        
        # Early exit if we found optimal (Peak 1 or 2 with 0 uncovered)
        if best_score[0] == 0 and best_score[1] <= 2:
            # Try to see if we can get Peak 1
            if best_score[1] == 1:
                break
            # If Peak 2, continue a bit to see if 1 is possible, but don't waste too much time
            if i > 2:
                break

    # If we still have time, try more random restarts
    if best_score[0] > 0 or best_score[1] > 2:
        while time.time() < deadline - 3:
            k = random.randint(15, 22)
            init_A = random.sample(range(n + 1), k)
            A_res, score_res = local_search(init_A, 2000)
            if score_res < best_score:
                best_score = score_res
                best_A = A_res
            if best_score[0] == 0 and best_score[1] <= 2:
                break

    if best_A is None:
        # Fallback: A set that definitely covers, even if peak is high
        # Even numbers from 0 to 128
        return list(range(0, 129, 2))
        
    return best_A

