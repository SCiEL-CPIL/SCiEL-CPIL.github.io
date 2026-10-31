import numpy as np
import random

def construct():
    def calc_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -np.inf
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        size = 2 * max_u + 1
        n_fft = 1
        while n_fft < size:
            n_fft *= 2
            
        vec = np.zeros(n_fft, dtype=np.float64)
        for x in U:
            vec[x] = 1.0
            
        fft_vec = np.fft.fft(vec)
        
        # Sumset U+U
        sum_poly = np.fft.ifft(fft_vec * fft_vec).real
        sum_counts = np.round(sum_poly).astype(int)
        # U+U is in [0, 2*max_u]
        size_sum = np.sum(sum_counts[:2*max_u+1] > 0)
        
        # Diffset U-U
        corr = np.fft.ifft(fft_vec * np.conj(fft_vec)).real
        diff_counts = np.round(corr).astype(int)
        
        # Diffset is in [-max_u, max_u]
        # Indices:
        # 0 -> 0
        # 1..max_u -> 1..max_u
        # n_fft - max_u .. n_fft - 1 -> -max_u .. -1
        # We need to count non-zeros in [0, max_u] and [n_fft - max_u, n_fft)
        # Note: index 0 is counted in the first part.
        size_diff = np.sum(diff_counts[:max_u+1] > 0) + np.sum(diff_counts[n_fft - max_u:] > 0)
        # Wait, diff_counts[n_fft - max_u:] includes indices n_fft - max_u to n_fft - 1.
        # If max_u = 0, n_fft - 0 = n_fft, slice is empty. Correct.
        # If max_u = 1, n_fft - 1. Slice is [n_fft-1]. Index n_fft-1 corresponds to lag -1.
        # Index 0 corresponds to lag 0.
        # Indices 1..max_u correspond to lags 1..max_u.
        # So diff_counts[0:max_u+1] covers lags 0..max_u.
        # diff_counts[n_fft - max_u : n_fft] covers lags -max_u .. -1.
        # These sets of lags are disjoint (except if max_u=0, but handled).
        
        if size_sum == 0 or size_diff == 0:
            return -np.inf
            
        ratio = size_diff / size_sum
        if ratio <= 0:
            return -np.inf
            
        score = 1 + np.log(ratio) / np.log(2 * max_u + 1)
        return score

    # Initial best set
    # Start with a small set that might be good, then expand
    # The paper mentions sizes 2003 and 54265, but we are limited by time.
    # Let's try to find a good set of size ~50-100.
    
    best_set = [0, 1, 2, 3]
    best_score = calc_score(best_set)
    
    random.seed(42)
    
    # Simulated Annealing / Hill Climbing
    # We want to maximize the score.
    # Score = 1 + log(|U-U|/|U+U|) / log(2*maxU+1)
    # We want |U-U| large and |U+U| small relative to maxU.
    # |U-U| is at most 2*maxU+1. |U+U| is at most 2*maxU+1.
    # If U is a Sidon set, |U+U| is ~2|U|. |U-U| is ~2|U|^2? No, |U-U| is at most 2|U|^2.
    # Actually, for a set of size N, |U+U| >= 2N-1. |U-U| <= 2N^2.
    # Max possible ratio is ~N.
    # log(N) / log(2*maxU).
    # If maxU is large, the denominator is large.
    # We want maxU to be small relative to N?
    # If U = {0, 1, ..., N-1}, maxU = N-1.
    # |U+U| = 2N-1. |U-U| = 2N-1. Ratio = 1. Score = 1.
    # If U is sparse, maxU is large.
    # Example: U = {0, 1, 2, 4, 8, ...}. maxU = 2^k. N = k+1.
    # |U+U| is roughly 2N^2? No, sumset of powers of 2 is all numbers with bit count <= 2?
    # Actually, sumset of {0, 1, 2, 4} is {0,1,2,3,4,5,6,7,8}. Size 9. N=4. 2N-1=7.
    # Diffset is symmetric.
    
    # Let's use a random walk.
    max_iterations = 5000
    current_set = best_set.copy()
    current_score = best_score
    
    # Try to grow the set
    for i in range(max_iterations):
        U = current_set.copy()
        max_u = max(U)
        
        # Mutation
        op = random.choice(['add', 'remove', 'swap'])
        
        if op == 'add' and len(U) < 200:
            # Add element in [0, 2*max_u + 10]
            # Bias towards smaller values to keep max_u down?
            # Or larger values to increase range?
            # Let's try uniform in [0, 2*max_u + 50]
            new_val = random.randint(0, 2 * max_u + 50)
            if new_val not in U:
                U.append(new_val)
        elif op == 'remove' and len(U) > 1:
            # Remove random non-zero
            idx = random.randint(1, len(U)-1)
            U.pop(idx)
        elif op == 'swap':
            if len(U) > 1:
                idx = random.randint(1, len(U)-1)
                # Replace with value in [0, 2*max_u + 50]
                new_val = random.randint(0, 2 * max_u + 50)
                if new_val not in U or U[idx] == new_val:
                    if new_val in U and U[idx] != new_val:
                         # Collision, reject or pick another
                         pass
                    else:
                        U[idx] = new_val
        
        if 0 not in U:
            # Ensure 0 is in U
            if 0 in U:
                pass
            else:
                # Remove a random non-zero and add 0
                idx = random.randint(0, len(U)-1)
                U[idx] = 0
                U = list(set(U))
                if len(U) == 0:
                    continue

        new_score = calc_score(U)
        
        # Accept if better, or with some probability if worse (annealing)
        # Simple hill climbing first
        if new_score > best_score:
            best_score = new_score
            best_set = U
            current_set = U
            current_score = new_score
        elif new_score > current_score:
            # Accept to escape local minima?
            # Let's do simple hill climbing for stability
            pass
        else:
            # Random restart?
            if i % 500 == 0 and i > 0:
                 # Keep best, but maybe perturb current
                 pass

    return sorted(best_set)
