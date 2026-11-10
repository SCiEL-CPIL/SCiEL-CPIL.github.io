import itertools
import numpy as np
import time

def construct(q, n, radius):
    # Generate all possible words
    all_words = np.array(list(itertools.product(range(q), repeat=n)))
    num_words = len(all_words)
    
    # Pre-compute is_in_ball
    a = all_words[:, None, :]
    b = all_words[None, :, :]
    dist = np.sum(a != b, axis=2)
    is_in_ball = dist <= radius
    
    # Convert to float for faster matrix multiplication
    is_in_ball_float = is_in_ball.astype(np.float32)
    
    best_code_indices = None
    best_len = num_words + 1
    
    # Set a time limit
    start_time = time.time()
    time_limit = 110
    
    rng = np.random.default_rng(42)
    
    # Threshold for switching to sparse/loop method
    # If uncovered count is below this, use the loop method
    threshold = 50
    
    while time.time() - start_time < time_limit:
        covered = np.zeros(num_words, dtype=bool)
        code_indices = []
        
        # Phase 1: Dense Greedy
        while covered.sum() < num_words and (num_words - covered.sum()) > threshold:
            uncovered = ~covered
            uncovered_float = uncovered.astype(np.float32)
            
            # Compute gains: dot product of is_in_ball_float (N x N) and uncovered_float (N x 1)
            # Result is N x 1 array of gains
            gains = is_in_ball_float @ uncovered_float
            gains = gains.flatten()
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > 1:
                best_idx = candidates[rng.integers(len(candidates))]
            else:
                best_idx = candidates[0]
                
            code_indices.append(best_idx)
            covered |= is_in_ball[best_idx]
            
        # Phase 2: Sparse/Loop Greedy for the tail
        if covered.sum() < num_words:
            while covered.sum() < num_words:
                uncovered_indices = np.where(~covered)[0]
                if len(uncovered_indices) == 0:
                    break
                    
                # Gains for each candidate codeword
                # is_in_ball is (N, N)
                # is_in_ball[:, uncovered_indices] is (N, K)
                gains = np.sum(is_in_ball[:, uncovered_indices], axis=1)
                
                max_gain = np.max(gains)
                
                if max_gain == 0:
                    break
                    
                candidates = np.where(gains == max_gain)[0]
                
                if len(candidates) > 1:
                    best_idx = candidates[rng.integers(len(candidates))]
                else:
                    best_idx = candidates[0]
                    
                code_indices.append(best_idx)
                covered |= is_in_ball[best_idx]
                
        if len(code_indices) < best_len:
            best_len = len(code_indices)
            best_code_indices = code_indices
            
    if best_code_indices is None:
        return []
        
    return [all_words[i].tolist() for i in best_code_indices]
