import numpy as np
import time
import itertools

def calc_score(U):
    if not U:
        return 1.0
    U = sorted(list(set(U)))
    if 0 not in U:
        return 1.0 # Invalid per task description "containing 0", but score formula might still work. 
        # Task says "containing 0".
    
    max_u = max(U)
    # Compute sumset and diffset sizes efficiently for small/medium U
    # For U size n, |U+U| <= n^2.
    # If n is large (e.g., > 1000), set difference might be slow if done naively.
    # But we are constructing structured sets, so we can compute sizes analytically or using intervals.
    
    # For the structured sets (unions of intervals), we can compute sizes analytically.
    # However, to be safe and general, let's use a fast method for the specific structure.
    # Structure: Union of k intervals of length m.
    
    # Let's just compute it using sets for small m, and analytic for large m?
    # No, let's stick to the analytic calculation for the specific "block" structure.
    
    return 1.0

def score_blocks(blocks):
    # blocks is a list of (start, length)
    # U = union of [start, start+length-1]
    # We need |U+U| and |U-U|.
    
    # This is complex analytically for k>2.
    # Let's use a numerical approach with a time guard.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Search 2-block sets: {0..m-1} U {N..N+m-1}
    # m in [2, 50], N in [m, 10*m]
    for m in range(2, 51):
        if time.time() > deadline:
            break
        for N in range(m, 10*m + 1):
            if time.time() > deadline:
                break
            U = list(range(0, m)) + list(range(N, N+m))
            U = sorted(set(U))
            
            # Calculate score
            # For small m, use sets
            if len(U) <= 200:
                sumset = set()
                diffset = set()
                for a in U:
                    for b in U:
                        sumset.add(a+b)
                        diffset.add(a-b)
                size_sum = len(sumset)
                size_diff = len(diffset)
                max_u = max(U)
                score = 1.0 + np.log(size_diff / size_sum) / np.log(2 * max_u + 1)
                
                if score > best_score:
                    best_score = score
                    best_set = U
            else:
                # Analytic approximation for 2 blocks
                # If N >= 2m, blocks in sumset are disjoint
                # Sumset: [0, 2m-2] and [N, N+2m-2]
                # Size = 2*(2m-1) = 4m-2
                # Diffset: 
                # Internal diffs: [-(m-1), m-1] (size 2m-1)
                # Cross diffs: N - [0, m-1] = [N-m+1, N] (size m)
                #            [0, m-1] - N = [-N, -(N-m+1)] (size m)
                # Are they disjoint?
                # Internal: max m-1. Cross min: N-m+1.
                # If N-m+1 > m-1 => N > 2m-2, they are disjoint.
                # If N <= 2m-2, they overlap.
                
                if N > 2*m - 2:
                    size_sum = 4*m - 2
                    size_diff = (2*m - 1) + m + m
                else:
                    # Overlap case is complex, skip or approximate
                    continue
                    
                max_u = N + m - 1
                score = 1.0 + np.log(size_diff / size_sum) / np.log(2 * max_u + 1)
                
                if score > best_score:
                    best_score = score
                    best_set = U

    # Search 3-block sets: {0..m-1} U {N1..N1+m-1} U {N2..N2+m-1}
    # m in [2, 20]
    for m in range(2, 21):
        if time.time() > deadline:
            break
        # Randomize N1, N2
        for _ in range(50):
            if time.time() > deadline:
                break
            N1 = np.random.randint(m, 5*m)
            N2 = np.random.randint(N1 + m, 10*m)
            
            U = list(range(0, m)) + list(range(N1, N1+m)) + list(range(N2, N2+m))
            U = sorted(set(U))
            
            if len(U) <= 200:
                sumset = set()
                diffset = set()
                for a in U:
                    for b in U:
                        sumset.add(a+b)
                        diffset.add(a-b)
                size_sum = len(sumset)
                size_diff = len(diffset)
                max_u = max(U)
                score = 1.0 + np.log(size_diff / size_sum) / np.log(2 * max_u + 1)
                
                if score > best_score:
                    best_score = score
                    best_set = U

    return best_set
