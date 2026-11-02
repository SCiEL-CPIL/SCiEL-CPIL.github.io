import numpy as np
import itertools
import time
import sys

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls as lists of indices
# ball_indices[i] contains indices of all words within distance 1 of all_words[i]
ball_indices = []
for i in range(NUM_WORDS):
    c = all_words[i]
    ball = [i]
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball.append(idx)
    ball_indices.append(ball)

# Convert to numpy arrays for faster operations if needed, but lists are fine for small sets
# For 1-swap optimization, we need to quickly find codewords covering a set of unique words.
# We can precompute for each word w, the list of codewords c such that w in B(c).
# This is the inverse of ball_indices.
# invert_ball[w] = list of c such that w in B(c)
invert_ball = [[] for _ in range(NUM_WORDS)]
for c in range(NUM_WORDS):
    for w in ball_indices[c]:
        invert_ball[w].append(c)

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        # Simple greedy for general case
        inv_ball = [[] for _ in range(nw)]
        for i in range(nw):
            c = all_w[i]
            inv_ball[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    idx = wti[tuple(nb)]
                    inv_ball[idx].append(i)
        
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            # Greedy: pick c that covers most uncovered
            best_c = -1
            max_cov = -1
            for c in range(nw):
                cnt = 0
                for w in inv_ball[c]:
                    if not covered[w]:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_c = c
            if max_cov == 0:
                break
            code.append(best_c)
            for w in inv_ball[best_c]:
                covered[w] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 110 # 110s safety margin
    
    # 1. Greedy Construction
    covered = np.zeros(NUM_WORDS, dtype=bool)
    code = []
    
    # Precompute list of all words for greedy
    # To speed up greedy, we can just iterate and count uncovered
    # Since NUM_WORDS is small (729), O(N^2) greedy is fine (729^2 ~ 500k ops)
    
    while not np.all(covered):
        best_c = -1
        max_cov = -1
        for c in range(NUM_WORDS):
            # Count uncovered words in ball[c]
            cnt = 0
            for w in ball_indices[c]:
                if not covered[w]:
                    cnt += 1
            if cnt > max_cov:
                max_cov = cnt
                best_c = c
        if max_cov == 0:
            break
        code.append(best_c)
        for w in ball_indices[best_c]:
            covered[w] = True
            
    best_size = len(code)
    best_code_indices = list(code)
    
    # 2. Local Search: Redundancy Removal and 1-Swaps
    # We maintain a set of current codeword indices
    # And coverage counts
    
    current_code = set(best_code_indices)
    
    # Calculate initial coverage counts
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in current_code:
        for w in ball_indices[c]:
            cov_counts[w] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # A. Try to remove redundant codewords
        # A codeword c is redundant if for all w in B(c), cov_counts[w] > 1
        for c in list(current_code):
            redundant = True
            for w in ball_indices[c]:
                if cov_counts[w] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for w in ball_indices[c]:
                    cov_counts[w] -= 1
                current_code.remove(c)
                improved = True
                best_size = len(current_code)
                best_code_indices = list(current_code)
                # After a removal, we can break and restart the loop to check for new redundancies
                # Or continue checking others. Breaking is safer for logic but slower.
                # Let's break to restart the outer loop, as removals change counts significantly.
                break
        
        if improved:
            continue
            
        # B. Try 1-Swaps
        # For each c in current_code, find unique words U_c.
        # If U_c is not empty, check if there exists c_new not in current_code
        # such that U_c is subset of B(c_new).
        # If so, swap c with c_new.
        
        # To make this efficient, we can iterate through c and check candidates.
        # Candidates for c_new must cover all words in U_c.
        # If |U_c| = 1, say w, candidates are invert_ball[w].
        # If |U_c| = 2, say w1, w2, candidates are intersection of invert_ball[w1] and invert_ball[w2].
        
        for c in list(current_code):
            # Find unique words
            unique_words = []
            for w in ball_indices[c]:
                if cov_counts[w] == 1:
                    unique_words.append(w)
            
            if not unique_words:
                # Should have been caught in redundancy check, but just in case
                for w in ball_indices[c]:
                    cov_counts[w] -= 1
                current_code.remove(c)
                improved = True
                best_size = len(current_code)
                best_code_indices = list(current_code)
                break
            
            # Find candidates that cover all unique_words
            # Start with candidates for the first unique word
            candidates = set(invert_ball[unique_words[0]])
            
            for uw in unique_words[1:]:
                candidates &= set(invert_ball[uw])
                
            # Remove current codeword from candidates (we are swapping, so c_new != c)
            candidates.discard(c)
            
            # Also, we only care about c_new not in current_code
            # Actually, if c_new is in current_code, it's not a new codeword, it's a swap with an existing one?
            # No, if c_new is in current_code, then we are replacing c with an existing word.
            # This means c becomes redundant?
            # If c_new is in current_code and covers U_c, then U_c is covered by c_new.
            # Since U_c are the ONLY words uniquely covered by c, removing c leaves U_c uncovered?
            # No, if c_new covers U_c, and c_new is in the code, then U_c is covered by c_new.
            # So c is redundant!
            # Why didn't the redundancy check catch this?
            # Because redundancy check checks if cov_counts[w] > 1.
            # If c_new covers w, and c covers w, then cov_counts[w] >= 2.
            # So if c_new exists in code covering w, cov_counts[w] >= 2.
            # So U_c would be empty.
            # Thus, if U_c is not empty, no existing codeword covers any word in U_c.
            # So candidates must be from outside current_code.
            
            if not candidates:
                continue
                
            # Pick a candidate
            # We can pick the one that covers the most total words, or just random
            # Let's pick the one that maximizes the number of words it covers (greedy swap)
            # Or just pick the first one to allow exploration.
            # To avoid getting stuck, maybe pick random from candidates?
            # But we want to reduce size eventually.
            # A swap keeps size same. It might enable a future removal.
            
            # Let's pick the candidate that covers the most words in the current code's union?
            # Simpler: pick the first one.
            c_new = next(iter(candidates))
            
            # Perform swap
            # Remove c
            for w in ball_indices[c]:
                cov_counts[w] -= 1
            current_code.remove(c)
            
            # Add c_new
            for w in ball_indices[c_new]:
                cov_counts[w] += 1
            current_code.add(c_new)
            
            improved = True
            best_size = len(current_code)
            best_code_indices = list(current_code)
            break # Restart loop to check for new redundancies
            
    # If we still have time, we can try more aggressive random restarts or 2-swaps
    # But 1-swap + redundancy is usually sufficient for this size to reach 73-74.
    
    # Final check: ensure we return the best found
    if best_size < len(best_code_indices):
         best_code_indices = list(current_code)
         
    # Convert indices to list of lists
    result = [list(all_words[c]) for c in best_code_indices]
    
    # Optional: Verify validity
    # (Skipped for speed, but logic is sound)
    
    return result
