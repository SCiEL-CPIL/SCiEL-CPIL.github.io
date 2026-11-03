import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate the seed code: direct sum of ternary tetracode and all words of length n-4
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all tails of length n-4
    tails = list(itertools.product(range(q), repeat=n-4))
    
    # Construct the full code
    code = [h + list(tail) for h in tetracode for tail in tails]
    
    # Convert to numpy array for efficient operations
    code_arr = np.array(code, dtype=np.int8)
    
    # Generate all words in the space {0..q-1}^n
    space_size = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Efficient word-to-index conversion using base-q encoding
    # Precompute powers of q for fast conversion
    q_powers = [1] * n
    for i in range(n - 1, 0, -1):
        q_powers[i] = q_powers[i + 1] * q
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    # Precompute ball indices for each possible codeword position
    # ball_indices[cw_idx] = list of space word indices covered by codeword at position cw_idx
    codeword_balls = []
    for cw in code:
        ball_indices = []
        idx_center = word_to_idx(cw)
        ball_indices.append(idx_center)
        for pos in range(n):
            for val in range(q):
                if val == cw[pos]:
                    continue
                new_word = list(cw)
                new_word[pos] = val
                idx = word_to_idx(new_word)
                ball_indices.append(idx)
        codeword_balls.append(ball_indices)
    
    # Initialize coverage counts
    cover_count = np.zeros(space_size, dtype=np.int32)
    for ball in codeword_balls:
        cover_count[ball] += 1
    
    # Active codewords
    active = [True] * len(code)
    
    # Greedy removal: remove codewords with no exclusive coverage
    removed_any = True
    while removed_any:
        if time.time() > deadline:
            break
        removed_any = False
        for i in range(len(code)):
            if not active[i]:
                continue
            has_exclusive = False
            for idx in codeword_balls[i]:
                if cover_count[idx] == 1:
                    has_exclusive = True
                    break
            if not has_exclusive:
                active[i] = False
                for idx in codeword_balls[i]:
                    cover_count[idx] -= 1
                removed_any = True
    
    # Local search: try replacing codewords to reduce total count
    # We'll try to find a codeword that can be replaced by a non-codeword
    # that covers the same or more uniquely covered points
    best_count = sum(active)
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False
        # Try each active codeword
        for i in range(len(code)):
            if time.time() > deadline:
                break
            if not active[i]:
                continue
            
            # Count how many points are exclusively covered by this codeword
            exclusive_count = 0
            for idx in codeword_balls[i]:
                if cover_count[idx] == 1:
                    exclusive_count += 1
            
            # If no exclusive coverage, just remove it
            if exclusive_count == 0:
                active[i] = False
                for idx in codeword_balls[i]:
                    cover_count[idx] -= 1
                best_count -= 1
                improvement_found = True
                continue
            
            # Try to find a replacement codeword from the space
            # that covers all exclusively-covered points and doesn't add too many new ones
            # We'll sample some candidates
            current_ball_set = set(codeword_balls[i])
            
            # Sample candidate positions
            num_candidates = min(50, space_size // 10)
            candidates = random.sample(range(space_size), num_candidates)
            
            best_candidate = -1
            best_exclusive_covered = 0
            
            for cand_idx in candidates:
                cand_word = idx_to_word(cand_idx)
                # Compute ball for candidate
                cand_ball = []
                cand_ball.append(cand_idx)
                for pos in range(n):
                    for val in range(q):
                        if val == cand_word[pos]:
                            continue
                        new_word = list(cand_word)
                        new_word[pos] = val
                        cand_ball.append(word_to_idx(new_word))
                
                # Check if candidate covers all exclusive points
                cand_ball_set = set(cand_ball)
                exclusive_covered = 0
                for idx in codeword_balls[i]:
                    if cover_count[idx] == 1 and idx in cand_ball_set:
                        exclusive_covered += 1
                
                if exclusive_covered > best_exclusive_covered:
                    best_exclusive_covered = exclusive_covered
                    best_candidate = cand_idx
            
            # If we found a candidate that covers all exclusive points
            if best_candidate >= 0 and best_exclusive_covered == exclusive_count:
                # Check if best_candidate is already in our code
                if best_candidate not in [word_to_idx(code[j]) for j in range(len(code)) if active[j]]:
                    # Replace codeword i with candidate
                    active[i] = False
                    for idx in codeword_balls[i]:
                        cover_count[idx] -= 1
                    
                    # Add new codeword
                    code.append(idx_to_word(best_candidate))
                    new_ball = []
                    cw = code[-1]
                    new_ball.append(best_candidate)
                    for pos in range(n):
                        for val in range(q):
                            if val == cw[pos]:
                                continue
                            new_word = list(cw)
                            new_word[pos] = val
                            new_ball.append(word_to_idx(new_word))
                    codeword_balls.append(new_ball)
                    active.append(True)
                    for idx in new_ball:
                        cover_count[idx] += 1
                    improvement_found = True
    
    # Final greedy removal pass
    removed_any = True
    while removed_any:
        if time.time() > deadline:
            break
        removed_any = False
        for i in range(len(code)):
            if not active[i]:
                continue
            has_exclusive = False
            for idx in codeword_balls[i]:
                if cover_count[idx] == 1:
                    has_exclusive = True
                    break
            if not has_exclusive:
                active[i] = False
                for idx in codeword_balls[i]:
                    cover_count[idx] -= 1
                removed_any = True
    
    final_code = [code[i] for i in range(len(code)) if active[i]]
    
    return final_code
