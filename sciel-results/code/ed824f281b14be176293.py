import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Map word tuple to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls B(i, radius) for all i
    # Since radius is 1, B(i, 1) = {i} U {j : dist(i, j) = 1}
    ball_list = [[] for _ in range(total_words)]
    ball_bitsets = [0] * total_words
    for i in range(total_words):
        word = idx_to_word(i)
        ball = [i]
        # Generate all neighbors at distance 1
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_word = word.copy()
                new_word[pos] = new_val
                j = word_to_idx(new_word)
                ball.append(j)
        ball_list[i] = ball
        # Build bitset
        bs = 0
        for j in ball:
            bs |= (1 << j)
        ball_bitsets[i] = bs

    # State
    covered_bits = 0  # bitset of covered words
    code = []
    cover_count = [0] * total_words

    def add_codeword(idx):
        nonlocal covered_bits
        code.append(idx)
        for j in ball_list[idx]:
            cover_count[j] += 1
        covered_bits |= ball_bitsets[idx]

    def remove_codeword(idx):
        nonlocal covered_bits
        try:
            pos = code.index(idx)
        except ValueError:
            return
        code.pop(pos)
        for j in ball_list[idx]:
            cover_count[j] -= 1
        # Recompute covered_bits: a word is covered if cover_count[j] > 0
        # This is O(total_words) but only called during local search
        covered_bits = 0
        for j in range(total_words):
            if cover_count[j] > 0:
                covered_bits |= (1 << j)

    def all_covered():
        return covered_bits == (1 << total_words) - 1

    # Greedy Phase - use bitset for fast gain computation
    while not all_covered():
        best_idx = -1
        best_gain = -1
        for c in range(total_words):
            # Gain = number of bits in ball_bitsets[c] not in covered_bits
            gain = bin(ball_bitsets[c] & ~covered_bits).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = c
                if gain == 15:  # Max possible for q=3, n=7, r=1
                    break
        if best_gain == 0:
            break
        add_codeword(best_idx)
        
        if time.time() - start_time > time_limit * 0.15:
            break

    # Local Search Phase with bitset-accelerated operations
    def is_redundant(idx):
        # Check if removing idx would uncover any word
        # A word j in ball[idx] is uncovered if cover_count[j] == 1
        for j in ball_list[idx]:
            if cover_count[j] == 1:
                return False
        return True

    def try_remove(idx):
        """Try to remove idx if redundant. Returns True if removed."""
        if is_redundant(idx):
            remove_codeword(idx)
            return True
        return False

    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Try removals in random order for better exploration
        order = list(range(len(code)))
        random.shuffle(order)
        for i in order:
            if i < len(code):  # code may have changed
                if try_remove(code[i]):
                    improved = True
                    break
        
        if not improved and len(code) > 0:
            # Try a kick: remove a random codeword, then greedily fix
            kick_idx = random.choice(code)
            old_code = code.copy()
            old_cover_count = cover_count.copy()
            old_covered_bits = covered_bits
            
            remove_codeword(kick_idx)
            
            # Greedily add words to cover uncovered ones
            while not all_covered():
                best_idx = -1
                best_gain = -1
                for c in range(total_words):
                    if c in code:
                        continue
                    gain = bin(ball_bitsets[c] & ~covered_bits).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = c
                if best_gain == 0:
                    break
                add_codeword(best_idx)
                
                if time.time() - start_time > time_limit:
                    break
            
            if len(code) < len(old_code):
                improved = True
                # Keep new state (already updated)
            else:
                # Revert
                code = old_code
                cover_count = old_cover_count
                covered_bits = old_covered_bits

    # Final cleanup: try removals again with multiple passes
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        removed = False
        order = list(range(len(code)))
        random.shuffle(order)
        for i in order:
            if i < len(code):
                if try_remove(code[i]):
                    removed = True
        if not removed:
            break

    # Convert indices back to words
    return [idx_to_word(idx) for idx in code]
