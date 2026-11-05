import itertools
import random
import time

def construct(q, n, radius):
        # 1. Initial Solution: Direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Precompute neighbors for efficiency?
        # For a word w, its closed ball is w itself + all words at distance 1.
        # Total words = 3**n = 729.
        
        # Helper to check if a set of codewords covers the space
        def is_covering(code_set):
            covered = [False] * (q ** n)
            for c in code_set:
                idx = int(''.join(map(str, c)), 3) # This is slow. Use base-3 conversion properly.
                # Better: represent words as integers 0..3^n-1
                pass

        # Let's represent words as integers for speed.
        # w = w_0 * 3^(n-1) + ... + w_{n-1}
        
        def word_to_int(w):
            val = 0
            for i in range(n):
                val = val * 3 + w[i]
            return val

        def int_to_word(val):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = val % 3
                val //= 3
            return w

        # Precompute all words as ints and their closed balls
        all_words = list(range(q ** n))
        balls = []
        for w_int in all_words:
            w = int_to_word(w_int)
            ball = [w_int]
            for i in range(n):
                orig = w[i]
                for new_val in range(q):
                    if new_val == orig: continue
                    w[i] = new_val
                    ball.append(word_to_int(w))
                w[i] = orig
            balls.append(ball)
            
        # Convert initial code to ints
        code_ints = set(word_to_int(c) for c in initial_code)
        
        # Helper to check coverage of a set of ints
        def is_covering_ints(code_set):
            covered = [False] * len(all_words)
            for c in code_set:
                for w in balls[c]:
                    covered[w] = True
            return all(covered)
            
        # Local Search
        current_code = set(code_ints)
        
        def try_remove_redundant():
            removed = False
            # Randomize order to avoid bias
            candidates = list(current_code)
            random.shuffle(candidates)
            for c in candidates:
                if c not in current_code: continue
                test_set = current_code - {c}
                if is_covering_ints(test_set):
                    current_code.remove(c)
                    removed = True
            return removed

        def try_swap():
            swapped = False
            candidates = list(current_code)
            random.shuffle(candidates)
            for c in candidates:
                if c not in current_code: continue
                # Pick a random replacement
                # To be efficient, we can pick random words not in current_code
                # But checking validity is the bottleneck.
                # Let's try a few random swaps per codeword
                for _ in range(10):
                    new_c = random.choice(all_words)
                    if new_c in current_code: continue
                    test_set = (current_code - {c}) | {new_c}
                    if is_covering_ints(test_set):
                        current_code.remove(c)
                        current_code.add(new_c)
                        swapped = True
                        break
            return swapped

        start_time = time.time()
        deadline = 140.0 # Safety margin
        
        # Initial cleanup
        while try_remove_redundant():
            if time.time() - start_time > deadline:
                break

        # Iterative improvement
        while time.time() - start_time < deadline:
            # Try to remove redundant
            if try_remove_redundant():
                continue
            
            # Try to swap to escape local minima
            if try_swap():
                # After a swap, try to remove redundant again immediately
                # This is handled by the loop structure if we just continue, 
                # but try_remove_redundant is at the top of the loop.
                pass
            else:
                # If no swap works, we are stuck. 
                # We can try to restart with a slightly perturbed initial solution?
                # Or just break if we are stuck for a long time?
                # Given the small space, getting stuck is possible.
                # Let's try a few more random swaps with more attempts per codeword
                # or just break to avoid wasting time.
                # Actually, if no swap works, we might be at a local minimum.
                # We can try to add a random codeword, then remove redundant?
                # No, adding increases size. We want to reduce.
                # Let's try a "kick": remove a random codeword, add a random one, 
                # even if it's not a valid swap? No, must maintain validity.
                
                # If stuck, break.
                break

        # Convert back to lists of ints
        result = [int_to_word(c) for c in current_code]
        return result
