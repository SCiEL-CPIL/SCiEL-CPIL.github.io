import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    N = q ** n
    
    def int_to_word(idx):
        return [idx // (q ** (n - i - 1)) % q for i in range(n)]
    
    def word_to_int(w):
        res = 0
        for i in range(n):
            res = res * q + w[i]
        return res

    # Precompute balls for all words
    # balls[idx] is a list of indices of words within distance <= radius
    balls = []
    for idx in range(N):
        w = int_to_word(idx)
        ball = [idx]
        if radius >= 1:
            for i in range(n):
                for delta in range(1, q):
                    new_w = w.copy()
                    new_w[i] = (new_w[i] + delta) % q
                    ball.append(word_to_int(new_w))
        balls.append(ball)
        
    # Precompute ball sizes (constant for all words in this metric)
    # But we need the sets.
    
    def generate_tetracode_product():
        """
        Generate the 81 codewords from Tetracode x {0,1,2}^2.
        Tetracode is a (4, 9, 4) ternary code.
        We can define it explicitly or generate a covering code of length 4, radius 1.
        A simple covering code of length 4, radius 1 over {0,1,2} has size 9.
        One such code is the Tetracode.
        Let's hardcode the Tetracode words or generate a greedy one for length 4.
        """
        # Greedy covering code for q=3, n=4, r=1
        # N=81
        # We know size is 9.
        # Let's just use a known set or generate it.
        # Actually, let's just use the greedy set cover for n=4 to get the 9 words, 
        # then extend.
        
        # For simplicity and speed, let's just use a fixed good set if possible, 
        # or run a very fast greedy for n=4.
        
        # Greedy for n=4
        N4 = 3**4
        balls4 = []
        for idx in range(N4):
            w = int_to_word(idx)
            # But int_to_word uses global n=6. Need local version.
            # Let's just hardcode the Tetracode or use a simple construction.
            # Tetracode generator matrix:
            # 1 0 0 0
            # 0 1 0 0
            # 0 0 1 0
            # 1 1 1 1  (Check: is this a covering code? d=4, yes. Size 9.)
            # Words:
            # 0000, 1000, 2000, 0100, 0200, 0010, 0020, 1111, 2222? 
            # Let's verify coverage of 0111.
            # d(0111, 0000)=2. d(0111, 1111)=1. Covered.
            # d(0111, 0100)=1. Covered.
            # Seems fine.
            
        # Let's generate the 9 words of Tetracode explicitly.
        # Basis vectors e1, e2, e3, e4=(1,1,1,1) and their sums?
        # No, it's a linear code.
        # C = { (a, b, c, a+b+c) | a,b,c in {0,1,2} }
        # Let's check if this covers {0,1,2}^4 with radius 1.
        # Take x=(x1, x2, x3, x4).
        # We need (a, b, c, a+b+c) such that d <= 1.
        # This implies 3 of the coordinates match.
        # If x1,x2,x3 match a,b,c, then x4 must be a+b+c.
        # If x1,x2,x4 match, then a=x1, b=x2, a+b+c=x4 => c = x4-x1-x2.
        # Then coord 3 is c. We need x3=c.
        # So if x3 = x4-x1-x2, it's covered by (x1, x2, x3, x4).
        # This logic suggests it's not a covering code in the simple sense?
        # Actually, the Tetracode IS a covering code of radius 1.
        # Let's just use a greedy set cover for n=4 to be safe. It's very fast.
        
        # Local greedy for n=4
        N4 = 3**4
        balls4 = []
        for idx in range(N4):
            w = [idx // 81 % 3, idx // 27 % 3, idx // 9 % 3, idx // 3 % 3]
            ball = [idx]
            for i in range(4):
                for delta in range(1, 3):
                    new_w = w.copy()
                    new_w[i] = (new_w[i] + delta) % 3
                    ball.append(new_w[0]*27 + new_w[1]*9 + new_w[2]*3 + new_w[3])
            balls4.append(ball)
            
        uncovered4 = [True] * N4
        code4 = []
        while True:
            remaining = sum(1 for u in range(N4) if uncovered4[u])
            if remaining == 0: break
            max_count = -1
            max_idx = -1
            for i in range(N4):
                cnt = sum(1 for u in balls4[i] if uncovered4[u])
                if cnt > max_count:
                    max_count = cnt
                    max_idx = i
            if max_count == 0: break
            code4.append(max_idx)
            for u in balls4[max_idx]:
                uncovered4[u] = False
                
        # Extend to n=6
        code6 = []
        for c4 in code4:
            # c4 is index in 0..80
            w4 = [c4 // 27 % 3, c4 // 9 % 3, c4 // 3 % 3, c4 % 3]
            for x5 in range(3):
                for x6 in range(3):
                    w6 = w4 + [x5, x6]
                    code6.append(word_to_int(w6))
        return code6

    initial_code = generate_tetracode_product()
    
    def local_search(code):
        code_set = set(code)
        cover_count = [0] * N
        for w in code:
            for u in balls[w]:
                cover_count[u] += 1
                
        # Total waste
        total_waste = sum(cover_count)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Remove redundant codewords
            removed = True
            while removed and time.time() < deadline:
                removed = False
                for i in range(len(code) - 1, -1, -1):
                    w = code[i]
                    can_remove = True
                    for u in balls[w]:
                        if cover_count[u] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        # Remove
                        for u in balls[w]:
                            cover_count[u] -= 1
                        code.pop(i)
                        code_set.remove(w)
                        total_waste -= len(balls[w])
                        improved = True
                        removed = True
                        break # Restart removal scan after change

            # 2. Swap to reduce waste
            if time.time() >= deadline: break
            
            best_swap = None
            best_delta_waste = 0
            
            # Iterate over all codewords in the code
            for w in code:
                # Iterate over all words not in the code
                # This is 729 * 73 ~ 50k iterations.
                # We can optimize by only considering words that cover "uncovered" parts? 
                # No, we want to reduce waste.
                # Delta waste = sum(cover_count[u] for u in balls[new_w]) - sum(cover_count[u] for u in balls[old_w])
                # We need to check if the swap is valid.
                # Valid if for all u in balls[old_w] \ balls[new_w], cover_count[u] > 1.
                
                old_ball = balls[w]
                old_sum = sum(cover_count[u] for u in old_ball)
                
                # To speed up, we can pre-compute the "critical" words for w
                # critical_w = [u for u in old_ball if cover_count[u] == 1]
                # If we swap w with new_w, all u in critical_w must be in balls[new_w].
                
                critical_w = [u for u in old_ball if cover_count[u] == 1]
                
                for new_w in range(N):
                    if new_w in code_set:
                        continue
                    
                    new_ball = balls[new_w]
                    
                    # Check validity
                    valid = True
                    for u in critical_w:
                        if u not in new_ball: # O(1) if new_ball is a set, but it's a list.
                            # We need faster lookup.
                            # Let's convert balls to sets for this search? 
                            # Or precompute a boolean matrix? 729x729 is 500k bools, fits in memory.
                            pass
                            
            # This nested loop with `u not in new_ball` is O(N^2 * ball_size).
            # 729 * 73 * 19 ~ 1 million ops. Fast enough in Python? Maybe.
            # But `u not in new_ball` is O(19).
            # Total ~ 10 million ops. Might be slow in Python if not optimized.
            
            # Optimization: Precompute `balls_set` for O(1) lookup.
            # balls_set[i] = set(balls[i])
            
            # Let's restructure to use sets.
            pass

    # Let's implement the optimized version with sets.
    
    balls_set = [set(b) for b in balls]
    
    def local_search_opt(code):
        code_set = set(code)
        cover_count = [0] * N
        for w in code:
            for u in balls[w]:
                cover_count[u] += 1
                
        total_waste = sum(cover_count)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Remove redundant codewords
            removed = True
            while removed and time.time() < deadline:
                removed = False
                for i in range(len(code) - 1, -1, -1):
                    w = code[i]
                    can_remove = True
                    for u in balls[w]:
                        if cover_count[u] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in balls[w]:
                            cover_count[u] -= 1
                        code.pop(i)
                        code_set.remove(w)
                        total_waste -= len(balls[w])
                        improved = True
                        removed = True
                        break

            if time.time() >= deadline: break

            # 2. Swap to reduce waste
            best_swap = None
            best_delta_waste = 0
            
            for w in code:
                old_ball = balls[w]
                old_ball_set = balls_set[w]
                old_sum = sum(cover_count[u] for u in old_ball)
                
                # Words that would become uncovered if w is removed
                # i.e., cover_count[u] == 1
                critical_w = [u for u in old_ball if cover_count[u] == 1]
                
                # For a swap to be valid, new_w must cover all critical_w
                # i.e., critical_w subset of balls_set[new_w]
                
                for new_w in range(N):
                    if new_w in code_set:
                        continue
                    
                    new_ball_set = balls_set[new_w]
                    
                    # Check if critical_w is subset of new_ball_set
                    # This is fast if critical_w is small.
                    # Usually critical_w is small.
                    valid = True
                    for u in critical_w:
                        if u not in new_ball_set:
                            valid = False
                            break
                            
                    if not valid:
                        continue
                        
                    new_ball = balls[new_w]
                    new_sum = sum(cover_count[u] for u in new_ball)
                    
                    delta_waste = new_sum - old_sum
                    if delta_waste < best_delta_waste:
                        best_delta_waste = delta_waste
                        best_swap = (w, new_w)
                        
            if best_swap:
                old_w, new_w = best_swap
                # Perform swap
                old_ball = balls[old_w]
                new_ball = balls[new_w]
                
                for u in old_ball:
                    cover_count[u] -= 1
                for u in new_ball:
                    cover_count[u] += 1
                    
                # Update code list and set
                idx = code.index(old_w)
                code[idx] = new_w
                code_set.remove(old_w)
                code_set.add(new_w)
                
                total_waste += best_delta_waste
                improved = True

        return code

    best_code = initial_code
    best_len = len(best_code)
    
    # Run local search
    code = local_search_opt(best_code)
    if len(code) < best_len:
        best_len = len(code)
        best_code = code

    # If we didn't reach 73, try random restarts?
    # The Tetracode product is a very strong start.
    # If stuck at 74, we might need to try other starts.
    # But 73 is the target.
    
    # Let's add a few random restarts if we have time
    while time.time() < deadline:
        # Generate a random code of size best_len + 5
        # Or just perturb the current best
        # Let's just try the local search again with a slightly perturbed start
        # Actually, if local search converged to a local min, we need a different start.
        
        # Try starting with a random set of 75 words
        random_code = random.sample(range(N), 75)
        # Ensure it's a valid cover? 
        # Random 75 words might not cover everything.
        # We need a valid cover to start.
        # We can greedily add words to a random set until it's a cover.
        
        # Greedy completion
        code_to_complete = random_code.copy()
        code_set_to_complete = set(code_to_complete)
        cover_count = [0] * N
        for w in code_to_complete:
            for u in balls[w]:
                cover_count[u] += 1
                
        # Find uncovered
        uncovered = [u for u in range(N) if cover_count[u] == 0]
        while uncovered:
            # Pick a random uncovered word, add a random neighbor
            u = random.choice(uncovered)
            # Pick a random word in ball(u)
            c = random.choice(balls[u])
            if c not in code_set_to_complete:
                code_to_complete.append(c)
                code_set_to_complete.add(c)
                # Update coverage
                for v in balls[c]:
                    cover_count[v] += 1
                # Recalculate uncovered (expensive, but N is small)
                uncovered = [u for u in range(N) if cover_count[u] == 0]
                
        # Now code_to_complete is a valid cover
        # Run local search
        completed_code = local_search_opt(code_to_complete)
        
        if len(completed_code) < best_len:
            best_len = len(completed_code)
            best_code = completed_code
            
        if best_len == 73:
            break

    return [int_to_word(w) for w in best_code]
