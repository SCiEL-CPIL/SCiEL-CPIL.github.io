import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    def evaluate(S):
        """Count distinct d>0 such that a, a+d, a+2d all in S for some a."""
        S_set = set(S)
        valid_diffs = set()
        for a in S:
            for b in S:
                if b > a:
                    d = b - a
                    c = a + 2 * d
                    if c in S_set:
                        valid_diffs.add(d)
        return len(valid_diffs)
    
    def evaluate_fast(S_set):
        """Evaluate using a set directly."""
        valid_diffs = set()
        for a in S_set:
            for b in S_set:
                if b > a:
                    d = b - a
                    c = a + 2 * d
                    if c in S_set:
                        valid_diffs.add(d)
        return len(valid_diffs)
    
    def make_initial_structured():
        """Create a structured initial set likely to have many 3-AP differences."""
        # Strategy: use elements spread across the range with some structure
        # Try: take a geometric-like spread, or use primes, or use a grid
        candidates = []
        
        # Option 1: Spread evenly
        s1 = [1 + i * 16 for i in range(32)]
        candidates.append(s1)
        
        # Option 2: Two interleaved APs
        s2 = []
        for i in range(16):
            s2.append(1 + i * 32)
            s2.append(17 + i * 32)
        candidates.append(s2)
        
        # Option 3: Use a mix of small and large gaps
        s3 = list(range(1, 17)) + [100 + i * 10 for i in range(16)]
        s3 = [x for x in s3 if 1 <= x <= 512][:32]
        if len(s3) == 32:
            candidates.append(s3)
        
        # Option 4: Random spread
        s4 = sorted(random.sample(range(1, 513), 32))
        candidates.append(s4)
        
        # Option 5: Clustered with some spread
        s5 = []
        base_positions = [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450]
        for bp in base_positions:
            s5.append(bp)
            if bp + 5 <= 512:
                s5.append(bp + 5)
        s5 = s5[:32]
        if len(s5) == 32:
            candidates.append(s5)
        
        # Pick the best
        best = max(candidates, key=lambda s: evaluate(s))
        return best
    
    # Generate multiple initial candidates
    initial_candidates = []
    for _ in range(3):
        initial_candidates.append(make_initial_structured())
    
    # Also try a greedy construction
    def greedy_construct():
        """Greedy: add elements that maximize new distinct differences."""
        S = []
        S_set = set()
        current_diffs = set()
        
        # Start with a small seed
        seed = [1, 2, 3, 100, 200, 300]
        for x in seed:
            if x not in S_set:
                S.append(x)
                S_set.add(x)
        
        # Update current diffs
        for a in S:
            for b in S:
                if b > a:
                    d = b - a
                    c = a + 2 * d
                    if c in S_set:
                        current_diffs.add(d)
        
        while len(S) < 32:
            best_val = None
            best_gain = -1
            # Sample a subset of candidates for speed
            candidates = random.sample(range(1, 513), min(200, 512 - len(S_set)))
            for v in candidates:
                if v in S_set:
                    continue
                new_S_set = S_set | {v}
                new_diffs = set()
                # Only check differences involving v
                for a in S_set:
                    # v as first or middle
                    if v > a:
                        d = v - a
                        c = v + d
                        if c in new_S_set:
                            new_diffs.add(d)
                    # a as first, v as middle or end
                    if a > v:
                        d = a - v
                        c = a + d
                        if c in new_S_set:
                            new_diffs.add(d)
                    # v as middle: need a and c such that a+v+c... 
                    # Actually: a, a+d, a+2d where a+2d=v => a = v-2d, a+d = v-d
                    # Or a+d=v => a=v-d, a+2d=v+d
                    for d in range(1, 256):
                        # v is the end: a = v-2d, mid = v-d
                        if v - 2*d >= 1 and (v-2*d) in new_S_set and (v-d) in new_S_set:
                            new_diffs.add(d)
                        # v is the middle: a = v-d, end = v+d
                        if v + d <= 512 and (v-d) in new_S_set and (v+d) in new_S_set:
                            new_diffs.add(d)
                
                gain = len(new_diffs - current_diffs)
                if gain > best_gain:
                    best_gain = gain
                    best_val = v
            
            if best_val is None:
                # Pick random
                while True:
                    v = random.randint(1, 512)
                    if v not in S_set:
                        break
                best_val = v
            
            S.append(best_val)
            S_set.add(best_val)
            
            # Update current_diffs
            current_diffs = set()
            for a in S:
                for b in S:
                    if b > a:
                        d = b - a
                        c = a + 2 * d
                        if c in S_set:
                            current_diffs.add(d)
        
        return S
    
    initial_candidates.append(greedy_construct())
    
    # Find best initial
    best_S = max(initial_candidates, key=evaluate)
    best_score = evaluate(best_S)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    current_S = best_S[:]
    current_score = best_score
    current_set = set(current_S)
    
    no_improve = 0
    
    while time.time() - start_time < timeout:
        if T < 0.1:
            break
            
        move_type = random.random()
        accepted = False
        
        if move_type < 0.6:
            # Replace one element
            i = random.randint(0, 31)
            old_val = current_S[i]
            new_val = random.randint(1, 512)
            if new_val in current_set:
                continue
            
            current_S[i] = new_val
            current_set.remove(old_val)
            current_set.add(new_val)
            new_score = evaluate_fast(current_set)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                accepted = True
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_S[i] = old_val
                current_set.remove(new_val)
                current_set.add(old_val)
                no_improve += 1
                
        elif move_type < 0.85:
            # Swap two elements
            i, j = random.sample(range(32), 2)
            old_i, old_j = current_S[i], current_S[j]
            current_S[i], current_S[j] = old_j, old_i
            new_score = evaluate(current_S)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                accepted = True
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_S[i], current_S[j] = old_i, old_j
                no_improve += 1
                
        else:
            # Replace two elements
            i, j = random.sample(range(32), 2)
            old_i, old_j = current_S[i], current_S[j]
            new_i = random.randint(1, 512)
            new_j = random.randint(1, 512)
            if new_i in current_set or new_j in current_set or new_i == new_j:
                continue
            
            current_S[i] = new_i
            current_S[j] = new_j
            current_set.discard(old_i)
            current_set.discard(old_j)
            current_set.add(new_i)
            current_set.add(new_j)
            new_score = evaluate_fast(current_set)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                accepted = True
                if new_score > best_score:
                    best_score = new_score
                    best_S = current_S[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_S[i] = old_i
                current_S[j] = old_j
                current_set.discard(new_i)
                current_set.discard(new_j)
                current_set.add(old_i)
                current_set.add(old_j)
                no_improve += 1
        
        T *= cooling
        
        # Restart if stuck
        if no_improve > 5000:
            # Reset to best and restart with higher temperature
            current_S = best_S[:]
            current_score = best_score
            current_set = set(current_S)
            T = 5.0
            no_improve = 0
    
    return best_S
