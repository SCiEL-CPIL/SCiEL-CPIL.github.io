import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 100.0  # Safety margin (evaluator kills at 150s)
    
    def evaluate_fast(S):
        """Fast evaluation using set lookup."""
        S_set = set(S)
        valid_diffs = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(n):
                if i == j:
                    continue
                b = S[j]
                if a > b:
                    continue
                d = b - a
                c = a + 2 * d
                if c > 512:
                    continue
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def random_init():
        return random.sample(range(1, 513), 32)
    
    def convex_init():
        pts = [1]
        for i in range(1, 32):
            gap = 1 + i
            pts.append(pts[-1] + gap)
        return pts
    
    def grid_init():
        """Create a 4x8 grid structure that was found to have high score (~18)."""
        pts = []
        for row in range(4):
            for col in range(8):
                val = 1 + row * 128 + col * 16
                if val <= 512 and val not in pts:
                    pts.append(val)
        # Ensure we have exactly 32 distinct values
        while len(pts) < 32:
            v = random.randint(1, 512)
            if v not in pts:
                pts.append(v)
        return pts[:32]
    
    def structured_block_init():
        """Create set with blocks of consecutive numbers separated by gaps."""
        # Try different block structures
        candidates = []
        
        # 4 blocks of 8
        for offset in range(0, 50, 5):
            pts = []
            for block in range(4):
                start = 1 + block * 130 + offset
                for k in range(8):
                    if start + k <= 512:
                        pts.append(start + k)
            if len(pts) == 32 and len(set(pts)) == 32:
                candidates.append(pts)
        
        # 8 blocks of 4
        for offset in range(0, 50, 5):
            pts = []
            for block in range(8):
                start = 1 + block * 65 + offset
                for k in range(4):
                    if start + k <= 512:
                        pts.append(start + k)
            if len(pts) == 32 and len(set(pts)) == 32:
                candidates.append(pts)
                
        if candidates:
            return random.choice(candidates)
        return random_init()
    
    def geometric_init():
        """Geometric progression based."""
        pts = []
        base = 2
        for i in range(32):
            val = 1 + int(base ** i)
            if val > 512:
                val = 512 - (31 - i)
            if val not in pts and val >= 1 and val <= 512:
                pts.append(val)
            else:
                # Fill remaining
                while len(pts) < 32:
                    v = random.randint(1, 512)
                    if v not in pts:
                        pts.append(v)
        return pts[:32]
    
    # Generate diverse initial candidates
    candidates = [
        list(range(1, 33)),       # Consecutive
        convex_init(),            # Convex
        random_init(),
        random_init(),
        random_init(),
        grid_init(),              # Grid structure (high score ~18)
        grid_init(),
        structured_block_init(),
        structured_block_init(),
        geometric_init(),
    ]
    
    # Evaluate initial candidates
    best_set = None
    best_score = -1
    for c in candidates:
        if len(set(c)) != 32:
            continue
        s = evaluate_fast(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated Annealing with multiple move types
    T = 3.0
    T_min = 0.1
    cooling = 0.9995
    
    def generate_neighbor(S, move_type):
        """Generate a neighbor using different move types."""
        new_set = S[:]
        S_set = set(S)
        
        if move_type == 0:
            # Swap two elements
            i, j = random.sample(range(32), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
        elif move_type == 1:
            # Replace one element
            i = random.randrange(32)
            new_val = random.randint(1, 512)
            if new_val in S_set:
                return None
            new_set[i] = new_val
        elif move_type == 2:
            # Double replace
            i, j = random.sample(range(32), 2)
            new_val1 = random.randint(1, 512)
            new_val2 = random.randint(1, 512)
            if new_val1 in S_set or new_val2 in S_set:
                return None
            new_set[i] = new_val1
            new_set[j] = new_val2
        elif move_type == 3:
            # Local perturbation: shift one element by a small amount
            i = random.randrange(32)
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_val = S[i] + shift
            if new_val < 1 or new_val > 512 or new_val in S_set:
                return None
            new_set[i] = new_val
        elif move_type == 4:
            # Pair shift: shift two elements by small amounts
            i, j = random.sample(range(32), 2)
            shift1 = random.choice([-3, -2, -1, 1, 2, 3])
            shift2 = random.choice([-3, -2, -1, 1, 2, 3])
            new_val1 = S[i] + shift1
            new_val2 = S[j] + shift2
            if (new_val1 < 1 or new_val1 > 512 or new_val1 in S_set or 
                new_val2 < 1 or new_val2 > 512 or new_val2 in S_set):
                return None
            new_set[i] = new_val1
            new_set[j] = new_val2
        return new_set
    
    # Run multiple independent annealing runs to escape local optima
    num_restarts = 5
    steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if time.time() >= start_time + timeout - 5:
            break
        
        # For restarts > 0, start from a perturbed version of the best found so far
        if restart == 0:
            current_set = best_set[:]
            current_score = best_score
        else:
            current_set = best_set[:]
            # Apply random perturbations
            for _ in range(5):
                i = random.randrange(32)
                new_val = random.randint(1, 512)
                if new_val not in current_set:
                    current_set[i] = new_val
            current_score = evaluate_fast(current_set)
        
        # Reset temperature for this restart
        T = 2.0 + 0.5 * restart
        
        for step in range(steps_per_restart):
            if time.time() >= start_time + timeout - 2:
                break
            
            # Choose move type with weights
            r = random.random()
            if r < 0.25:
                move_type = 0  # Swap
            elif r < 0.5:
                move_type = 1  # Replace
            elif r < 0.65:
                move_type = 2  # Double replace
            elif r < 0.85:
                move_type = 3  # Local perturbation
            else:
                move_type = 4  # Pair shift
            
            new_set = generate_neighbor(current_set, move_type)
            if new_set is None or len(set(new_set)) != 32:
                continue
                
            new_score = evaluate_fast(new_set)
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(delta/T) if worse
            if delta > 0:
                accept = True
            else:
                # Use math.exp for scalar computation
                accept = random.random() < math.exp(delta / T)
                
            if accept:
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
            
            # Cool down
            T = max(T_min, T * cooling)
            
            # If stuck at minimum temperature, restart with higher T occasionally
            if T <= T_min and random.random() < 0.005:
                T = 1.5
    
    return best_set
