import time
import random
import math

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We should stop earlier to be safe.
    # Let's use 145s as a hard limit for the loop, and return best.
    deadline = start_time + 145.0
    
    def evaluate(S):
        """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in S."""
        # Use bitmask for fast membership check
        # Range 1..512
        mask = 0
        for x in S:
            mask |= (1 << x)
        
        valid_diffs = set()
        for i in range(32):
            a = S[i]
            for j in range(i + 1, 32):
                b = S[j]
                d = b - a
                c = 2 * b - a
                # Check if c is in S using bitmask
                # c must be in 1..512 to be in S
                if 1 <= c <= 512:
                    if (mask >> c) & 1:
                        valid_diffs.add(d)
        return len(valid_diffs)

    def random_init():
        return random.sample(range(1, 513), 32)

    def convex_init():
        # Heuristic: spread out points
        pts = [1]
        for i in range(1, 32):
            gap = 1 + i
            pts.append(pts[-1] + gap)
        # Scale to fit in 512?
        # Current max is 1 + sum(1..31) = 1 + 496 = 497. Fits.
        return pts

    # Initial candidates
    candidates = [
        list(range(1, 33)),       # Seed
        convex_init(),            
        random_init(),
        random_init(),
        random_init()
    ]
    
    # Evaluate initial candidates
    best_set = None
    best_score = -1
    for c in candidates:
        s = evaluate(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated Annealing
    # T starts high, cools down.
    # We want to maximize score.
    # Delta E = new_score - current_score.
    # If Delta E > 0, accept.
    # If Delta E < 0, accept with prob exp(Delta E / T).
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.05
    
    # We will run SA until time runs out.
    
    while time.time() < deadline:
        # Generate a neighbor
        # Move types:
        # 0: Swap two elements
        # 1: Move one element to a new random position
        # 2: "Nudge" one element by a small random delta (if valid)
        
        move_type = random.choice([0, 1, 2])
        new_set = current_set[:]
        
        if move_type == 0:
            i, j = random.sample(range(32), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
        elif move_type == 1:
            i = random.randrange(32)
            old_val = new_set[i]
            # Pick a new value not in new_set
            # To avoid O(N) check for distinctness in the loop, we can just pick random and check.
            # Since N=32 and range=512, collision is low (~6%).
            # But we must ensure distinctness.
            new_val = random.randint(1, 512)
            if new_val == old_val:
                continue
            # Check if new_val is in new_set (excluding i)
            # We can just remove old, add new, check len.
            # Or check membership.
            if new_val in new_set:
                # If collision, try a few times or skip?
                # Skipping is safer.
                # But this reduces effective moves.
                # Let's try a few times.
                found = False
                for _ in range(5):
                    new_val = random.randint(1, 512)
                    if new_val not in new_set:
                        found = True
                        break
                if not found:
                    continue
            new_set[i] = new_val
        else:
            # Nudge
            i = random.randrange(32)
            old_val = new_set[i]
            # Pick a small delta
            # Uniform distribution over small deltas
            delta = random.randint(-10, 10)
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 1 or new_val > 512:
                continue
            if new_val in new_set:
                continue
            new_set[i] = new_val
        
        # Check distinctness (should be guaranteed, but safe)
        if len(set(new_set)) != 32:
            continue
            
        new_score = evaluate(new_set)
        
        if new_score > current_score:
            current_score = new_score
            current_set = new_set
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            # Simulated Annealing step
            # If T is very small, exp(delta/T) will be 0 for negative delta.
            if T > min_T:
                delta_e = new_score - current_score
                # delta_e <= 0
                # prob = exp(delta_e / T)
                # To avoid underflow/overflow, just compute directly.
                # delta_e is integer, T is float.
                # If delta_e is very negative, exp is 0.
                # We can skip if delta_e < -50 * T
                if delta_e > -50 * T:
                    if random.random() < math.exp(delta_e / T):
                        current_score = new_score
                        current_set = new_set
                        # Do NOT update best_set if we accepted a worse move.
                        # But wait, if we accept a worse move, we might find a better one later.
                        # best_set tracks the global maximum found so far.
        
        # Cool down
        if T > min_T:
            T *= cooling
        else:
            # If T is too low, we are likely stuck.
            # Restart with a new random set or current best?
            # Let's restart with a new random set to explore new basins.
            # Or reheat?
            # Reheating is standard.
            T = 10.0 * random.uniform(0.5, 1.0)
            # Optionally, mix current_set with a random set?
            # Let's just keep current_set and reheat.
            # This allows local moves to happen again at higher T.

    return best_set
