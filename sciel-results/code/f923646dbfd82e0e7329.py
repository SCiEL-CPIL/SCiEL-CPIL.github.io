import time
import random
import math

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE = 58.0  # Safety margin before 60s kill

def construct():
    start_time = time.time()
    
    def build_next(trans):
        """Precompute subset transitions for a single letter."""
        size = 1 << N
        next_sub = [0] * size
        # next_sub[0] is 0
        for mask in range(1, size):
            # Use lowest set bit trick
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            rest = mask ^ lsb
            next_sub[mask] = next_sub[rest] | (1 << trans[idx])
        return next_sub

    def compute_threshold(next_a, next_b):
        """BFS on subset automaton to find shortest reset word length."""
        size = 1 << N
        dist = [-1] * size
        dist[FULL_MASK] = 0
        queue = [FULL_MASK]
        head = 0
        
        while head < len(queue):
            s = queue[head]
            head += 1
            d = dist[s]
            
            # Try 'a'
            n1 = next_a[s]
            if n1 == 1: # Single state 0
                return d + 1
            if dist[n1] == -1:
                dist[n1] = d + 1
                queue.append(n1)
                
            # Try 'b'
            n2 = next_b[s]
            if n2 == 1:
                return d + 1
            if dist[n2] == -1:
                dist[n2] = d + 1
                queue.append(n2)
                
        return 0 # Not synchronizing

    def evaluate(fa, fb):
        next_a = build_next(fa)
        next_b = build_next(fb)
        return compute_threshold(next_a, next_b)

    rng = random.Random(20240517)
    
    best_score = 0
    best_fa = None
    best_fb = None
    
    def random_cycle():
        """Generate a random single cycle permutation of length N."""
        perm = list(range(N))
        rng.shuffle(perm)
        return perm

    def generate_pinwheel_b():
        """
        Standard pinwheel structure for 'b':
        b(i) = i - 1 mod N, but b(0) = 0.
        This creates a 'comb' that contracts the cycle.
        """
        fb = [0] * N
        for i in range(N):
            fb[i] = (i - 1) % N
        fb[0] = 0
        return fb

    def generate_shift_break_b():
        """
        Shift with a break:
        b(i) = i + 1 mod N, but b(0) = 0.
        """
        fb = [0] * N
        for i in range(N):
            fb[i] = (i + 1) % N
        fb[0] = 0
        return fb

    def generate_random_b():
        """Completely random b."""
        return [rng.randrange(N) for _ in range(N)]

    def simulated_annealing(fa, fb, max_iter=20000):
        nonlocal best_score, best_fa, best_fb
        
        curr_fa = fa[:]
        curr_fb = fb[:]
        curr_score = evaluate(curr_fa, curr_fb)
        
        if curr_score > best_score:
            best_score = curr_score
            best_fa = curr_fa[:]
            best_fb = curr_fb[:]
            
        # Annealing parameters
        # Start with high temperature to explore, cool down to exploit
        T = 10.0
        alpha = 0.999  # Cooling rate
        min_T = 0.01
        
        next_a = build_next(curr_fa)
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE:
                return
            
            # Decide mutation strategy
            # We want to keep 'a' as a permutation (cycle) as much as possible
            # because cycle+defect structures are known to be hard.
            # However, allowing 'a' to break the cycle might help in some cases,
            # but usually, a single cycle 'a' is the backbone of high-threshold automata.
            
            r = rng.random()
            
            if r < 0.7:
                # Mutate b: change destination of one state
                old_b = curr_fb[:]
                state = rng.randrange(N)
                new_dest = rng.randrange(N)
                curr_fb[state] = new_dest
                
                next_b = build_next(curr_fb)
                new_score = compute_threshold(next_a, next_b)
                
                delta = new_score - curr_score
                
                # Accept if better, or with probability exp(-delta/T) if worse
                if delta > 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    curr_fb = old_b
                    
            elif r < 0.9:
                # Mutate b: swap two destinations
                old_b = curr_fb[:]
                s1, s2 = rng.sample(range(N), 2)
                curr_fb[s1], curr_fb[s2] = curr_fb[s2], curr_fb[s1]
                
                next_b = build_next(curr_fb)
                new_score = compute_threshold(next_a, next_b)
                
                delta = new_score - curr_score
                
                if delta > 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    curr_fb = old_b

            else:
                # Mutate a: swap two destinations to keep it a permutation
                old_fa = curr_fa[:]
                s1, s2 = rng.sample(range(N), 2)
                curr_fa[s1], curr_fa[s2] = curr_fa[s2], curr_fa[s1]
                
                next_a = build_next(curr_fa)
                next_b = build_next(curr_fb)
                new_score = compute_threshold(next_a, next_b)
                
                delta = new_score - curr_score
                
                if delta > 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    curr_fa = old_fa
                    # Rebuild next_a if reverted
                    next_a = build_next(curr_fa)
            
            # Cool down
            T *= alpha
            if T < min_T:
                T = min_T

    # Main Loop: Multiple restarts with different initial structures
    while time.time() - start_time < DEADLINE:
        # 1. Generate a random cycle for 'a'
        fa = random_cycle()
        
        # 2. Generate initial 'b' using high-potential templates
        # We weight the templates: Pinwheel is often the best starting point
        variant = rng.randrange(10)
        if variant < 4:
            fb = generate_pinwheel_b()
        elif variant < 7:
            fb = generate_shift_break_b()
        else:
            fb = generate_random_b()
            
        # Perturb b slightly to avoid exact symmetry if needed, 
        # but the templates are already asymmetric enough.
        
        # Run simulated annealing
        # We limit iterations per restart to allow multiple diverse starts
        simulated_annealing(fa, fb, max_iter=15000)
        
        # If we are close to deadline, do one final long climb on the best found so far
        if time.time() - start_time > DEADLINE - 10:
            if best_fa:
                simulated_annealing(best_fa, best_fb, max_iter=100000) # Run until deadline
            break

    if best_fa is None:
        # Fallback: return a known good automaton or random
        # Just return a random one, though it might not sync well
        return [rng.randrange(N) for _ in range(N)], [rng.randrange(N) for _ in range(N)]

    return best_fa, best_fb
