import random
import time
import numpy as np

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    # Precompute all 4-APs (0-indexed)
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)
    
    num_aps = len(aps)
    
    def count_violations(coloring):
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations
    
    def get_violating_aps(coloring):
        violating = []
        for ap_id, ap in enumerate(aps):
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violating.append(ap_id)
        return violating
    
    def greedy_init(k, seed=None):
        """Structured initialization using modular patterns."""
        if seed is None:
            seed = random.randint(0, 1000000)
        rng = random.Random(seed)
        
        # Try different structural patterns
        patterns = []
        
        # Pattern 1: Based on position mod k
        patterns.append([i % k for i in range(N)])
        
        # Pattern 2: Based on position mod (k+1)
        patterns.append([i % (k + 1) for i in range(N)])
        
        # Pattern 3: Alternating blocks
        patterns.append([(i // k) % k for i in range(N)])
        
        # Pattern 4: Random with structure
        base = [i % k for i in range(N)]
        for _ in range(10):
            idx = rng.randint(0, N - 1)
            base[idx] = rng.randint(0, k - 1)
        patterns.append(base[:])
        
        # Pattern 5: Fibonacci-like
        fib = [0, 1]
        while len(fib) < N:
            fib.append((fib[-1] + fib[-2]) % k)
        patterns.append(fib[:N])
        
        # Choose the pattern with fewest violations
        best_pattern = None
        best_violations = float('inf')
        for pattern in patterns:
            v = count_violations(pattern)
            if v < best_violations:
                best_violations = v
                best_pattern = pattern[:]
        
        return best_pattern, best_violations
    
    def local_search(coloring, k, max_time=5.0):
        """Enhanced local search with simulated annealing."""
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        T = 3.0
        cooling = 0.9995
        steps = 0
        max_steps = 50000
        local_deadline = time.time() + max_time
        
        # Track which positions are in violating APs for prioritization
        violating_positions = set()
        if current_violations > 0:
            for ap_id in get_violating_aps(coloring):
                ap = aps[ap_id]
                violating_positions.update(aps[ap_id])
        
        while current_violations > 0 and steps < max_steps and time.time() < local_deadline:
            steps += 1
            
            # Prioritize positions in violating APs
            if violating_positions and random.random() < 0.7:
                idx = random.choice(list(violating_positions))
            else:
                idx = random.randint(0, N - 1)
            
            old_color = coloring[idx]
            new_color = random.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations efficiently
            delta_violations = 0
            new_violating_positions = set()
            
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = coloring[ap[0]]
                c1 = coloring[ap[1]]
                c2 = coloring[ap[2]]
                c3 = coloring[ap[3]]
                
                old_set_size = len({c0, c1, c2, c3})
                
                new_c0 = new_color if ap[0] == idx else c0
                new_c1 = new_color if ap[1] == idx else c1
                new_c2 = new_color if ap[2] == idx else c2
                new_c3 = new_color if ap[3] == idx else c3
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
                    new_violating_positions.update(ap)
            
            if delta_violations < 0:
                # Always accept improvements
                coloring[idx] = new_color
                current_violations += delta_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
                
                # Update violating positions
                if current_violations == 0:
                    violating_positions = set()
                else:
                    violating_positions = set()
                    for ap_id in get_violating_aps(coloring):
                        violating_positions.update(aps[ap_id])
            else:
                # Accept worse solutions with some probability
                if delta_violations == 0:
                    accept = random.random() < 0.6
                else:
                    if T > 0.01:
                        accept = random.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
                    violating_positions = set()
                    for ap_id in get_violating_aps(coloring):
                        violating_positions.update(aps[ap_id])
            
            T *= cooling
        
        return best_coloring, best_violations
    
    # Try to find a valid coloring with k colors, starting from k=3
    for k in range(3, 129):
        if time.time() > deadline - 15.0:
            break
        
        # Multiple restarts for small k
        num_restarts = 5 if k <= 5 else 3
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Different initialization strategies
            if restart == 0:
                coloring, initial_violations = greedy_init(k)
            elif restart == 1:
                coloring, initial_violations = greedy_init(k, seed=restart * 12345)
            elif restart == 2:
                coloring = [random.randint(0, k - 1) for _ in range(N)]
                initial_violations = count_violations(coloring)
            elif restart == 3:
                # Structured random
                coloring = [(i * 7 + restart * 13) % k for i in range(N)]
                initial_violations = count_violations(coloring)
            else:
                # Another pattern
                coloring = [(i * 11 + restart * 23) % k for i in range(N)]
                initial_violations = count_violations(coloring)
            
            if initial_violations == 0:
                return coloring
            
            # Run local search
            best_coloring, best_violations = local_search(coloring, k, max_time=3.0)
            
            if best_violations == 0:
                return best_coloring
    
    # Fallback: Use N distinct colors. 
    # This guarantees validity because every 4-AP has 4 distinct positions, 
    # thus 4 distinct colors (since all labels are unique).
    return list(range(N))
