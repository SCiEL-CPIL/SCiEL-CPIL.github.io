import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    global_deadline = start_time + 110  # 10s safety margin
    
    def solve_n(n, deadline):
        num_triples = 0
        triples = []
        triple_index = {}
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_index[(a,b,c)] = num_triples
                    triples.append((a,b,c))
                    num_triples += 1
                    
        # Precompute K4s and K5s
        k4s = []
        k4_to_triple_idxs = []
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            idxs = [triple_index[(a,b,c)], triple_index[(a,b,d)], triple_index[(a,c,d)], triple_index[(b,c,d)]]
            k4s.append(quad)
            k4_to_triple_idxs.append(idxs)
            
        k5s = []
        k5_to_triple_idxs = []
        for quint in combinations(range(n), 5):
            idxs = []
            for tri in combinations(quint, 3):
                idxs.append(triple_index[tri])
            k5s.append(quint)
            k5_to_triple_idxs.append(idxs)
            
        # Map triple index to list of (k4_index) and (k5_index)
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k4_to_triple_idxs):
            for idx in idxs:
                triple_to_k4s[idx].append(i)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k5_to_triple_idxs):
            for idx in idxs:
                triple_to_k5s[idx].append(i)
                
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        k4_red_counts = [0] * len(k4s)
        k5_blue_counts = [0] * len(k5s)
        
        # Initialize counts
        for i, idxs in enumerate(k4_to_triple_idxs):
            count = 0
            for idx in idxs:
                if colors[idx] == 0:
                    count += 1
            k4_red_counts[i] = count
            
        for i, idxs in enumerate(k5_to_triple_idxs):
            count = 0
            for idx in idxs:
                if colors[idx] == 1:
                    count += 1
            k5_blue_counts[i] = count
            
        red_k4_set = set()
        blue_k5_set = set()
        
        for i in range(len(k4s)):
            if k4_red_counts[i] == 4:
                red_k4_set.add(i)
                
        for i in range(len(k5s)):
            if k5_blue_counts[i] == 10:
                blue_k5_set.add(i)
                
        while time.time() < deadline:
            if not red_k4_set and not blue_k5_set:
                return n, colors[:]
                
            # Pick a random conflict
            if red_k4_set:
                # Convert to list to pick random. 
                # Optimization: if set is large, this is slow. 
                # But for n=34, it's likely manageable.
                k4_idx = random.choice(list(red_k4_set))
                tri_idx = random.choice(k4_to_triple_idxs[k4_idx])
            else:
                k5_idx = random.choice(list(blue_k5_set))
                tri_idx = random.choice(k5_to_triple_idxs[k5_idx])
                
            old_color = colors[tri_idx]
            new_color = 1 - old_color
            colors[tri_idx] = new_color
            
            # Update K4s containing tri_idx
            for k4_i in triple_to_k4s[tri_idx]:
                old_count = k4_red_counts[k4_i]
                if old_color == 0:
                    # Red -> Blue, count decreases
                    k4_red_counts[k4_i] = old_count - 1
                    new_count = old_count - 1
                    if new_count == 4:
                        red_k4_set.add(k4_i)
                    elif old_count == 4:
                        red_k4_set.discard(k4_i)
                else:
                    # Blue -> Red, count increases
                    k4_red_counts[k4_i] = old_count + 1
                    new_count = old_count + 1
                    if new_count == 4:
                        red_k4_set.add(k4_i)
                    elif old_count == 4:
                        red_k4_set.discard(k4_i)
                        
            # Update K5s containing tri_idx
            for k5_i in triple_to_k5s[tri_idx]:
                old_count = k5_blue_counts[k5_i]
                if old_color == 1:
                    # Blue -> Red, count decreases
                    k5_blue_counts[k5_i] = old_count - 1
                    new_count = old_count - 1
                    if new_count == 10:
                        blue_k5_set.add(k5_i)
                    elif old_count == 10:
                        blue_k5_set.discard(k5_i)
                else:
                    # Red -> Blue, count increases
                    k5_blue_counts[k5_i] = old_count + 1
                    new_count = old_count + 1
                    if new_count == 10:
                        blue_k5_set.add(k5_i)
                    elif old_count == 10:
                        blue_k5_set.discard(k5_i)
                        
        return -1, None

    # Try n=34 with most of the time
    n, colors = solve_n(34, global_deadline - 20)
    if n != -1:
        return {'n': n, 'colors': colors}
        
    # Fallback
    remaining_time = global_deadline - time.time()
    if remaining_time > 5:
        n, colors = solve_n(33, global_deadline - 5)
        if n != -1:
            return {'n': n, 'colors': colors}
            
    # Last resort
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
