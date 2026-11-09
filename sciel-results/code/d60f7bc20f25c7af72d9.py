import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def run_sa(n, time_limit):
        # Precompute triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple to index
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Initial random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Precompute partners for fast delta calculation
        red_partners = [[] for _ in range(num_triples)]
        blue_partners = [[] for _ in range(num_triples)]
        
        # To speed up, we can precompute the sorted lists of indices for the "other" triples
        # For red_partners[i], we store list of lists of 3 indices.
        # For blue_partners[i], we store list of lists of 7 indices.
        
        for i, t in enumerate(triples):
            a, b, c = t
            # Red K4
            for d in range(n):
                if d in (a, b, c): continue
                verts = sorted([a, b, c, d])
                if len(verts) != 4: continue
                t1 = tuple(sorted((a, b, d)))
                t2 = tuple(sorted((a, c, d)))
                t3 = tuple(sorted((b, c, d)))
                if t1 in triple_index and t2 in triple_index and t3 in triple_index:
                    red_partners[i].append([triple_index[t1], triple_index[t2], triple_index[t3]])
            
            # Blue K5
            for d in range(n):
                if d in (a, b, c): continue
                for e in range(d+1, n):
                    if e in (a, b, c) or e == d: continue
                    verts = sorted([a, b, c, d, e])
                    if len(verts) != 5: continue
                    v = list(verts)
                    all_triples = list(combinations(v, 3))
                    other_triples = [tt for tt in all_triples if tt != t]
                    if len(other_triples) == 7:
                        idxs = [triple_index[tt] for tt in other_triples]
                        blue_partners[i].append(idxs)
        
        # Initial violation count
        red_viol = 0
        blue_viol = 0
        for i in range(num_triples):
            if colors[i] == 0: # Red
                for idxs in red_partners[i]:
                    if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0:
                        red_viol += 1
            else: # Blue
                for idxs in blue_partners[i]:
                    if all(colors[j] == 1 for j in idxs):
                        blue_viol += 1
        
        total_viol = red_viol + blue_viol
        
        if total_viol == 0:
            return colors
        
        T = 1.0
        cooling = 0.9999
        min_T = 1e-4
        
        # Local variables for speed
        colors_local = colors
        red_partners_local = red_partners
        blue_partners_local = blue_partners
        num_triples_local = num_triples
        
        while time.time() < time_limit:
            i = random.randint(0, num_triples_local - 1)
            old_color = colors_local[i]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            if old_color == 0: # Red -> Blue
                # Destroy red K4s
                for idxs in red_partners_local[i]:
                    if colors_local[idxs[0]] == 0 and colors_local[idxs[1]] == 0 and colors_local[idxs[2]] == 0:
                        delta_red -= 1
                # Create blue K5s
                for idxs in blue_partners_local[i]:
                    if all(colors_local[j] == 1 for j in idxs):
                        delta_blue += 1
            else: # Blue -> Red
                # Create red K4s
                for idxs in red_partners_local[i]:
                    if colors_local[idxs[0]] == 0 and colors_local[idxs[1]] == 0 and colors_local[idxs[2]] == 0:
                        delta_red += 1
                # Destroy blue K5s
                for idxs in blue_partners_local[i]:
                    if all(colors_local[j] == 1 for j in idxs):
                        delta_blue -= 1
            
            delta = delta_red + delta_blue
            
            if delta <= 0:
                colors_local[i] = new_color
                total_viol += delta
            else:
                if random.random() < pow(2.718281828, -delta / T):
                    colors_local[i] = new_color
                    total_viol += delta
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if total_viol == 0:
                return colors_local
                
        return None

    # Try n=34
    t_end = time.time() + 120
    res = run_sa(34, t_end)
    if res:
        return {'n': 34, 'colors': res}
    
    # If 34 failed, try 33?
    if time.time() < deadline:
        t_end = min(time.time() + 10, deadline)
        res = run_sa(33, t_end)
        if res:
            return {'n': 33, 'colors': res}
            
    # Fallback to valid small n
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
