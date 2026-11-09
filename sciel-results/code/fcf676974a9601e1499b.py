import numpy as np
import random
import time
import itertools

def construct():
    target_ns = [30, 29, 28]
    
    for n in target_ns:
        start_time = time.time()
        num_edges = n * (n - 1) // 2
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        
        # Precompute sample instances for fast fitness
        # 2000 samples should be enough to guide the search
        sample_instances = []
        for _ in range(2000):
            verts = random.sample(range(n), 6)
            random.shuffle(verts)
            pairs = [(verts[0], verts[1]), (verts[2], verts[3]), (verts[4], verts[5])]
            edges = []
            for p1 in pairs:
                for p2 in pairs:
                    if p1 != p2:
                        u1, u2 = p1
                        v1, v2 = p2
                        for u in (u1, u2):
                            for v in (v1, v2):
                                if u < v:
                                    edges.append(edge_idx[(u, v)])
                                else:
                                    edges.append(edge_idx[(v, u)])
            sample_instances.append(edges)
        sample_array = np.array(sample_instances)
        
        def sample_fitness(colors):
            c_inst = colors[sample_array]
            is_mono = (np.all(c_inst == 0, axis=1) | np.all(c_inst == 1, axis=1))
            return is_mono.sum()

        # Precompute all instances for verification (lazy)
        all_instances = None
        def get_all_instances():
            nonlocal all_instances
            if all_instances is None:
                all_instances = []
                for comb in itertools.combinations(range(n), 6):
                    # 15 partitions
                    # We generate all 15 matchings
                    # Matchings of 6 elements:
                    # We can generate by fixing (0,1) and permuting the rest?
                    # There are 15 perfect matchings.
                    # Let's just iterate all 15 explicitly using a helper.
                    # Helper to get all 15 matchings of 0..5
                    # We can hardcode the 15 sets of 3 pairs.
                    # Pairs are (a,b), (c,d), (e,f)
                    # Let's generate them:
                    # Fix 0. Pair with 1, 2, 3, 4, 5.
                    # If 0-1, remaining 2,3,4,5. Matchings: (2,3)(4,5), (2,4)(3,5), (2,5)(3,4). (3 ways)
                    # If 0-2, remaining 1,3,4,5. Matchings: (1,3)(4,5), (1,4)(3,5), (1,5)(3,4). (3 ways)
                    # If 0-3, remaining 1,2,4,5. Matchings: (1,2)(4,5), (1,4)(2,5), (1,5)(2,4). (3 ways)
                    # If 0-4, remaining 1,2,3,5. Matchings: (1,2)(3,5), (1,3)(2,5), (1,5)(2,3). (3 ways)
                    # If 0-5, remaining 1,2,3,4. Matchings: (1,2)(3,4), (1,3)(2,4), (1,4)(2,3). (3 ways)
                    # Total 15.
                    
                    matchings = [
                        [(0,1), (2,3), (4,5)],
                        [(0,1), (2,4), (3,5)],
                        [(0,1), (2,5), (3,4)],
                        [(0,2), (1,3), (4,5)],
                        [(0,2), (1,4), (3,5)],
                        [(0,2), (1,5), (3,4)],
                        [(0,3), (1,2), (4,5)],
                        [(0,3), (1,4), (2,5)],
                        [(0,3), (1,5), (2,4)],
                        [(0,4), (1,2), (3,5)],
                        [(0,4), (1,3), (2,5)],
                        [(0,4), (1,5), (2,3)],
                        [(0,5), (1,2), (3,4)],
                        [(0,5), (1,3), (2,4)],
                        [(0,5), (1,4), (2,3)]
                    ]
                    
                    for match in matchings:
                        # Map local indices to comb
                        pairs = [(comb[a], comb[b]) for a, b in match]
                        edges = []
                        for p1 in pairs:
                            for p2 in pairs:
                                if p1 != p2:
                                    u1, u2 = p1
                                    v1, v2 = p2
                                    for u in (u1, u2):
                                        for v in (v1, v2):
                                            if u < v:
                                                edges.append(edge_idx[(u, v)])
                                            else:
                                                edges.append(edge_idx[(v, u)])
                        all_instances.append(edges)
                all_instances = np.array(all_instances)
            return all_instances

        for restart in range(10):
            if time.time() - start_time > 100:
                break
                
            colors = np.random.randint(0, 2, size=num_edges)
            best_fit = sample_fitness(colors)
            
            for step in range(3000):
                e = random.randint(0, num_edges - 1)
                old_c = colors[e]
                colors[e] = 1 - old_c
                new_fit = sample_fitness(colors)
                
                if new_fit < best_fit:
                    best_fit = new_fit
                    if best_fit == 0:
                        # Verify
                        all_inst = get_all_instances()
                        c_all = colors[all_inst]
                        is_mono = (np.all(c_all == 0, axis=1) | np.all(c_all == 1, axis=1))
                        if is_mono.sum() == 0:
                            return {'n': n, 'colors': colors.tolist()}
                        # If not 0, continue searching
                else:
                    colors[e] = old_c

    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
