import random
import time

N = 64

class State:
    __slots__ = ('points', 'points_set', 'pair_dists', 'size')
    
    def __init__(self, points, pair_dists):
        self.points = points
        self.points_set = set(points)
        self.pair_dists = pair_dists
        self.size = len(points)

def try_add(state, p):
    px, py = p
    d_sq_list = []
    for a in state.points:
        ax, ay = a
        d_sq = (px - ax) ** 2 + (py - ay) ** 2
        d_sq_list.append(d_sq)
        
    # Check for duplicates in new distances
    # Using a set for d_sq_list is O(K)
    if len(d_sq_list) != len(set(d_sq_list)):
        return None
        
    # Check against existing pair_dists
    for d_sq in d_sq_list:
        if d_sq in state.pair_dists:
            return None
            
    # Valid
    new_points = state.points + [p]
    new_pair_dists = state.pair_dists.union(d_sq_list)
    return State(new_points, new_pair_dists)

def quick_greedy(seed=None):
    if seed is not None:
        random.seed(seed)
    all_points = [(x, y) for y in range(N) for x in range(N)]
    random.shuffle(all_points)
    
    selected = []
    pair_dists = set()
    for p in all_points:
        px, py = p
        d_sq_list = []
        valid = True
        for a in selected:
            ax, ay = a
            d_sq = (px - ax) ** 2 + (py - ay) ** 2
            d_sq_list.append(d_sq)
        
        if len(d_sq_list) != len(set(d_sq_list)):
            continue
            
        for d_sq in d_sq_list:
            if d_sq in pair_dists:
                valid = False
                break
                
        if not valid:
            continue
            
        selected.append(p)
        pair_dists.update(d_sq_list)
        
    return State(selected, pair_dists)

def construct():
    start_time = time.time()
    time_limit = 280
    
    # Beam parameters
    W = 50
    C = 100
    L = 1000 # Let time limit control actual iterations
    
    # Initialize beam with a few good greedy solutions
    beam = []
    for i in range(5):
        beam.append(quick_greedy(seed=i))
        
    best_state = max(beam, key=lambda s: s.size)
    
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    steps = 0
    while (time.time() - start_time) < time_limit:
        steps += 1
        new_beam_candidates = []
        
        for state in beam:
            # Sample C random points
            for _ in range(C):
                p = random.choice(all_points)
                if p in state.points_set:
                    continue
                
                new_state = try_add(state, p)
                if new_state is not None:
                    new_beam_candidates.append(new_state)
                    
        if not new_beam_candidates:
            # Try to inject fresh greedy solutions if stuck
            fresh = quick_greedy()
            if fresh.size > best_state.size:
                best_state = fresh
            beam.append(fresh)
            continue
            
        # Sort by size
        new_beam_candidates.sort(key=lambda s: s.size, reverse=True)
        
        # Keep top W
        beam = new_beam_candidates[:W]
        
        # Update best
        if beam[0].size > best_state.size:
            best_state = beam[0]
            
        # Periodic injection of diversity
        if steps % 50 == 0:
            fresh = quick_greedy()
            beam.append(fresh)
            if len(beam) > W:
                beam.sort(key=lambda s: s.size, reverse=True)
                beam = beam[:W]
                
    return best_state.points
