import time
import random
import sys

N = 9
NUM_VERTICES = 1 << N

# Precompute neighbors for each vertex
neighbors = [[] for _ in range(NUM_VERTICES)]
for v in range(NUM_VERTICES):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_path = []
    best_len = 0
    
    # We will run multiple randomized beam searches
    # Beam width
    BEAM_WIDTH = 200
    
    # Random seed for reproducibility or variation
    # Using a fixed seed might be better for consistent improvement if the algorithm is deterministic given the seed
    # But random restarts help escape local optima.
    
    # Helper to check if a move is valid using bitmasks
    # used: bitmask of vertices in path
    # disallowed: bitmask of vertices that cannot be next (neighbors of non-consecutive vertices)
    
    def run_beam_search(seed):
        random.seed(seed)
        
        # Initial state
        # Path: [0]
        # Used: 1 << 0
        # Disallowed: 0 (no non-consecutive vertices yet)
        
        # Beam: list of (path, used, disallowed)
        # To save memory, we can store just the path and recompute used/disallowed?
        # Recomputing is O(L^2). L~200. 40000 ops.
        # Bitmasks are O(1) update.
        # Let's store (path, used_mask, disallowed_mask)
        
        beam = [([0], 1, 0)]
        
        max_len_found = 0
        
        while beam and time.time() < deadline:
            next_beam = []
            for path, used_mask, disallowed_mask in beam:
                last_v = path[-1]
                
                # Find valid candidates
                # Candidates are neighbors of last_v
                # Not in used_mask
                # Not in disallowed_mask
                
                candidates = []
                for n in neighbors[last_v]:
                    bit = 1 << n
                    if not (used_mask & bit) and not (disallowed_mask & bit):
                        candidates.append(n)
                
                if not candidates:
                    # Dead end
                    if len(path) > max_len_found:
                        max_len_found = len(path)
                        # Update global best if needed?
                        # We'll collect all dead ends and check at the end, or update global
                        pass
                    continue
                
                # Select top K candidates to expand?
                # Or expand all and keep top K resulting states?
                # Expanding all might be too slow if branching factor is high.
                # Branching factor in hypercube is 9.
                # So each state produces at most 9 new states.
                # Beam size 200 -> 1800 new states.
                # This is manageable.
                
                for n in candidates:
                    new_path = path + [n]
                    new_used = used_mask | (1 << n)
                    
                    # Update disallowed
                    # The new disallowed set for the NEXT step will include neighbors of the previous last vertex (last_v)
                    # Because last_v becomes non-consecutive to the one after n.
                    # Wait, the definition of disallowed_mask in my state is:
                    # "Vertices that cannot be the NEXT vertex because they are neighbors of non-consecutive vertices in the current path."
                    # Current path ends at last_v.
                    # Non-consecutive vertices are path[0] to path[-2].
                    # So disallowed_mask currently contains N(v_0) U ... U N(v_{-2}).
                    # When we append n, the new path ends at n.
                    # The non-consecutive vertices for the NEXT step will be path[0] to path[-1] (which is last_v).
                    # So we need to add N(last_v) to the disallowed set.
                    
                    # Precompute neighbor mask for last_v
                    # We can precompute neighbor_masks[v]
                    
                    new_disallowed = disallowed_mask | neighbor_masks[last_v]
                    
                    next_beam.append((new_path, new_used, new_disallowed))
            
            if not next_beam:
                break
            
            # Sort next_beam by length (all same length in a level) or by some heuristic?
            # All paths in next_beam have the same length (len of current beam paths + 1).
            # So we just need to keep the best BEAM_WIDTH ones.
            # How to define "best"?
            # Heuristic: Number of available candidates for the next step?
            # Or just random?
            # Let's compute the number of valid extensions for each state in next_beam.
            
            scored_beam = []
            for path, used_mask, disallowed_mask in next_beam:
                last_v = path[-1]
                count = 0
                for n in neighbors[last_v]:
                    bit = 1 << n
                    if not (used_mask & bit) and not (disallowed_mask & bit):
                        count += 1
                # Heuristic: Prefer states with more options (less constrained)
                # Or prefer states with fewer options (more forced)?
                # Usually, "more options" is better for avoiding dead ends.
                scored_beam.append((count, path, used_mask, disallowed_mask))
            
            # Sort by count descending
            scored_beam.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top BEAM_WIDTH
            beam = [(p, u, d) for c, p, u, d in scored_beam[:BEAM_WIDTH]]
            
            # Update best length
            if len(beam[0][0]) > max_len_found:
                max_len_found = len(beam[0][0])
        
        return max_len_found, beam[0][0] if beam else []

    # Precompute neighbor masks
    global neighbor_masks
    neighbor_masks = [0] * NUM_VERTICES
    for v in range(NUM_VERTICES):
        mask = 0
        for n in neighbors[v]:
            mask |= (1 << n)
        neighbor_masks[v] = mask

    # Run multiple seeds
    # Since we have ~115s, we can run many seeds.
    # Each beam search step is ~200 * 9 * (bit ops) ~ fast.
    # Let's say 1000 steps per second?
    # We can run 100 seeds.
    
    best_overall_path = []
    
    for seed in range(100):
        if time.time() > deadline:
            break
        # Use a different strategy for each seed?
        # Maybe vary beam width?
        # Let's just run the same algorithm with different random seeds for tie-breaking or initial perturbations?
        # The algorithm above is deterministic if we don't use random.
        # We used random.seed(seed) but didn't use random in the logic yet.
        # Let's add randomness to the selection if counts are tied.
        
        # Actually, the sorting by count is deterministic.
        # To get diversity, we can randomize the initial state?
        # Or randomize the tie-breaking in the sort.
        
        # Let's modify the sort to include a random component for ties.
        
        # For simplicity, let's just run the deterministic beam search first.
        # If it's fast, we can add randomization.
        
        # Let's re-implement run_beam_search to accept a random generator
        
        pass

    # The logic above is a bit messy with global variables.
    # Let's clean it up.
    
    # We'll just run a few different "greedy" strategies first to get a baseline,
    # then use the beam search to improve.
    
    # 1. Greedy with lowest bit (Seed)
    # 2. Greedy with highest bit
    # 3. Greedy with random bit order
    
    # Then Beam Search.
    
    # Given the time, I will implement the Beam Search carefully.
    
    # Re-implementation:
    
    def get_best_snake():
        # Precomputation
        nbr_masks = [0] * NUM_VERTICES
        for v in range(NUM_VERTICES):
            m = 0
            for b in range(N):
                m |= 1 << (v ^ (1 << b))
            nbr_masks[v] = m

        def beam_search(seed):
            random.seed(seed)
            # Start with 0
            # State: (path, used_mask, disallowed_mask)
            # disallowed_mask contains neighbors of all vertices except the last one.
            
            beam = [([0], 1, 0)]
            best_path = [0]
            
            while beam:
                next_beam = []
                for path, used, disallowed in beam:
                    last = path[-1]
                    # Candidates: neighbors of last, not used, not disallowed
                    cand_mask = nbr_masks[last] & ~used & ~disallowed
                    # Iterate through set bits in cand_mask
                    cm = cand_mask
                    while cm:
                        lsb = cm & -cm
                        n = lsb.bit_length() - 1
                        new_path = path + [n]
                        new_used = used | lsb
                        # New disallowed: add neighbors of 'last'
                        new_disallowed = disallowed | nbr_masks[last]
                        next_beam.append((new_path, new_used, new_disallowed))
                        cm ^= lsb
                
                if not next_beam:
                    break
                
                # Score and prune
                # Score: number of available moves for the next step
                scored = []
                for path, used, disallowed in next_beam:
                    last = path[-1]
                    avail_mask = nbr_masks[last] & ~used & ~disallowed
                    score = bin(avail_mask).count('1')
                    # Add random noise to break ties and encourage diversity
                    noise = random.random() * 0.1
                    scored.append((score + noise, path, used, disallowed))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                beam = [(p, u, d) for s, p, u, d in scored[:BEAM_WIDTH]]
                
                if len(beam[0][0]) > len(best_path):
                    best_path = beam[0][0]
                    
            return best_path

        best_path = []
        # Run multiple seeds
        for s in range(50):
            if time.time() > deadline:
                break
            p = beam_search(s)
            if len(p) > len(best_path):
                best_path = p
                
        return best_path

    return get_best_snake()

# Precompute neighbors
neighbors = [[] for _ in range(NUM_VERTICES)]
for v in range(NUM_VERTICES):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

# Global variable for neighbor masks to be used in inner function
neighbor_masks = [0] * NUM_VERTICES
for v in range(NUM_VERTICES):
    mask = 0
    for n in neighbors[v]:
        mask |= (1 << n)
    neighbor_masks[v] = mask

# BEAM_WIDTH is defined in construct
BEAM_WIDTH = 200

# The code above has a nested function definition which is fine.
# I need to make sure `neighbor_masks` is accessible.
# I will define it globally.
