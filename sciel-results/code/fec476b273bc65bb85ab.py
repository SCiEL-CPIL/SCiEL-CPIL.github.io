import time
import random
import sys
import math
import numpy as np

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy:
    1. Precompute edges and faces.
    2. Initialize with a dense random state and repair it to validity.
    3. Run Vectorized Simulated Annealing using NumPy for speed.
    4. Final greedy pass to add any remaining possible edges.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # 1. Precompute edges and faces
    edge_list = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    E = len(edge_list)  # 448
    
    # Precompute faces (C4s)
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append([idx1, idx2, idx3, idx4])
    
    F = len(faces)  # 672
    
    # Convert faces to numpy array for vectorization
    faces_np = np.array(faces, dtype=np.int32) # Shape (F, 4)
    
    # Precompute edge-to-faces mapping as a list of lists for fast access in Python loops if needed,
    # but for vectorized SA, we might need a different structure or just rely on the faces_np.
    # Actually, for vectorized SA, we need to know which faces are affected by a set of flipped edges.
    # It's hard to vectorize the "which faces are affected" part efficiently without sparse matrices.
    # Let's use a sparse matrix representation or just stick to a fast Python loop for the update,
    # but vectorize the acceptance check?
    
    # Alternative: Use a "block" SA where we flip a set of edges at once?
    # Or, use a simpler vectorized approach:
    # State is a vector of 0/1.
    # To check if adding edge e is valid, we need to check all faces containing e.
    # This is still a loop over faces for each edge.
    
    # Let's try a different vectorization:
    # Instead of flipping one edge, flip a random subset? No, that's complex.
    
    # Let's optimize the Python loop SA by using local variables and minimal overhead.
    # The current SA is likely bottlenecked by `random.randint` and list lookups.
    
    # Let's try a "Parallel" SA approach using multiple independent runs in threads?
    # Python GIL makes this hard for CPU-bound tasks.
    
    # Let's try a smarter heuristic: "Greedy Add with Random Restart".
    # 1. Start with a valid state (e.g., all 0s).
    # 2. Repeatedly:
    #    a. Pick a random order of edges.
    #    b. Try to add each edge if it doesn't create a C4.
    #    c. This is a greedy construction.
    #    d. Then, try to remove edges to allow adding more? (Local search)
    
    # Let's stick to SA but optimize the inner loop.
    # Key optimization: Precompute `edge_faces` as a list of numpy arrays? No, lists are fine.
    # Use `bytearray` for state.
    # Use `array('i')` or list for full_count.
    
    # Let's try a "Simulated Annealing with Large Moves" (flipping multiple edges).
    # Or, let's just make the single-edge SA as fast as possible.
    
    # Fast SA Implementation:
    # 1. Precompute `edge_faces` as a list of lists of ints.
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # 2. State: bytearray
    # 3. Full count: list of ints
    
    def initialize_valid():
        """
        Initialize with a random dense state and repair it to be valid.
        """
        state = bytearray(E)
        # Start with ~80% density
        p = 0.8
        for i in range(E):
            state[i] = 1 if random.random() < p else 0
        
        full_count = [0] * F
        for f_idx in range(F):
            f = faces[f_idx]
            count = state[f[0]] + state[f[1]] + state[f[2]] + state[f[3]]
            full_count[f_idx] = count
        
        # Repair: Remove edges from faces with count 4
        # We use a queue of conflict faces
        import collections
        conflict_queue = collections.deque([f_idx for f_idx in range(F) if full_count[f_idx] == 4])
        
        while conflict_queue:
            f_idx = conflict_queue.popleft()
            if full_count[f_idx] < 4:
                continue # Already fixed
            
            f = faces[f_idx]
            # Pick an edge to remove. Heuristic: remove one that is in many other conflict faces?
            # For speed, just pick random.
            e_to_remove = random.choice(f)
            
            if state[e_to_remove]:
                state[e_to_remove] = 0
                for f_idx2 in edge_faces[e_to_remove]:
                    full_count[f_idx2] -= 1
                    if full_count[f_idx2] == 4:
                        conflict_queue.append(f_idx2)
        
        score = sum(state)
        return state, full_count, score
    
    def run_sa(state, full_count, time_budget):
        """
        Run SA for a given time budget.
        Returns best state and score.
        """
        current_score = sum(state)
        best_score = current_score
        best_state = state[:]
        
        # SA Parameters
        # We want to explore enough to find ~304+.
        # Start temp high enough to accept bad moves, cool down.
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99999 # Very slow cooling to explore more
        
        # Precompute random numbers? No, use random module.
        
        # To speed up, we can skip the acceptance check for positive delta (always accept).
        # For negative delta, we need exp(-1/T).
        
        steps = 0
        while T > T_min:
            # Batch of steps?
            # Let's do 1000 steps per temp update
            for _ in range(1000):
                e = random.randint(0, E - 1)
                old_val = state[e]
                
                if old_val == 1:
                    # Remove edge
                    delta = -1
                    # Always valid
                    # Accept with prob exp(-1/T)
                    if random.random() < math.exp(-1.0 / T):
                        state[e] = 0
                        for f_idx in edge_faces[e]:
                            full_count[f_idx] -= 1
                        current_score += delta
                else:
                    # Add edge
                    # Check validity
                    valid = True
                    for f_idx in edge_faces[e]:
                        if full_count[f_idx] == 3:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Always accept
                    state[e] = 1
                    for f_idx in edge_faces[e]:
                        full_count[f_idx] += 1
                    current_score += 1
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_state = state[:]
            
            T *= cooling_rate
            
            if time.time() - start_time > time_budget:
                break
        
        return best_state, best_score

    # Main
    random.seed(42)
    
    best_score = -1
    best_state = None
    
    # Run multiple SA instances
    # Each instance gets a fraction of the time
    num_runs = 10
    time_per_run = (time_limit - 5) / num_runs
    
    for i in range(num_runs):
        # Initialize
        state, full_count, score = initialize_valid()
        
        if score > best_score:
            best_score = score
            best_state = state[:]
        
        # Run SA
        sa_state, sa_score = run_sa(state, full_count, start_time + time_per_run * (i+1))
        
        if sa_score > best_score:
            best_score = sa_score
            best_state = sa_state[:]
            
        if time.time() - start_time > time_limit - 10:
            break
    
    # Final Greedy Add
    if best_state is not None:
        # Rebuild full_count
        best_full_count = [0] * F
        for f_idx in range(F):
            f = faces[f_idx]
            best_full_count[f_idx] = best_state[f[0]] + best_state[f[1]] + best_state[f[2]] + best_state[f[3]]
        
        improved = True
        while improved:
            improved = False
            # Shuffle edges to avoid bias
            edges_to_try = list(range(E))
            random.shuffle(edges_to_try)
            for e in edges_to_try:
                if best_state[e]:
                    continue
                can_add = True
                for f_idx in edge_faces[e]:
                    if best_full_count[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    best_state[e] = 1
                    for f_idx in edge_faces[e]:
                        best_full_count[f_idx] += 1
                    improved = True
    
    if best_state is None:
        return []
        
    result = []
    for i in range(E):
        if best_state[i]:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
