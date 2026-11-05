"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d

    # 1. Precompute edges and faces
    # Edges are represented by their ID. 
    # edge_id_map maps (u, v) to id
    edges = []
    edge_id_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id = len(edges)
                edges.append((x, y))
                edge_id_map[(x, y)] = edge_id
                edge_id_map[(y, x)] = edge_id

    n_edges = len(edges)
    
    # Faces: A 2D face is defined by a vertex v and two distinct dimensions i, j.
    # The 4 edges of the face are:
    # (v, v^i), (v^i, v^i^j), (v^j, v^i^j), (v, v^j)
    face_edges = []
    seen_faces = set()
    for v in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v_i = v ^ (1 << i)
                v_j = v ^ (1 << j)
                v_ij = v ^ (1 << i) ^ (1 << j)
                
                # Get edge IDs
                e1 = edge_id_map[(min(v, v_i), max(v, v_i))]
                e2 = edge_id_map[(min(v_i, v_ij), max(v_i, v_ij))]
                e3 = edge_id_map[(min(v, v_j), max(v, v_j))]
                e4 = edge_id_map[(min(v_j, v_ij), max(v_j, v_ij))]
                
                # Normalize face representation to avoid duplicates (though each face is unique per v,i,j)
                # Actually, each face is uniquely identified by its 4 edges. 
                # We can just store the list of 4 edge IDs.
                face_key = tuple(sorted([e1, e2, e3, e4]))
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    face_edges.append([e1, e2, e3, e4])

    n_faces = len(face_edges)
    
    # Incidence: edge_faces[e] = list of face indices containing edge e
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)

    # Helper: Calculate cover counts for a given set R
    def get_cover_counts(R):
        cover_count = [0] * n_faces
        for e in R:
            for f in edge_faces[e]:
                cover_count[f] += 1
        return cover_count

    # Helper: Check if R is a valid hitting set (covers all faces)
    def is_valid(R, cover_count):
        for f in range(n_faces):
            if cover_count[f] == 0:
                return False
        return True

    # Helper: Greedily repair R to cover all faces
    def repair(R, cover_count, rng):
        # cover_count is modified in place
        while True:
            uncovered = [f for f in range(n_faces) if cover_count[f] == 0]
            if not uncovered:
                break
            
            # Pick a random uncovered face
            f = rng.choice(uncovered)
            
            # Find the best edge in face_edges[f] to add
            best_e = -1
            max_benefit = -1
            
            for e in face_edges[f]:
                if e in R:
                    continue
                # Benefit is number of currently uncovered faces this edge covers
                benefit = 0
                for ff in edge_faces[e]:
                    if cover_count[ff] == 0:
                        benefit += 1
                if benefit > max_benefit:
                    max_benefit = benefit
                    best_e = e
                elif benefit == max_benefit and rng.random() < 0.5:
                    best_e = e
            
            if best_e != -1:
                R.add(best_e)
                for ff in edge_faces[best_e]:
                    cover_count[ff] += 1
            else:
                # Fallback: add a random edge from the face
                e = rng.choice(face_edges[f])
                R.add(e)
                for ff in edge_faces[e]:
                    cover_count[ff] += 1
            
            if time.time() - start_time > time_limit - 5:
                break
        
        return R

    # Helper: Remove redundant edges from R
    def reduce(R, cover_count, rng):
        # Randomly shuffle R to remove redundancy in different orders
        R_list = list(R)
        rng.shuffle(R_list)
        changed = True
        while changed:
            changed = False
            for e in R_list:
                if e not in R:
                    continue
                # Check if e is redundant
                redundant = True
                for f in edge_faces[e]:
                    if cover_count[f] == 1:
                        redundant = False
                        break
                if redundant:
                    R.remove(e)
                    for f in edge_faces[e]:
                        cover_count[f] -= 1
                    changed = True
                    if time.time() - start_time > time_limit - 5:
                        return R
        return R

    # 2. Initial Solution: Greedy Hitting Set
    def greedy_init(rng):
        R = set()
        cover_count = [0] * n_faces
        # Precompute benefits dynamically? 
        # For simplicity, just pick edges covering most uncovered faces.
        # This is O(n_edges * n_faces) per step, which is 448 * 1344 ~ 600k ops.
        # 448 steps -> ~260M ops. Might be slow in Python.
        # Optimization: Only check edges not in R.
        
        # Actually, a faster greedy:
        # 1. Start with all faces uncovered.
        # 2. Pick edge e that covers max uncovered faces.
        # 3. Add to R, update cover_count.
        
        # To speed up, we can just use a random greedy with some intelligence.
        # Or use the "repair" logic starting from empty set.
        # Repair from empty set:
        # 1. All faces uncovered.
        # 2. Pick face, pick best edge in it.
        # This is effectively the same.
        
        # Let's try a standard greedy:
        # Calculate initial benefit for all edges (number of faces they cover)
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        for _ in range(n_faces): # Max n_faces steps needed
            best_e = -1
            max_ben = -1
            candidates = []
            
            for e in range(n_edges):
                if e in R:
                    continue
                if benefit[e] > max_ben:
                    max_ben = benefit[e]
                    candidates = [e]
                elif benefit[e] == max_ben:
                    candidates.append(e)
            
            if not candidates or max_ben == 0:
                break
                
            chosen_e = rng.choice(candidates)
            R.add(chosen_e)
            
            # Update benefits for edges in the same faces
            for f in edge_faces[chosen_e]:
                if cover_count[f] == 0: # Only if this was the first edge covering it? No, benefit is "uncovered faces covered".
                    # If cover_count[f] was 0, it was uncovered. Now it is covered.
                    # So for all other edges in face f, their benefit decreases by 1.
                    for e in face_edges[f]:
                        if e != chosen_e and e not in R:
                            benefit[e] -= 1
                cover_count[f] += 1
            
            # Check if all covered
            all_covered = True
            for f in range(n_faces):
                if cover_count[f] == 0:
                    all_covered = False
                    break
            if all_covered:
                break
            
            if time.time() - start_time > time_limit - 20:
                break
        
        # Reduce
        R = reduce(R, cover_count, rng)
        return R, cover_count

    # 3. Main Search Loop: Simulated Annealing with Kicks
    best_R = None
    best_size = float('inf')
    
    # Get initial solution
    rng = random.Random(42)
    R, cover_count = greedy_init(rng)
    R = reduce(R, cover_count, rng)
    best_R = set(R)
    best_size = len(R)
    
    current_R = set(R)
    current_cover_count = list(cover_count)
    current_size = len(R)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    no_improve = 0
    
    while time.time() - start_time < time_limit - 5:
        # Kick: Remove some random edges from R
        num_remove = rng.randint(2, 5)
        if len(current_R) > num_remove:
            to_remove = rng.sample(list(current_R), num_remove)
            perturbed_R = set(current_R)
            for e in to_remove:
                perturbed_R.remove(e)
            
            # Calculate new cover counts
            perturbed_cover = list(current_cover_count)
            for e in to_remove:
                for f in edge_faces[e]:
                    perturbed_cover[f] -= 1
            
            # Repair
            perturbed_R = repair(perturbed_R, perturbed_cover, rng)
            # Reduce
            perturbed_R = reduce(perturbed_R, perturbed_cover, rng)
            perturbed_size = len(perturbed_R)
            
            # Acceptance
            delta = perturbed_size - current_size
            if delta <= 0:
                current_R = perturbed_R
                current_cover_count = perturbed_cover
                current_size = perturbed_size
                no_improve = 0
                if current_size < best_size:
                    best_size = current_size
                    best_R = set(current_R)
            else:
                # Accept with probability exp(-delta / T)
                if T > 0 and rng.random() < math.exp(-delta / T):
                    current_R = perturbed_R
                    current_cover_count = perturbed_cover
                    current_size = perturbed_size
                no_improve += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
            # If stuck for a while, restart T
            if no_improve > 100:
                T = 10.0
                no_improve = 0

    # Final Reduction
    final_cover = get_cover_counts(best_R)
    best_R = reduce(best_R, final_cover, rng)

    # 4. Convert Hitting Set R to Edge List
    # The result is all edges NOT in R
    result = []
    for e_id, (u, v) in enumerate(edges):
        if e_id not in best_R:
            result.append([u, v])
            
    return result
# EVOLVE-BLOCK-END
