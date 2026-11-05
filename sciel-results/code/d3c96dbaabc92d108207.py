import time
import random
import sys
import math

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy:
    1. Precompute all edges and faces (C4s).
    2. Initialize with a strong deterministic construction:
       Use a linear algebraic rule over GF(2).
       Rule: Edge (u, u^i) is included if sum_{k=0}^{6} u_k * c_{k,i} == 0 (mod 2),
       where c_{k,i} are coefficients from a fixed matrix.
       Specifically, we can use the rule: include edge in dimension i at vertex u
       if the dot product of u (as a vector) with a fixed vector v_i is 0.
       To ensure C4-freeness, we need that for any face (u, u^i, u^j, u^i^j),
       the four edges are not all present.
       
       A known good construction for Q_n C4-free is related to finite fields or specific linear forms.
       Let's try: Include edge (u, u^i) if (u >> i) & 1 == 0? No, that's just half.
       
       Let's use the "Parity of popcount" variant but optimized.
       Actually, a better deterministic start:
       Randomly permute dimensions and use a greedy fill?
       
       Let's use a simple linear rule:
       Let A be a 7x7 matrix over GF(2).
       Edge (u, u^i) is included if sum_{j} A[i][j] * u_j == 0.
       For a face with bits i, j:
       Edges:
       e1: (u, u^i) -> condition: A[i] . u == 0
       e2: (u, u^j) -> condition: A[j] . u == 0
       e3: (u^i, u^i^j) -> edge in dim j from u^i. Condition: A[j] . (u^i) == A[j].u + A[j][i] == 0
       e4: (u^j, u^i^j) -> edge in dim i from u^j. Condition: A[i] . (u^j) == A[i].u + A[i][j] == 0
       
       We want to avoid e1,e2,e3,e4 all being 1.
       If we choose A to be the identity matrix:
       e1: u_i == 0
       e2: u_j == 0
       e3: (u^i)_j = u_j == 0
       e4: (u^j)_i = u_i == 0
       So if u_i=0 and u_j=0, all 4 are present. This creates C4s.
       
       We need A[i][j] such that we don't have this.
       If A is a matrix where A[i][j] != delta_{ij}, we might break it.
       Let's try A[i][j] = 1 for all i != j, and 0 for i == j.
       e1: sum_{k != i} u_k == 0
       e2: sum_{k != j} u_k == 0
       e3: sum_{k != j} (u_k + delta_{k,i}) == sum_{k != j, k != i} u_k + (1 if i != j else 0)
       This is getting complex.
       
       Let's just use a high-quality random restart + Simulated Annealing.
       The current seed uses "Repair + Add". SA explores the space more thoroughly.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
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
    
    # Precompute for each edge, which faces it belongs to
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # Precompute for each face, the 4 edges (already have this in faces)
    # And for fast checking, let's store faces as tuples of 4 ints.
    faces_tuple = [tuple(f) for f in faces]
    
    # State representation:
    # We maintain:
    # 1. `present`: list of bools (or bytearray for speed)
    # 2. `full_count`: list of ints, count of present edges in each face.
    #    If full_count[f] == 4, it's a violation.
    
    # Initialization: Random dense state, then repair to valid.
    # We will run multiple independent SA runs.
    
    best_score = -1
    best_state = None
    
    def initialize_valid():
        """
        Initialize with a random state and repair it to be valid.
        Returns (state, full_count, score)
        """
        state = bytearray(E)
        # Start with ~70% density to have a good starting point
        p = 0.7
        for i in range(E):
            state[i] = 1 if random.random() < p else 0
        
        full_count = [0] * F
        for f_idx in range(F):
            f = faces_tuple[f_idx]
            count = state[f[0]] + state[f[1]] + state[f[2]] + state[f[3]]
            full_count[f_idx] = count
        
        # Repair: Remove edges from faces with count 4
        # To be efficient, we can just iterate until no conflicts.
        # Since we start dense, there will be many conflicts.
        # Greedy removal: pick a random face with count 4, remove a random edge from it.
        
        # Optimization: Keep a list of "conflict faces"
        conflict_faces = [f_idx for f_idx in range(F) if full_count[f_idx] == 4]
        
        while conflict_faces:
            # Pick a random conflict face
            f_idx = random.choice(conflict_faces)
            f = faces_tuple[f_idx]
            
            # Pick a random edge to remove from this face
            # Heuristic: Remove an edge that is in many other conflict faces?
            # For simplicity and speed, random is okay for initialization.
            # But let's try to remove the edge with the highest "load" (number of conflict faces it's in).
            # This requires tracking load, which is expensive.
            # Let's just remove a random edge from the 4.
            e_to_remove = random.choice(f)
            
            if state[e_to_remove]:
                state[e_to_remove] = 0
                
                # Update full_count for all faces containing e_to_remove
                for f_idx2 in edge_faces[e_to_remove]:
                    full_count[f_idx2] -= 1
                    if full_count[f_idx2] == 4:
                        # This face became a conflict? No, we removed an edge, so count decreased.
                        # It can only go from 4 to 3, or 3 to 2, etc.
                        # It cannot become 4.
                        pass
                    # We need to update the conflict_faces list.
                    # This is tricky with a list.
                    # Let's just rebuild conflict_faces periodically or use a set?
                    # For initialization, a simple loop is fine if we do it smart.
                    
                # To avoid O(F) scan for conflict_faces update, we can just re-scan every K steps
                # or just accept that the while loop condition checks `conflict_faces` which is stale.
                # Let's use a different repair strategy:
                # Just scan all faces, if count==4, remove edge. Repeat until clean.
                # This is O(F * E) worst case but F=672, E=448. 672*448 is small.
                # But we do this in a loop.
                
                # Let's restart the repair loop with a fresh scan for simplicity and robustness.
                # It's initialization, so it doesn't need to be super fast, just valid.
                break
        
        # Proper Repair Loop
        while True:
            found_conflict = False
            for f_idx in range(F):
                if full_count[f_idx] == 4:
                    f = faces_tuple[f_idx]
                    # Remove a random edge
                    e_to_remove = random.choice(f)
                    if state[e_to_remove]:
                        state[e_to_remove] = 0
                        for f_idx2 in edge_faces[e_to_remove]:
                            full_count[f_idx2] -= 1
                    found_conflict = True
                    break
            if not found_conflict:
                break
        
        score = sum(state)
        return state, full_count, score
    
    def simulated_annealing(state, full_count, initial_temp, min_temp, cooling_rate, steps_per_iter):
        """
        Run SA on the given state.
        Returns best state found during this run.
        """
        current_score = sum(state)
        best_score_local = current_score
        best_state_local = state[:]
        
        temp = initial_temp
        
        # Precompute random edges list for shuffling?
        # We will pick random edges to flip.
        
        steps_done = 0
        
        while temp > min_temp:
            for _ in range(steps_per_iter):
                # Pick a random edge to flip
                e = random.randint(0, E - 1)
                
                old_val = state[e]
                
                # Calculate delta in score
                if old_val == 1:
                    # Removing edge
                    delta_score = -1
                    # Check if this causes any face to go from 4 to 3 (good) or 3 to 2 (neutral)
                    # It cannot cause a violation because we are removing an edge.
                    # So this move is always "safe" in terms of validity, but reduces score.
                    # We only accept if we want to explore?
                    # In SA for maximization with hard constraints:
                    # If move is valid (no new C4s), accept if delta_score > 0 or with prob exp(delta_score/T).
                    # Here, removing an edge is always valid (reduces C4s).
                    # So we accept with prob 1 if we are cooling?
                    # No, we want to maximize edges. Removing an edge is bad.
                    # We only remove if it allows adding MORE edges later?
                    # SA handles this via the acceptance probability.
                    
                    # However, if the current state has violations, we MUST remove.
                    # Our state is maintained to be valid (full_count < 4 always).
                    # So removing an edge is always a valid move, but decreases score.
                    # Acceptance: exp(-1/T).
                    
                    new_val = 0
                else:
                    # Adding edge
                    delta_score = 1
                    # Check if this creates a C4.
                    # It creates a C4 if any face containing e has full_count == 3.
                    # If so, full_count becomes 4 (violation).
                    is_valid_move = True
                    for f_idx in edge_faces[e]:
                        if full_count[f_idx] == 3:
                            is_valid_move = False
                            break
                    
                    if not is_valid_move:
                        # This move would create a C4.
                        # We can reject it immediately (infinite penalty)
                        # Or allow it with a penalty?
                        # Standard approach: Reject invalid moves.
                        # But sometimes we need to temporarily allow violations to escape local optima?
                        # Given the small size, rejecting is safer and faster.
                        # Let's stick to valid moves only.
                        continue
                    
                    new_val = 1
                
                # Calculate acceptance
                if delta_score > 0:
                    accept = True
                else:
                    # delta_score is -1
                    accept = (random.random() < math.exp(delta_score / temp))
                
                if accept:
                    # Apply move
                    state[e] = new_val
                    
                    # Update full_count
                    for f_idx in edge_faces[e]:
                        full_count[f_idx] += (1 if new_val else -1)
                    
                    current_score += delta_score
                    
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_state_local = state[:]
            
            temp *= cooling_rate
            steps_done += steps_per_iter
            
            # Time check
            if time.time() - start_time > time_limit - 5:
                break
        
        return best_state_local, best_score_local

    # Main Loop: Multiple SA runs
    random.seed(42)
    
    # Parameters for SA
    # Start with high temp, cool down.
    # Since E=448, a "good" solution has ~300 edges.
    # Random walk needs to explore.
    
    num_runs = 20
    initial_temp = 1.0
    min_temp = 0.01
    cooling_rate = 0.99
    steps_per_iter = 1000
    
    for run in range(num_runs):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
        
        # Initialize
        state, full_count, score = initialize_valid()
        
        if score > best_score:
            best_score = score
            best_state = state[:]
        
        # Run SA
        best_state_sa, best_score_sa = simulated_annealing(
            state, full_count, initial_temp, min_temp, cooling_rate, steps_per_iter
        )
        
        if best_score_sa > best_score:
            best_score = best_score_sa
            best_state = best_state_sa[:]
            
            if best_score >= 308:
                break
        
        # Optional: Try to "add" edges greedily after SA?
        # SA might leave some addable edges if it cooled too fast.
        # Let's do a final greedy add on the best state found so far.
        
        if best_state is not None:
            # Reconstruct full_count for best_state?
            # Or just do a quick check on best_state.
            # Let's just trust SA.
            pass

    # Final Greedy Add on the absolute best state
    if best_state is not None:
        # Rebuild full_count for the best state to ensure we can add
        best_full_count = [0] * F
        for f_idx in range(F):
            f = faces_tuple[f_idx]
            best_full_count[f_idx] = best_state[f[0]] + best_state[f[1]] + best_state[f[2]] + best_state[f[3]]
        
        improved = True
        while improved:
            improved = False
            for e in range(E):
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
        
        final_score = sum(best_state)
        if final_score > best_score:
            best_score = final_score
            best_state = best_state[:]

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
