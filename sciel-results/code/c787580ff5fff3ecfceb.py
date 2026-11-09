import time
import numpy as np

N = 9
MAX_VERTS = 2 ** N
MASKS = np.array([1 << i for i in range(N)], dtype=np.uint32)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    # Precompute neighbor masks
    # For a set of vertices V (shape M), neighbors are V ^ mask for each mask
    # We need to check if neighbor is unused and not adjacent to path[:-1]
    
    # Strategy:
    # 1. Generate K random snakes in parallel using vectorized operations.
    # 2. Each snake is extended greedily.
    # 3. Heuristic: At each step, for each active snake, find all valid neighbors.
    #    Score each neighbor by how many *future* neighbors it has (Warnsdorff).
    #    Pick the one with the fewest future neighbors to avoid dead ends.
    # 4. Since we can't easily do dynamic branching in pure vectorization for different lengths,
    #    we simulate a "population" of snakes.
    #    Actually, a simpler and more robust vectorized approach for snakes is:
    #    - Represent the path as a list of vertices.
    #    - But lists are slow.
    #    - Alternative: Use a fixed maximum length array for each snake, and a "length" counter.
    #    - However, the "validity" check depends on the entire history of the snake.
    #    - Checking "non-consecutive vertices differ in at least two bits" means:
    #      For candidate v at step t, v must not be at Hamming distance 1 from any v_i where i < t-1.
    #      This is equivalent to: v is not in the "shadow" of the path.
    #      The shadow of a vertex u is its 9 neighbors.
    #      So v is invalid if v is in the union of neighbors of {v_0, ..., v_{t-2}}.
    #      Let S_t be the set of vertices forbidden at step t.
    #      S_t = { u | dist(u, v_i) == 1 for some i < t-1 }
    #      Note: v_{t-1} is the current end. Its neighbors are candidates.
    #      One of the candidates is v_{t-2} (backtrack), which is used.
    #      Others might be in S_t.
    #      
    #      We can maintain a "forbidden" bitset or array.
    #      Since MAX_VERTS = 512, we can use a 512-bit integer or a boolean array of size 512.
    #      For M snakes, we have M x 512 boolean matrix.
    #      This is very small (M=1000 -> 1000*512 = 512k booleans = 512KB).
    #      
    #      Algorithm for one step for M snakes:
    #      1. For each snake m, current end is cur[m].
    #      2. Generate all 9 neighbors for each snake: Neigh[m, k] = cur[m] ^ MASKS[k].
    #      3. A neighbor Neigh[m, k] is a valid candidate if:
    #         a. It is not used in snake m. (Used[m, Neigh[m,k]] == False)
    #         b. It is not in the forbidden set of snake m. (Forbidden[m, Neigh[m,k]] == False)
    #            Note: Forbidden set contains neighbors of all vertices v_0...v_{t-2}.
    #            When we add a new vertex v_{t-1} to the path, we add its neighbors to the Forbidden set?
    #            No. The constraint is on the *next* vertex v_t.
    #            v_t must not be adjacent to v_0...v_{t-2}.
    #            So Forbidden[m] should be the union of neighbors of v_0...v_{t-2}.
    #            When we move from v_{t-1} to v_t, the new forbidden set for the *next* step (v_{t+1})
    #            will be Forbidden[m] U neighbors(v_{t-1}).
    #            
    #      Let's refine the state:
    #      - cur[m]: current vertex (uint32)
    #      - used[m, v]: boolean, is v in path?
    #      - forb[m, v]: boolean, is v forbidden for the next step? (i.e., dist(v, v_i)==1 for i < t-1)
    #      
    #      Update rules when moving from cur to next:
    #      1. Check candidates: neigh = cur ^ masks.
    #         valid_mask = ~used[m, neigh] & ~forb[m, neigh]
    #      2. If no valid candidates, snake dies.
    #      3. If valid candidates exist, pick one.
    #         Heuristic: Count how many valid *next* neighbors the candidate has.
    #         This is hard to vectorize easily because it depends on the new state.
    #         Approximation: Pick the candidate that minimizes the number of unused neighbors?
    #         Or just pick randomly among valid ones?
    #         Random parallel greedy often works well for snakes if we have many restarts.
    #         Let's try "Random Valid" first, then maybe "Warnsdorff-like" if time permits.
    #         Actually, a good heuristic for snakes is to prefer vertices that are "tighter".
    #         Let's stick to random valid for speed, but use many snakes.
    #         
    #      4. Update state:
    #         used[m, next] = True
    #         forb[m, :] |= neighbors(next)  # The new forbidden set includes neighbors of the vertex we just left?
    #         Wait. 
    #         Step t: Path is v_0...v_{t-1}. Forbidden set F_t = N(v_0) U ... U N(v_{t-2}).
    #         We pick v_t from N(v_{t-1}) \ (Used U F_t).
    #         New Path v_0...v_t.
    #         New Forbidden set F_{t+1} = N(v_0) U ... U N(v_{t-1}).
    #         So F_{t+1} = F_t U N(v_{t-1}).
    #         Yes.
    #
    #      5. cur[m] = next
    #      6. length[m] += 1

    M = 2000  # Number of parallel snakes
    cur = np.zeros(M, dtype=np.uint32)
    # Initialize with random starts
    rng = np.random.default_rng(42)
    cur[:] = rng.integers(0, MAX_VERTS, size=M)
    
    used = np.zeros((M, MAX_VERTS), dtype=bool)
    forb = np.zeros((M, MAX_VERTS), dtype=bool)
    
    # Initialize used
    used[np.arange(M), cur] = True
    
    # Initialize forb:
    # For the first step, F_1 = N(v_0)? No.
    # Path: v_0.
    # Next vertex v_1 must be in N(v_0) and not adjacent to v_{-1} (none).
    # So F_1 is empty?
    # Condition: v_1 must not be adjacent to v_i for i < 0. None.
    # So forb is initially all False.
    # Wait, v_1 must be distinct from v_0. used handles that.
    # v_1 must not be adjacent to v_0? No, v_1 IS adjacent to v_0. That's the edge.
    # The condition is "non-consecutive vertices differ in at least two bits".
    # So v_1 and v_0 are consecutive, dist=1 is OK.
    # v_2 and v_0 must have dist != 1.
    # So when picking v_2, we must avoid neighbors of v_0.
    # So F_2 = N(v_0).
    # Generally, when picking v_t, we must avoid N(v_0) U ... U N(v_{t-2}).
    # So at step t (picking v_t), forb[m] should contain neighbors of v_0...v_{t-2}.
    
    # Let's trace:
    # t=0: cur=v_0. used has v_0.
    # Pick v_1. Candidates: N(v_0). Check used? v_0 is used, but v_0 is in N(v_0)? No, v_0 is not its own neighbor.
    # Check forb? F_1 is empty. So all 9 neighbors are valid candidates (unless used, which they aren't).
    # Update: used[v_1]=True.
    # New forb for next step (t=1, picking v_2): F_2 = N(v_0).
    # So after picking v_1, we set forb[m, N(v_0)] = True.
    
    # So initialization:
    # cur = v_0.
    # used[v_0] = True.
    # forb is all False.
    # Step 1:
    #   neigh = cur ^ masks.
    #   valid = ~used[neigh] & ~forb[neigh].
    #   Pick next.
    #   Update used[next] = True.
    #   Update forb: forb |= neighbors(cur).  # Because cur is now v_{t-1}, and for the *next* step, we must avoid neighbors of v_{t-1}?
    #   Wait.
    #   If we are at v_{t-1} and pick v_t.
    #   The next step picks v_{t+1}.
    #   v_{t+1} must not be adjacent to v_0...v_{t-1}.
    #   So the new forbidden set is Old_Forbidden U N(v_{t-1}).
    #   Yes.
    
    lengths = np.zeros(M, dtype=np.int32)
    lengths[:] = 1
    
    # Precompute neighbor indices for vectorization?
    # neigh = np.stack([cur ^ m for m in MASKS], axis=1) # Shape (M, 9)
    
    start_time = time.time()
    
    # Main loop
    while True:
        if time.time() > deadline - 5:
            break
        
        # 1. Generate neighbors (M, 9)
        neigh = np.stack([cur ^ m for m in MASKS], axis=1).astype(np.uint32)
        
        # 2. Check validity
        # used_neigh: (M, 9) boolean. used[m, neigh[m, k]]
        # We can use advanced indexing.
        # used[m, neigh[m, k]] is tricky with numpy 2D indexing.
        # used is (M, MAX_VERTS).
        # We want used[m, neigh[m, k]] for all m, k.
        # indices_m = np.arange(M)[:, None]
        # indices_n = neigh
        # used_neigh = used[indices_m, indices_n]
        
        indices_m = np.arange(M)[:, None]
        used_neigh = used[indices_m, neigh]
        forb_neigh = forb[indices_m, neigh]
        
        valid = ~used_neigh & ~forb_neigh
        
        # 3. Check if any valid
        has_valid = np.any(valid, axis=1)
        
        if not np.any(has_valid):
            break
            
        # 4. Pick a next vertex for each valid snake
        # For simplicity, pick the first valid one? Or random?
        # Random is better for diversity.
        # To vectorize random pick:
        # We can create a mask of valid indices.
        # This is a bit complex. 
        # Alternative: Assign a random priority to each of the 9 neighbors, then pick the argmin among valid ones.
        
        # Generate random priorities (M, 9)
        # But we only want to pick from valid ones.
        # Set invalid priorities to infinity.
        priorities = rng.random((M, 9))
        priorities[~valid] = np.inf
        
        next_idx = np.argmin(priorities, axis=1)
        
        # For snakes that had no valid move, next_idx will be 0 (arbitrary). We mask them out.
        
        next_v = neigh[np.arange(M), next_idx]
        
        # 5. Update state
        # Only update for snakes that are still alive
        active = has_valid
        
        # Update used
        used[active, next_v[active]] = True
        
        # Update forb
        # New forb = old forb | neighbors(cur)
        # neighbors(cur) is (M, 9). We need to set forb[m, neigh_cur[m, k]] = True.
        # neigh_cur = np.stack([cur ^ m for m in MASKS], axis=1)
        # We already have neigh which is neighbors of cur.
        # So we want to OR into forb the neighbors of cur.
        # forb[active, neigh[active]] = True
        
        # Update cur
        cur[active] = next_v[active]
        
        # Update lengths
        lengths[active] += 1
        
        # Check time
        if time.time() > deadline - 5:
            break

    # Find the best snake
    best_idx = np.argmax(lengths)
    best_len = lengths[best_idx]
    
    # We need to reconstruct the path for the best snake.
    # But we didn't store the history!
    # This is a problem. We need to return the list of vertices.
    
    # Strategy Adjustment:
    # We cannot just run parallel greedy without storing history if we need to output the path.
    # Storing history for M=2000 snakes of length ~200 is 2000*200*4 bytes = 1.6 MB. That's fine.
    # But updating a list of lists in a vectorized loop is hard.
    
    # Alternative:
    # Run the parallel greedy to find the *strategy* or just the *length*?
    # No, we need the actual path.
    
    # Revised Strategy:
    # 1. Use the parallel vectorized search to find a *good heuristic* or just to identify that random greedy works?
    # 2. Actually, let's just run a single, very fast, optimized Python greedy with a strong heuristic, 
    #    but use the vectorized idea for the "lookahead" or "scoring".
    #    
    # Let's go back to a single snake but with a much better heuristic and speed.
    # The previous program was slow because of the `is_induced_extension` loop checking all previous vertices.
    # We can maintain a `forbidden` set (or bitset) to make the check O(1).
    #
    # Fast Single-Snake Algorithm:
    # 1. Maintain `cur`, `used` (set or bool array), `forb` (bool array or set).
    # 2. At each step:
    #    - Candidates = neighbors of cur.
    #    - Filter: c not in used AND c not in forb.
    #    - If no candidates, stuck.
    #    - Score candidates:
    #      - For each candidate c, count how many of ITS neighbors are valid for the NEXT step.
    #      - Valid for next step: not in used, not in (forb U neighbors(c)).
    #      - This is O(N^2) per step, but N=9, so 81 ops. Very fast.
    #    - Pick best candidate.
    #    - Update: used.add(c), forb.update(neighbors(cur)), cur=c.
    #
    # This is much faster than the previous O(L) check per step.
    # Let's implement this.

    def fast_greedy(start_vertex, rng_seed):
        rng = np.random.default_rng(rng_seed)
        cur = start_vertex
        used = np.zeros(MAX_VERTS, dtype=bool)
        forb = np.zeros(MAX_VERTS, dtype=bool)
        used[cur] = True
        path = [cur]
        
        # Precompute neighbors for all vertices?
        # neighs = np.array([v ^ m for m in MASKS] for v in range(MAX_VERTS))
        # This is 512x9. Small.
        # But we can just compute on the fly.
        
        while True:
            # Get neighbors of cur
            neighs = cur ^ MASKS  # This doesn't work directly with numpy array broadcasting like this?
            # MASKS is (9,). cur is scalar.
            # neighs = np.array([cur ^ m for m in MASKS])
            
            # Filter valid
            valid_mask = ~used[neighs] & ~forb[neighs]
            
            if not np.any(valid_mask):
                break
            
            valid_neighs = neighs[valid_mask]
            
            # Score each valid neighbor
            # Score = number of valid next-next moves
            # For candidate c:
            #   next_forb = forb | (c ^ MASKS)  # Neighbors of c are added to forb for the step after c?
            #   Wait.
            #   If we pick c, the new forb for the *next* step is forb | neighbors(cur).
            #   Wait, no.
            #   Current state: Path ends at cur. forb contains neighbors of v_0...v_{t-2}.
            #   We pick c (neighbor of cur).
            #   New path ends at c.
            #   New forb for next step (picking v_{t+1}) is (old forb) | neighbors(cur).
            #   So the "forb" used to validate v_{t+1} is new_forb.
            #   The candidates for v_{t+1} are neighbors of c.
            #   So we check: for each u in neighbors(c), is u in used? is u in new_forb?
            
            new_forb = forb.copy()
            new_forb[cur ^ MASKS] = True  # Add neighbors of cur to forb
            
            scores = np.zeros(len(valid_neighs))
            for i, c in enumerate(valid_neighs):
                c_neighs = c ^ MASKS
                # Valid if not used and not in new_forb
                # Note: c itself is not in c_neighs.
                # cur is in c_neighs, but cur is used. So it will be filtered.
                valid_next = ~used[c_neighs] & ~new_forb[c_neighs]
                scores[i] = np.sum(valid_next)
            
            # Warnsdorff: pick min score
            # If tie, break randomly
            min_score = np.min(scores)
            tie_mask = scores == min_score
            tie_indices = np.where(tie_mask)[0]
            chosen_local_idx = tie_indices[rng.integers(len(tie_indices))]
            next_v = valid_neighs[chosen_local_idx]
            
            # Update
            path.append(next_v)
            used[next_v] = True
            # Update forb for the actual step taken
            # The forb for the NEXT iteration (picking v_{t+2}) will be based on the state after adding next_v.
            # Wait, my logic above for `new_forb` was for the *hypothetical* step.
            # In the actual update:
            # We moved from cur to next_v.
            # The forb set for the *next* iteration (which will pick a neighbor of next_v)
            # should be the old forb PLUS neighbors of cur.
            # Because the new path is v_0...v_{t-1}(cur), v_t(next_v).
            # The next vertex v_{t+1} must not be adjacent to v_0...v_{t-1}.
            # So yes, forb |= neighbors(cur).
            
            forb[cur ^ MASKS] = True
            cur = next_v
            
        return path

    # Run many fast greedy searches
    best_path = []
    best_len = 0
    
    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        start = rng.integers(0, MAX_VERTS)
        path = fast_greedy(start, i)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
    return best_path
# EVOLVE-BLOCK-END
