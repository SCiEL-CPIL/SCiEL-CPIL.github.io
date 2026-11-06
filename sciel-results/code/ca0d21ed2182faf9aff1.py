# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    # adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    # common[i][j] is the number of common neighbors of i and j
    # We maintain this to make O(1) checks and O(degree) updates
    common = [[0] * n for _ in range(n)]
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_adjacent(u, v):
        return (adj[u] >> v) & 1
    
    def try_add(u, v):
        """
        Attempts to add edge (u, v).
        Returns True if added, False otherwise.
        """
        if is_adjacent(u, v):
            return False
        
        # Check condition for pair (u, v)
        if common[u][v] >= 3:
            return False
            
        # Check conditions for neighbors of u
        # If we add (u, v), then u becomes a common neighbor of v and any w in N(u).
        # So common[v][w] increases by 1 for all w in N(u), w != v.
        # We need common[v][w] to be < 2 before adding, so it becomes <= 2.
        # Iterate through neighbors of u
        mask_u = adj[u]
        # Exclude v from mask_u for the loop? v is not in adj[u] yet.
        # So we just iterate all neighbors of u.
        temp_mask = mask_u
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            if w == v:
                # This shouldn't happen if not adjacent, but safe check
                pass
            else:
                if common[v][w] >= 2:
                    return False
            temp_mask &= temp_mask - 1
            
        # Check conditions for neighbors of v
        mask_v = adj[v]
        temp_mask = mask_v
        while temp_mask:
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            if w == u:
                pass
            else:
                if common[u][w] >= 2:
                    return False
            temp_mask &= temp_mask - 1
            
        # All checks passed, add the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update common counts
        common[u][v] += 1
        common[v][u] += 1
        
        # For each neighbor w of u (w != v), common[v][w] and common[w][v] increase by 1
        # Note: v is not in N(u) yet, so we iterate all w in N(u)
        mask_u = adj[u]
        # We need to update common[v][w] for w in N(u).
        # But wait, does adding (u,v) affect common[u][w] for w in N(u)?
        # N(u) gains v. N(w) is unchanged.
        # Common neighbors of u and w:
        # Old: N(u) \cap N(w).
        # New: (N(u) \cup {v}) \cap N(w) = (N(u) \cap N(w)) \cup ({v} \cap N(w)).
        # If v is in N(w), then common[u][w] increases by 1.
        # Similarly, if u is in N(w), common[v][w] increases by 1.
        
        # So we need to check if v is in N(w) and if u is in N(w).
        
        # Iterate neighbors of u
        temp_mask = adj[u]
        while temp_mask:
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            # w is a neighbor of u.
            # Check if v is neighbor of w
            if (adj[w] >> v) & 1:
                common[u][w] += 1
                common[w][u] += 1
            # Check if u is neighbor of w (always true since w in N(u))
            # This implies u is a common neighbor of v and w?
            # Common neighbors of v and w:
            # New N(v) includes u.
            # If u is in N(w), then u is a common neighbor.
            # Since w is in N(u), u is in N(w) (symmetry).
            # So common[v][w] always increases by 1 for w in N(u).
            common[v][w] += 1
            common[w][v] += 1
            temp_mask &= temp_mask - 1
            
        # Iterate neighbors of v
        temp_mask = adj[v]
        while temp_mask:
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            # w is a neighbor of v.
            # Check if u is neighbor of w
            if (adj[w] >> u) & 1:
                common[v][w] += 1
                common[w][v] += 1
            # Check if v is neighbor of w (always true)
            # So common[u][w] always increases by 1 for w in N(v).
            common[u][w] += 1
            common[w][u] += 1
            temp_mask &= temp_mask - 1
        # Note: The pair (u,v) was handled separately.
        # Did we double count?
        # In the first loop (neighbors of u), we updated common[v][w] and common[u][w] (if v in N(w)).
        # In the second loop (neighbors of v), we updated common[u][w] and common[v][w] (if u in N(w)).
        # For a vertex w that is neighbor to BOTH u and v:
        # Loop 1: common[v][w] += 1 (correct, u is new common nb).
        #         common[u][w] += 1 (correct, v is new common nb).
        # Loop 2: common[u][w] += 1 (correct, v is new common nb).
        #         common[v][w] += 1 (correct, u is new common nb).
        # Wait, this doubles the increment for common[u][w] and common[v][w] if w is common neighbor of u and v.
        # Let's trace carefully.
        
        # Let C(u,w) = |N(u) \cap N(w)|.
        # Add edge (u,v).
        # N(u) <- N(u) \cup {v}. N(v) <- N(v) \cup {u}.
        
        # For pair (u, w) where w \in N(u):
        # New N(u) \cap N(w) = (Old N(u) \cup {v}) \cap N(w) = (Old N(u) \cap N(w)) \cup ({v} \cap N(w)).
        # So C(u,w) increases by 1 if v \in N(w).
        
        # For pair (v, w) where w \in N(v):
        # New N(v) \cap N(w) = (Old N(v) \cup {u}) \cap N(w) = (Old N(v) \cap N(w)) \cup ({u} \cap N(w)).
        # So C(v,w) increases by 1 if u \in N(w).
        
        # My update logic:
        # Loop 1 (w in N(u)):
        #   common[v][w] += 1. This is C(v,w).
        #   Is u in N(w)? Yes, because w in N(u) => u in N(w).
        #   So C(v,w) SHOULD increase by 1. Correct.
        #   common[u][w] += 1 IF v in N(w). Correct.
        
        # Loop 2 (w in N(v)):
        #   common[u][w] += 1. This is C(u,w).
        #   Is v in N(w)? Yes, because w in N(v) => v in N(w).
        #   So C(u,w) SHOULD increase by 1. Correct.
        #   common[v][w] += 1 IF u in N(w). Correct.
        
        # If w is in N(u) AND N(v):
        # Loop 1: common[v][w] += 1. common[u][w] += 1 (since v in N(w) is true).
        # Loop 2: common[u][w] += 1. common[v][w] += 1 (since u in N(w) is true).
        # Total increment for C(v,w) is 2?
        # Let's check theory.
        # C(v,w) = |N(v) \cap N(w)|.
        # N(v) gains u. N(w) is fixed.
        # Intersection gains u if u in N(w).
        # Since w in N(u), u in N(w). So C(v,w) gains 1.
        # Why did I add 1 twice?
        # Ah, in Loop 1, I added 1 to common[v][w] unconditionally for w in N(u).
        # In Loop 2, I added 1 to common[v][w] conditionally if u in N(w).
        # If w in N(u), then u in N(w). So both conditions are true.
        # So I added 2. But it should only be 1.
        
        # Error in logic:
        # C(v,w) increases by 1 if u \in N(w).
        # u \in N(w) is equivalent to w \in N(u).
        # So for any w \in N(u), C(v,w) increases by 1.
        # For any w \in N(v), C(v,w) increases by 1 if u \in N(w).
        # If w is in N(u) \cap N(v), it is counted in Loop 1 (unconditional) and Loop 2 (conditional, which is true).
        # So yes, double counting.
        
        # Correction:
        # We should only update C(v,w) for w \in N(v)? No.
        # C(v,w) changes if N(v) changes or N(w) changes.
        # N(v) changes (gains u).
        # So C(v,w) changes for ALL w.
        # C(v,w) increases by 1 if u \in N(w).
        # u \in N(w) iff w \in N(u).
        # So C(v,w) increases by 1 for all w \in N(u).
        # It does NOT increase for w \notin N(u).
        # So the update for C(v,w) is: for all w \in N(u), C(v,w) += 1.
        # This is exactly what Loop 1 does (unconditional).
        # Loop 2 also does it if u \in N(w).
        # But if w \in N(v) and w \notin N(u), then u \notin N(w), so C(v,w) does NOT increase due to N(v) change?
        # Wait. N(v) gains u.
        # New C(v,w) = |(Old N(v) \cup {u}) \cap N(w)| = |Old N(v) \cap N(w)| + (1 if u \in N(w) else 0).
        # So it increases by 1 if u \in N(w).
        # u \in N(w) is false if w \notin N(u).
        # So for w \in N(v) \setminus N(u), C(v,w) does NOT increase.
        # So Loop 2's conditional check `if u in N(w)` is correct and necessary.
        # But Loop 1's unconditional update `common[v][w] += 1` for w in N(u) is also correct.
        # The problem is when w is in N(u) AND N(v).
        # Loop 1: w in N(u) -> common[v][w] += 1.
        # Loop 2: w in N(v) and u in N(w) (true) -> common[v][w] += 1.
        # Total +2. Should be +1.
        
        # Fix: Only do the update for C(v,w) in ONE place.
        # C(v,w) increases by 1 if u \in N(w).
        # This is true for all w \in N(u).
        # So we can just iterate w \in N(u) and do common[v][w] += 1.
        # We do NOT need to iterate w \in N(v) for updating common[v][w].
        
        # Similarly, C(u,w) increases by 1 if v \in N(w).
        # This is true for all w \in N(v).
        # So we can just iterate w \in N(v) and do common[u][w] += 1.
        # We do NOT need to iterate w \in N(u) for updating common[u][w].
        
        # Let's re-verify.
        # C(u,w) = |N(u) \cap N(w)|.
        # N(u) gains v.
        # Increases by 1 if v \in N(w).
        # v \in N(w) iff w \in N(v).
        # So for all w \in N(v), C(u,w) += 1.
        
        # So the correct updates are:
        # 1. common[u][v] += 1
        # 2. For all w \in N(u): common[v][w] += 1, common[w][v] += 1
        # 3. For all w \in N(v): common[u][w] += 1, common[w][u] += 1
        
        # Let's check double counting again with this logic.
        # If w \in N(u) \cap N(v):
        # Update 2: common[v][w] += 1.
        # Update 3: common[u][w] += 1.
        # No double counting of the same cell.
        # common[v][w] is updated in 2.
        # common[u][w] is updated in 3.
        # This is correct.
        
        # My previous code was:
        # Loop 1 (w in N(u)):
        #   common[v][w] += 1
        #   if v in N(w): common[u][w] += 1
        # Loop 2 (w in N(v)):
        #   common[u][w] += 1
        #   if u in N(w): common[v][w] += 1
        
        # If w in N(u) \cap N(v):
        # Loop 1: common[v][w] += 1. common[u][w] += 1 (since v in N(w)).
        # Loop 2: common[u][w] += 1. common[v][w] += 1 (since u in N(w)).
        # Double counted.
        
        # Correct logic:
        # common[v][w] += 1 for all w \in N(u).
        # common[u][w] += 1 for all w \in N(v).
        
        return True

    def remove(u, v):
        """
        Removes edge (u, v).
        """
        if not is_adjacent(u, v):
            return False
            
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        common[u][v] -= 1
        common[v][u] -= 1
        
        # Reverse of updates
        # common[v][w] -= 1 for all w \in N(u) (after removal, so current N(u) excluding v)
        # Note: adj[u] no longer has v. So iterating adj[u] gives N(u) \ {v}.
        # But the update was for w \in N(u) BEFORE removal?
        # Before removal, N(u) contained v? No, v was not in N(u).
        # So N(u) was the same set.
        # So we iterate current adj[u] (which is N(u) \ {v} ? No, v was not in N(u)).
        # Wait, adj[u] is updated BEFORE the loop in my draft?
        # Yes: `adj[u] &= ~(1 << v)`.
        # So adj[u] is N(u) \ {v}. But v was not in N(u). So adj[u] is N(u).
        # So iterating adj[u] is correct.
        
        temp_mask = adj[u]
        while temp_mask:
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            temp_mask &= temp_mask - 1
            
        temp_mask = adj[v]
        while temp_mask:
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            temp_mask &= temp_mask - 1
            
        return True

    def get_edge_count():
        return sum(bin(a).count('1') for a in adj) // 2

    # Phase 1: Greedy Fill
    random.shuffle(all_edges)
    for u, v in all_edges:
        if try_add(u, v):
            pass

    best_edges = []
    best_count = get_edge_count()
    
    # Save best state
    def save_state():
        return [a for a in adj], [row[:] for row in common]
    
    def load_state(state):
        nonlocal adj, common
        adj, common = state[0], state[1]
        
    best_adj, best_common = save_state()
    
    # Phase 2: Local Search
    # We will try to add edges. If stuck, remove one and try again.
    # To speed up, we maintain a list of missing edges.
    missing_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not is_adjacent(i, j)]
    
    # We need to update missing_edges when we add/remove.
    # For simplicity in the loop, we can just iterate all pairs or maintain the list.
    # Given N=20, iterating all 190 pairs is cheap.
    
    stuck_counter = 0
    max_stuck = 50
    
    while time.time() < deadline:
        added_any = False
        # Try to add any missing edge
        # We shuffle to avoid bias
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if is_adjacent(u, v):
                continue
            if try_add(u, v):
                added_any = True
                # Remove from missing_edges
                missing_edges.remove((u, v))
                # Update best if better
                curr_count = get_edge_count()
                if curr_count > best_count:
                    best_count = curr_count
                    best_adj, best_common = save_state()
        
        if not added_any:
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Do a bigger kick: remove 2-3 edges
                k = random.randint(2, 3)
                for _ in range(k):
                    if adj:
                        # Pick a random edge to remove
                        # Find a random adjacent pair
                        u = random.randint(0, n - 1)
                        if adj[u]:
                            v = random.choice([i for i in range(n) if (adj[u] >> i) & 1])
                            remove(u, v)
                            missing_edges.append((u, v))
                stuck_counter = 0
            else:
                # Remove 1 edge
                if adj:
                    u = random.randint(0, n - 1)
                    if adj[u]:
                        v = random.choice([i for i in range(n) if (adj[u] >> i) & 1])
                        remove(u, v)
                        missing_edges.append((u, v))
        else:
            stuck_counter = 0

    # Load best state
    adj, common = best_adj, best_common
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
