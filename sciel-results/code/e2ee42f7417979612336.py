"""Hexagon-free graph construction on 40 vertices using Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 40
TOTAL_PAIRS = N * (N - 1) // 2  # 780

def _build_bitmasks(edges):
    """Build adjacency bitmasks for fast C6 detection."""
    masks = [0] * N
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _has_c6_bitmask(masks, n):
    """
    Check for C6 using bitmasks.
    A C6 is a path u-v-w-x-y-z-u with distinct vertices.
    We iterate over potential starting edges (u,v) with u < v,
    then find paths of length 4 from v to u avoiding u and v.
    
    Path: v - w - x - y - z - u
    Constraints:
    w in N(v), w != u
    x in N(w), x not in {u, v, w}
    y in N(x), y not in {u, v, w, x}
    z in N(y), z not in {u, v, w, x, y} and z in N(u)
    """
    for u in range(n):
        for v in range(u + 1, n):
            if not (masks[u] & (1 << v)):
                continue
            
            # Neighbors of v, excluding u
            # We need w such that w is in N(v) and w != u
            # Let's iterate w
            # To speed up, we can precompute N(v) \ {u}
            # But iterating w is fine.
            
            # Optimization: If deg(v) is small, this is fast.
            # We can skip if deg(v) < 4 or deg(u) < 4 (need path of length 4)
            # Actually, we just check.
            
            # Bitmask for vertices allowed for w: N(v) & ~bit(u)
            allowed_w = masks[v] & ~(1 << u)
            
            while allowed_w:
                lsb_w = allowed_w & -allowed_w
                w = lsb_w.bit_length() - 1
                allowed_w ^= lsb_w
                
                # x in N(w), x not in {u, v, w}
                forbidden_x = (1 << u) | (1 << v) | lsb_w
                allowed_x = masks[w] & ~forbidden_x
                
                while allowed_x:
                    lsb_x = allowed_x & -allowed_x
                    x = lsb_x.bit_length() - 1
                    allowed_x ^= lsb_x
                    
                    # y in N(x), y not in {u, v, w, x}
                    forbidden_y = (1 << u) | (1 << v) | lsb_w | lsb_x
                    allowed_y = masks[x] & ~forbidden_y
                    
                    while allowed_y:
                        lsb_y = allowed_y & -allowed_y
                        y = lsb_y.bit_length() - 1
                        allowed_y ^= lsb_y
                        
                        # z in N(y), z not in {u, v, w, x, y} and z in N(u)
                        forbidden_z = (1 << u) | (1 << v) | lsb_w | lsb_x | lsb_y
                        allowed_z = masks[y] & ~forbidden_z
                        
                        # z must also be in N(u)
                        candidate_z = allowed_z & masks[u]
                        
                        if candidate_z:
                            return True
    return False

def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges

def _greedy_init():
    """Fast greedy initialization to get a good starting point."""
    random.seed(42)
    edges = _seed_graph_8K5()
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    masks = _build_bitmasks(edges)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if (u, v) in edge_set:
            continue
        # Check if adding creates C6
        # Temporary check
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        
        if not _has_c6_bitmask(masks, N):
            edge_set.add((u, v))
        else:
            masks[u] ^= (1 << v)
            masks[v] ^= (1 << u)
            
    return edge_set, masks, len(edge_set)

def _simulated_annealing(start_edges, start_masks, start_count, time_limit):
    """
    Simulated Annealing to maximize edges in a C6-free graph.
    Moves: Add a random edge, or remove a random edge.
    """
    current_edges = set(start_edges)
    current_masks = list(start_masks)
    current_count = start_count
    
    best_edges = set(current_edges)
    best_count = current_count
    best_masks = list(current_masks)
    
    start_time = time.time()
    T = 1.0
    T_min = 0.01
    alpha = 0.995  # Cooling rate
    
    # Precompute all pairs for random selection
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() - start_time < time_limit:
        # Generate a move
        if random.random() < 0.5 and current_count < TOTAL_PAIRS:
            # Try to add an edge
            # Pick a random non-edge
            # Efficiently: pick random pair, if edge, pick another?
            # Better: maintain list of non-edges? Too slow to update.
            # Just pick random pair, if edge, skip or pick another.
            # With ~700 edges out of 780, collision is high.
            # Let's pick a random pair.
            u, v = random.choice(all_pairs)
            if (u, v) in current_edges:
                continue
                
            # Add edge
            current_masks[u] |= (1 << v)
            current_masks[v] |= (1 << u)
            current_edges.add((u, v))
            current_count += 1
            
            # Check validity
            if _has_c6_bitmask(current_masks, N):
                # Reject if it creates C6 (always reject invalid moves in this constraint)
                current_masks[u] ^= (1 << v)
                current_masks[v] ^= (1 << u)
                current_edges.remove((u, v))
                current_count -= 1
            else:
                # Accept
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(current_edges)
                    best_masks = list(current_masks)
                    
        else:
            # Try to remove an edge
            if not current_edges:
                continue
            u, v = random.choice(list(current_edges))
            
            # Remove edge
            current_masks[u] ^= (1 << v)
            current_masks[v] ^= (1 << u)
            current_edges.remove((u, v))
            current_count -= 1
            
            # Always accept removal (it preserves C6-free property)
            # But this reduces count. We want to maximize.
            # So we only remove if it allows adding more later?
            # SA usually accepts worse moves with probability exp(-delta/T).
            # Here "worse" is lower edge count.
            # Let's use standard SA logic on energy = -edge_count.
            # Delta E = +1 (worse) when removing.
            # Accept with prob exp(-1/T).
            
            delta_e = 1.0  # Energy increases by 1 (count decreases by 1)
            if random.random() > np.exp(-delta_e / T):
                # Revert
                current_masks[u] |= (1 << v)
                current_masks[v] |= (1 << u)
                current_edges.add((u, v))
                current_count += 1
        
        # Cool down
        T = max(T * alpha, T_min)
        
    return best_edges, best_masks, best_count

def construct():
    n = N
    deadline = 140  # Safety margin before 150s kill
    start_time = time.time()
    
    # Phase 1: Greedy Init
    edge_set, masks, count = _greedy_init()
    
    # Phase 2: Simulated Annealing
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5:
        edge_set, masks, count = _simulated_annealing(edge_set, masks, count, remaining_time)
    
    # Convert to list of lists
    edges = [[u, v] for u, v in edge_set]
    
    # Final Verification
    if _has_c6_bitmask(masks, n):
        # Fallback
        return _seed_graph_8K5()
        
    return edges
# EVOLVE-BLOCK-END
