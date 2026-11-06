import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_alpha = 16
    best_edges = []
    
    # Helper to check linearity
    def is_linear(edge, existing_edges):
        # edge is a tuple or int mask
        # existing_edges is list of int masks
        # We need to check intersection size <= 1
        # If we use masks, intersection size is bit_count of (edge_mask & e)
        # But we need to check if intersection is exactly 2? No, >1.
        # So if bit_count(edge_mask & e) > 1, it's not linear.
        pass

    # Helper to get alpha
    def get_alpha(edges_masks):
        # edges_masks is a list of ints
        if not edges_masks:
            return 16
        
        # Check from largest k downwards
        for k in range(16, 0, -1):
            found = False
            for combo in combinations(range(16), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Check if any edge is subset of mask
                is_independent = True
                for e in edges_masks:
                    if (mask & e) == e:
                        is_independent = False
                        break
                
                if is_independent:
                    found = True
                    break
            
            if found:
                return k
        
        return 0

    # Helper to add edge
    def can_add_edge(edge_mask, edges_masks):
        for e in edges_masks:
            inter = edge_mask & e
            # Count bits in inter
            if inter & (inter - 1): # If more than 1 bit is set
                return False
        return True

    # Random initialization
    def random_saturated():
        edges = []
        # Try to add random edges
        attempts = 0
        while attempts < 500:
            attempts += 1
            a, b, c = random.sample(range(16), 3)
            mask = (1 << a) | (1 << b) | (1 << c)
            if can_add_edge(mask, edges):
                edges.append(mask)
        return edges

    # Local Search
    def local_search(edges):
        current_alpha = get_alpha(edges)
        current_edges = edges[:]
        
        while time.time() < deadline:
            # Pick a move
            move_type = random.choice(['add', 'swap', 'flip'])
            
            new_edges = current_edges[:]
            improved = False
            
            if move_type == 'add':
                # Try to add a random edge
                for _ in range(10): # Try a few
                    a, b, c = random.sample(range(16), 3)
                    mask = (1 << a) | (1 << b) | (1 << c)
                    if can_add_edge(mask, new_edges):
                        new_edges.append(mask)
                        new_alpha = get_alpha(new_edges)
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            current_edges = new_edges
                            improved = True
                        break
            elif move_type == 'swap':
                # Remove one, add one
                if not current_edges: continue
                idx = random.randint(0, len(current_edges) - 1)
                removed = current_edges[idx]
                new_edges.pop(idx)
                
                # Try to add a new edge
                for _ in range(10):
                    a, b, c = random.sample(range(16), 3)
                    mask = (1 << a) | (1 << b) | (1 << c)
                    if can_add_edge(mask, new_edges):
                        new_edges.append(mask)
                        new_alpha = get_alpha(new_edges)
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            current_edges = new_edges
                            improved = True
                        break
            elif move_type == 'flip':
                # Change one vertex in a random edge
                if not current_edges: continue
                idx = random.randint(0, len(current_edges) - 1)
                old_edge = current_edges[idx]
                
                # Get vertices of old edge
                verts = []
                m = old_edge
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m ^= lsb
                
                # Pick a vertex to change
                v_to_change = random.choice(verts)
                new_v = random.choice([v for v in range(16) if v != v_to_change and v not in verts])
                
                # Construct new edge
                new_verts = [v for v in verts if v != v_to_change] + [new_v]
                new_mask = 0
                for v in new_verts:
                    new_mask |= (1 << v)
                
                # Check if new edge is valid with others
                # Remove old edge temporarily
                new_edges.pop(idx)
                if can_add_edge(new_mask, new_edges):
                    new_edges.append(new_mask)
                    new_alpha = get_alpha(new_edges)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        current_edges = new_edges
                        improved = True
                    else:
                        # Revert
                        new_edges.pop()
                        new_edges.append(old_edge)
                else:
                    # Revert
                    new_edges.append(old_edge)
            
            if not improved:
                # Maybe do a random restart or accept worse solution with low prob?
                # For now, just keep trying.
                pass
        
        return current_alpha, current_edges

    # Main loop
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        init_edges = random_saturated()
        alpha, edges = local_search(init_edges)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges[:]
            
        # If we reached a very low alpha, we can stop early?
        # Theoretical min is likely 5 or 6.
        if best_alpha <= 5:
            break

    # Convert best_edges to list of lists
    result = []
    for e in best_edges:
        verts = []
        m = e
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        result.append(verts)
        
    return result
