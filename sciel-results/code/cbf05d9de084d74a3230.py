import time
import random
import sys
import math

N = 128
DEADLINE_MARGIN = 5.0

def _build_edges_bitmasks():
    edges = []
    edge_masks = []
    edge_sizes = []
    vert_to_edges = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                
                vertices = [a, b, c, s1, s2, s3]
                mask = 0
                unique_verts = set(vertices)
                for v in unique_verts:
                    mask |= (1 << v)
                
                edges.append(unique_verts)
                edge_masks.append(mask)
                edge_sizes.append(len(unique_verts))
                
                edge_idx = len(edge_masks) - 1
                for v in unique_verts:
                    vert_to_edges[v].append(edge_idx)
                    
    return edge_masks, edge_sizes, vert_to_edges

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    edge_masks, edge_sizes, vert_to_edges = _build_edges_bitmasks()
    num_edges = len(edge_masks)
    
    bits = [1 << i for i in range(N + 1)]
    all_vertices = list(range(1, N + 1))

    best_mask = 0
    best_size = 0
    best_counts = [0] * num_edges

    def _greedy_start():
        mask = 0
        counts = [0] * num_edges
        for v in all_vertices:
            can_add = True
            for ei in vert_to_edges[v]:
                if counts[ei] == edge_sizes[ei] - 1:
                    can_add = False
                    break
            if can_add:
                mask |= bits[v]
                for ei in vert_to_edges[v]:
                    counts[ei] += 1
        return mask, counts

    best_mask, best_counts = _greedy_start()
    best_size = bin(best_mask).count('1')

    def _sa_search(initial_mask, initial_counts):
        nonlocal best_mask, best_size, best_counts
        
        cur_mask = initial_mask
        cur_counts = initial_counts[:]
        cur_violations = 0
        cur_size = bin(initial_mask).count('1')
        
        T = 5.0
        T_min = 0.01
        decay = 0.995
        lambda_pen = 10.0
        
        sa_best_mask = cur_mask
        sa_best_size = cur_size
        sa_best_counts = cur_counts[:]
        
        while T > T_min and time.time() < deadline:
            move_type = random.random()
            
            if move_type < 0.5:
                # Add
                missing = [v for v in all_vertices if not (cur_mask & bits[v])]
                if not missing: continue
                v = random.choice(missing)
                
                delta_viol = 0
                for ei in vert_to_edges[v]:
                    if cur_counts[ei] == edge_sizes[ei] - 1:
                        delta_viol += 1
                
                delta_score = 1 - lambda_pen * delta_viol
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    cur_mask |= bits[v]
                    for ei in vert_to_edges[v]:
                        cur_counts[ei] += 1
                    cur_violations += delta_viol
                    cur_size += 1
                    
                    if cur_violations == 0 and cur_size > sa_best_size:
                        sa_best_size = cur_size
                        sa_best_mask = cur_mask
                        sa_best_counts = cur_counts[:]
                        
            else:
                # Remove
                present = [v for v in all_vertices if (cur_mask & bits[v])]
                if not present: continue
                v = random.choice(present)
                
                delta_viol = 0
                for ei in vert_to_edges[v]:
                    if cur_counts[ei] == edge_sizes[ei]:
                        delta_viol -= 1
                        
                delta_score = -1 - lambda_pen * delta_viol
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    cur_mask &= ~bits[v]
                    for ei in vert_to_edges[v]:
                        cur_counts[ei] -= 1
                    cur_violations += delta_viol
                    cur_size -= 1
                    
                    if cur_violations == 0 and cur_size > sa_best_size:
                        sa_best_size = cur_size
                        sa_best_mask = cur_mask
                        sa_best_counts = cur_counts[:]
                        
            T *= decay
            
            if cur_violations == 0 and cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
                best_counts = cur_counts[:]
                
        return sa_best_mask, sa_best_size, sa_best_counts

    for i in range(100):
        if time.time() > deadline: break
        
        cur_mask = best_mask
        cur_counts = best_counts[:]
        
        # Perturb
        k = random.randint(2, 5)
        present = [v for v in all_vertices if (cur_mask & bits[v])]
        if len(present) > k:
            to_remove = random.sample(present, k)
            for v in to_remove:
                cur_mask &= ~bits[v]
                for ei in vert_to_edges[v]:
                    cur_counts[ei] -= 1
        
        sa_mask, sa_size, sa_counts = _sa_search(cur_mask, cur_counts)
        
        if sa_size > best_size:
            best_size = sa_size
            best_mask = sa_mask
            best_counts = sa_counts

    # Final Polish
    cur_mask = best_mask
    cur_counts = [0] * num_edges
    for v in all_vertices:
        if cur_mask & bits[v]:
            for ei in vert_to_edges[v]:
                cur_counts[ei] += 1
                
    for v in all_vertices:
        if not (cur_mask & bits[v]):
            can_add = True
            for ei in vert_to_edges[v]:
                if cur_counts[ei] == edge_sizes[ei] - 1:
                    can_add = False
                    break
            if can_add:
                cur_mask |= bits[v]
                for ei in vert_to_edges[v]:
                    cur_counts[ei] += 1
                
    if bin(cur_mask).count('1') > best_size:
        best_size = bin(cur_mask).count('1')
        best_mask = cur_mask
        
    result = []
    for v in all_vertices:
        if best_mask & bits[v]:
            result.append(v)
            
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
