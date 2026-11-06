import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # 5s margin before 150s limit
    
    # Precompute all K4s and their edge indices
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
    
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
    
    k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        e = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4_edges.append(e)
    
    # Known good solution with 17 colors
    def get_17_sol():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    best_sol = get_17_sol()
    best_colors = 17
    
    # Try to find solutions with fewer colors
    for C in [14, 15, 16]:
        if time.time() > deadline - 15:
            break
        
        found = find_solution(C, start_time, deadline)
        if found is not None:
            if is_valid(found):
                best_colors = C
                best_sol = found
                break
    
    return best_sol

def is_valid(colors):
    """Check if the coloring is valid (every K4 has >= 5 distinct colors)."""
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
    
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
    
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        c1 = colors[edge_index[(a, b)]]
        c2 = colors[edge_index[(a, c)]]
        c3 = colors[edge_index[(a, d)]]
        c4 = colors[edge_index[(b, c)]]
        c5 = colors[edge_index[(b, d)]]
        c6 = colors[edge_index[(c, d)]]
        if len({c1, c2, c3, c4, c5, c6}) < 5:
            return False
    return True

def find_solution(C, start_time, deadline):
    """Find a valid coloring with C colors using local search."""
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
    
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
    
    k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        e = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4_edges.append(e)
    
    n_k4 = len(k4_edges)
    
    def count_violations(cur_colors):
        """Count number of K4s with < 5 distinct colors."""
        viol = 0
        for e in k4_edges:
            cols = [cur_colors[e[0]], cur_colors[e[1]], cur_colors[e[2]], 
                    cur_colors[e[3]], cur_colors[e[4]], cur_colors[e[5]]]
            if len(set(cols)) < 5:
                viol += 1
        return viol
    
    def get_violating_k4s(cur_colors):
        """Get list of indices of violating K4s."""
        viol_idx = []
        for i, e in enumerate(k4_edges):
            cols = [cur_colors[e[0]], cur_colors[e[1]], cur_colors[e[2]], 
                    cur_colors[e[3]], cur_colors[e[4]], cur_colors[e[5]]]
            if len(set(cols)) < 5:
                viol_idx.append(i)
        return viol_idx
    
    # Try multiple restarts with different initializations
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Random initialization
            cur_colors = [random.randint(0, C - 1) for _ in range(120)]
        elif restart == 1:
            # Modular initialization
            cur_colors = [(i + j) % C for i in range(16) for j in range(i + 1, 16)]
        elif restart == 2:
            # Another modular pattern
            cur_colors = [(i * 3 + j) % C for i in range(16) for j in range(i + 1, 16)]
        elif restart == 3:
            # Mixed pattern
            cur_colors = [(i + 2*j) % C for i in range(16) for j in range(i + 1, 16)]
        else:
            # Random with bias
            cur_colors = [random.randint(0, min(C-1, 5)) for _ in range(120)]
        
        # Check if initial solution is valid
        viol = count_violations(cur_colors)
        if viol == 0:
            return cur_colors[:]
        
        # Local search
        max_iters = 20000
        no_improve_count = 0
        best_viol = viol
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Count violations
            viol = count_violations(cur_colors)
            
            if viol == 0:
                return cur_colors[:]
            
            if viol < best_viol:
                best_viol = viol
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # If stuck for too long, restart
            if no_improve_count > 5000:
                break
            
            # Get a random violating K4
            viol_idx = get_violating_k4s(cur_colors)
            if not viol_idx:
                break
            
            k4_idx = random.choice(viol_idx)
            e = k4_edges[k4_idx]
            
            # Try to fix this K4 by changing one of its edges
            # Find which color changes would help
            current_cols = [cur_colors[e[k]] for k in range(6)]
            distinct_cols = set(current_cols)
            
            # We need at least 5 distinct colors
            if len(distinct_cols) >= 5:
                continue
            
            # Try changing each edge in the K4
            for edge_pos in random.sample(range(6), 6):
                global_edge_idx = e[edge_pos]
                old_color = cur_colors[global_edge_idx]
                
                # Try all possible new colors
                for new_color in range(C):
                    if new_color == old_color:
                        continue
                    
                    # Check if this change helps
                    cur_colors[global_edge_idx] = new_color
                    
                    # Check if this K4 is now fixed
                    new_cols = [cur_colors[e[k]] for k in range(6)]
                    if len(set(new_cols)) >= 5:
                        # Good change, keep it
                        break
                    else:
                        # Revert
                        cur_colors[global_edge_idx] = old_color
                else:
                    # No good color found for this edge, try next
                    continue
                break
            else:
                # No edge change helped, try random change
                edge_pos = random.randint(0, 5)
                global_edge_idx = e[edge_pos]
                new_color = random.randint(0, C - 1)
                if new_color == cur_colors[global_edge_idx]:
                    new_color = (new_color + 1) % C
                cur_colors[global_edge_idx] = new_color
    
    return None
