import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute all possible edges as (u,v) pairs with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)

    def calc_score(adj, degs):
        # adj is a list of integers (bitmasks)
        # degs is a list of degrees
        max_sum = 0
        # Iterate over all pairs u < v
        for u in range(n):
            if not adj[u]: continue
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # u and v are connected
                    common = adj[u] & adj[v]
                    # Iterate over common neighbors w
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    def generate_baseline():
        # K_{8,8} plus edge (0,1)
        adj = [0] * n
        degs = [0] * n
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        degs[0] += 1
        degs[1] += 1
        return adj, degs

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def random_swap_adj(adj):
        """Pick a random edge to remove and random non-edge to add"""
        # Build lists of current edges and non-edges
        edge_list = []
        nonedge_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edge_list.append((u, v))
                else:
                    nonedge_list.append((u, v))
        
        if not edge_list or not nonedge_list:
            return None, None
        
        e_rm = random.choice(edge_list)
        e_add = random.choice(nonedge_list)
        return e_rm, e_add

    def apply_swap(adj, degs, e_rm, e_add):
        """Apply a swap: remove e_rm, add e_add"""
        u1, v1 = e_rm
        u2, v2 = e_add
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        degs[u1] -= 1
        degs[v1] -= 1
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        degs[u2] += 1
        degs[v2] += 1

    def revert_swap(adj, degs, e_rm, e_add):
        """Revert a swap: re-add e_rm, remove e_add"""
        u1, v1 = e_rm
        u2, v2 = e_add
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        degs[u1] += 1
        degs[v1] += 1
        adj[u2] &= ~(1 << v2)
        adj[v2] &= ~(1 << u2)
        degs[u2] -= 1
        degs[v2] -= 1

    best_adj, best_degs = generate_baseline()
    best_score = calc_score(best_adj, best_degs)
    
    # Simulated annealing
    current_adj = best_adj[:]
    current_degs = best_degs[:]
    current_score = best_score
    
    initial_temp = 5.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    step = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        step += 1
        
        # Adaptive temperature based on progress
        if no_improve > max_no_improve:
            # Restart with perturbation
            current_adj = best_adj[:]
            current_degs = best_degs[:]
            current_score = best_score
            no_improve = 0
            # Apply several random swaps to kick out of local minimum
            n_kicks = random.randint(5, 15)
            for _ in range(n_kicks):
                if time.time() > deadline: break
                e_rm, e_add = random_swap_adj(current_adj)
                if e_rm is None: break
                apply_swap(current_adj, current_degs, e_rm, e_add)
            current_score = calc_score(current_adj, current_degs)
            # Reset temperature
            initial_temp = max(initial_temp * 0.5, 1.0)
            continue
        
        e_rm, e_add = random_swap_adj(current_adj)
        if e_rm is None:
            continue
        
        # Apply swap
        apply_swap(current_adj, current_degs, e_rm, e_add)
        
        new_score = calc_score(current_adj, current_degs)
        
        delta = new_score - current_score
        
        # Determine temperature
        # Use a cooling schedule
        temp = max(initial_temp * (cooling_rate ** step), min_temp)
        
        if delta <= 0:
            # Accept improving or equal move
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
                best_degs = current_degs[:]
        else:
            # Accept worse move with probability exp(-delta/temp)
            if random.random() < math.exp(-delta / temp):
                current_score = new_score
                no_improve = 0
            else:
                # Revert
                revert_swap(current_adj, current_degs, e_rm, e_add)
                no_improve += 1
    
    return {'n': n, 'edges': get_edges(best_adj)}
