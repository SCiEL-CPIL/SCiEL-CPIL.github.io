"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _topo_order(n, preds):
    """Kahn's algorithm; returns list of tasks in topological order and successors."""
    indeg = np.zeros(n, dtype=int)
    succs = [[] for _ in range(n)]
    for i, ps in enumerate(preds):
        for p in ps:
            succs[p].append(i)
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:
        remaining = [i for i in range(n) if i not in set(order)]
        order.extend(remaining)
    return order, succs

def _simulate_greedy_optimized(times, preds, succs, ct, priorities):
    n = len(times)
    assigned = [False] * n
    unassigned_preds = [len(preds[i]) for i in range(n)]
    
    stations = 0
    current_load = 0.0
    remaining = n
    
    while remaining > 0:
        best_task = -1
        best_priority = -np.inf
        
        for i in range(n):
            if not assigned[i] and unassigned_preds[i] == 0:
                if times[i] <= ct - current_load:
                    if priorities[i] > best_priority:
                        best_priority = priorities[i]
                        best_task = i
        
        if best_task == -1:
            if current_load > 0:
                stations += 1
                current_load = 0.0
            else:
                # Safety break for invalid DAGs or deadlock
                break
        else:
            assigned[best_task] = True
            current_load += times[best_task]
            remaining -= 1
            
            for s in succs[best_task]:
                unassigned_preds[s] -= 1
                
    return stations

def _compute_features(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    order, succs = _topo_order(n, preds)
    
    # 1. Critical Path (Longest path from i to any terminal, including i)
    cp = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succs[i]:
            if cp[s] > best:
                best = cp[s]
        cp[i] = times[i] + best
        
    # 2. Depth from Start (Longest path from any source to i, including i)
    depth = np.zeros(n, dtype=float)
    for i in order:
        best = 0.0
        for p in preds[i]:
            if depth[p] > best:
                best = depth[p]
        depth[i] = best + times[i]
        
    # 3. Transitive Successor Count
    trans_succ_count = np.zeros(n, dtype=int)
    for i in reversed(order):
        c = 0
        for s in succs[i]:
            c += 1 + trans_succ_count[s]
        trans_succ_count[i] = c
        
    # 4. Direct Successor Count
    direct_succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # 5. Direct Predecessor Count
    direct_pred_count = np.array([len(preds[i]) for i in range(n)], dtype=float)
    
    # 6. Time
    time_feat = times.copy()
    
    # 7. Slack-based fit score: How well does the task fit into a typical remaining capacity?
    # Approximate: tasks with time close to ct/2 or ct/3 are "efficient" packers.
    # Or simply: time itself is a feature for LPT.
    # Let's add a "fit" feature: 1 - |time - ct/2|/ct, rewarding tasks that fill half the cycle.
    # Actually, simpler: just use time and CP. The local search will find the right combo.
    
    features = {
        'cp': cp,
        'depth': depth,
        'trans_succ': trans_succ_count.astype(float),
        'direct_succ': direct_succ_count,
        'direct_pred': direct_pred_count,
        'time': time_feat
    }
    
    return features, order, succs

def _norm(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def priority(instance):
    t0 = time.time()
    deadline = 140.0 # Leave 10s margin for 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    ct = float(instance["cycle_time"])
    
    features, order, succs = _compute_features(instance)
    
    # Normalize features
    norm_features = {k: _norm(v) for k, v in features.items()}
    
    # Define feature vector for linear combination
    # Order matters for weight vector indexing
    feat_names = ['cp', 'depth', 'trans_succ', 'direct_succ', 'direct_pred', 'time']
    feat_matrix = np.vstack([norm_features[name] for name in feat_names]).T # n x 6
    
    # Initial weights: Heuristic guess
    # CP is usually strong. Time is strong.
    # Let's start with a mix.
    w0 = np.array([0.4, 0.1, 0.1, 0.1, 0.0, 0.3])
    w0 = w0 / w0.sum()
    
    current_scores = feat_matrix @ w0
    current_stations = _simulate_greedy_optimized(times, preds, succs, ct, current_scores)
    
    best_scores = current_scores.copy()
    best_stations = current_stations
    
    # Local Search: Hill Climbing with Simulated Annealing-like acceptance
    # We optimize the weight vector w (sum to 1, non-negative)
    
    # Strategy:
    # 1. Random restarts of weights? No, just perturb.
    # 2. Step size decay.
    
    step_size = 0.2
    no_improve_count = 0
    max_no_improve = 2000
    T = 1.0 # Temperature for acceptance
    cooling_rate = 0.999
    
    # Pre-compute simulation overhead check
    # Simulation is O(N^2) roughly. N=30. Very fast.
    
    while time.time() - t0 < deadline:
        if no_improve_count > max_no_improve:
            # If stuck, try a larger jump or restart weights
            if step_size < 0.01:
                break
            # Reset no_improve and increase step size? Or just keep going with small steps.
            # Let's try a random restart of weights around the current best
            # Actually, let's just reduce step size and continue.
            step_size *= 0.5
            no_improve_count = 0
            continue
            
        # Generate new weights
        new_w = w0.copy()
        # Pick 1 or 2 indices to perturb
        k = random.randint(1, 2)
        indices = random.sample(range(len(feat_names)), k)
        for idx in indices:
            delta = random.gauss(0, step_size)
            new_w[idx] += delta
        
        # Normalize
        new_w = np.maximum(new_w, 0)
        s = new_w.sum()
        if s < 1e-6:
            new_w = w0.copy()
        else:
            new_w = new_w / s
            
        new_scores = feat_matrix @ new_w
        new_stations = _simulate_greedy_optimized(times, preds, succs, ct, new_scores)
        
        if new_stations < best_stations:
            best_stations = new_stations
            best_scores = new_scores.copy()
            w0 = new_w.copy()
            no_improve_count = 0
            # Reset temperature?
            T = 1.0
        elif new_stations < current_stations:
            # Better than current, accept
            current_stations = new_stations
            w0 = new_w.copy()
            current_scores = new_scores
            no_improve_count = 0
        else:
            # Worse or equal
            # Accept with probability exp(-delta / T)
            if new_stations == current_stations:
                # Tie: accept with some probability to explore
                if random.random() < 0.1:
                    w0 = new_w.copy()
                    current_scores = new_scores
                else:
                    no_improve_count += 1
            else:
                delta = new_stations - current_stations
                # delta >= 1
                p = np.exp(-delta / T)
                if random.random() < p:
                    w0 = new_w.copy()
                    current_scores = new_scores
                    current_stations = new_stations
                else:
                    no_improve_count += 1
                    
        # Cool down
        T *= cooling_rate
        
        # Decay step size slightly
        step_size *= 0.999
        if step_size < 0.001:
            step_size = 0.001
            
    return np.asarray(best_scores, dtype=float)

# EVOLVE-BLOCK-END
