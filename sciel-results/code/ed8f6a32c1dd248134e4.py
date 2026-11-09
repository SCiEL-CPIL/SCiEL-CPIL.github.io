import time
from collections import deque
import numpy as np

_DEADLINE = 110.0

def priority(instance):
    t0 = time.time()
    try:
        times = np.asarray(instance["times"], dtype=float)
        n = int(len(times))
        C = float(instance.get("cycle_time", 40.0))
    except Exception:
        return np.zeros(0, dtype=float)

    def _lpt():
        return np.asarray(times, dtype=float) + np.arange(n, dtype=float) * 1e-9

    try:
        preds_raw = instance["predecessors"]
        preds = []
        for i in range(n):
            p = preds_raw[i]
            if p is None:
                preds.append([])
            elif isinstance(p, (list, tuple, np.ndarray)):
                preds.append([int(x) for x in p])
            else:
                preds.append([])

        successors = [[] for _ in range(n)]
        for i in range(n):
            for p in preds[i]:
                if 0 <= p < n and p != i:
                    successors[p].append(i)

        indeg = [0] * n
        for i in range(n):
            for p in preds[i]:
                if 0 <= p < n and p != i:
                    indeg[i] += 1

        queue = deque([i for i in range(n) if indeg[i] == 0])
        topo = []
        indeg2 = indeg[:]
        while queue:
            u = queue.popleft()
            topo.append(u)
            for v in successors[u]:
                indeg2[v] -= 1
                if indeg2[v] == 0:
                    queue.append(v)
        if len(topo) < n:
            seen = set(topo)
            for i in range(n):
                if i not in seen:
                    topo.append(i)

        # Pre-compute features
        forward_rank = np.zeros(n, dtype=float)
        for u in reversed(topo):
            fr = times[u]
            for v in successors[u]:
                cand = times[u] + forward_rank[v]
                if cand > fr:
                    fr = cand
            forward_rank[u] = fr

        backward_rank = np.zeros(n, dtype=float)
        for u in topo:
            br = times[u]
            for p in preds[u]:
                if 0 <= p < n and p != u:
                    cand = backward_rank[p] + times[u]
                    if cand > br:
                        br = cand
            backward_rank[u] = br

        desc_mask = [0] * n
        for u in reversed(topo):
            m = 0
            for v in successors[u]:
                m |= (1 << v) | desc_mask[v]
            desc_mask[u] = m
        desc_count = np.array([bin(m).count("1") for m in desc_mask], dtype=float)

        subtree_time = np.zeros(n, dtype=float)
        for u in reversed(topo):
            st = times[u]
            for v in successors[u]:
                st += subtree_time[v]
            subtree_time[u] = st

        min_stations = np.ceil(subtree_time / C)
        min_stations_safe = np.where(min_stations == 0, 1.0, min_stations)
        density = times / min_stations_safe

        # Candidate weights: (w_rank, w_back, w_desc, w_time, w_density)
        # We will generate a list of candidate priority vectors
        candidates = []
        
        # Base heuristics
        candidates.append(forward_rank)
        candidates.append(forward_rank + times)
        candidates.append(forward_rank + desc_count)
        candidates.append(forward_rank + backward_rank)
        candidates.append(forward_rank + density)
        candidates.append(forward_rank + 0.5 * density + 0.1 * desc_count)
        candidates.append(forward_rank + 0.1 * backward_rank + 0.5 * density)
        candidates.append(times) # LPT
        candidates.append(desc_count)
        
        # More mixes
        candidates.append(forward_rank * 2.0 + density)
        candidates.append(forward_rank + density * 2.0)
        candidates.append(forward_rank + times * 0.5 + density * 0.5)
        
        # Add small noise to break ties uniquely
        offset = np.arange(n, dtype=float) * 1e-9
        for i in range(len(candidates)):
            candidates[i] = candidates[i] + offset

        # Simulation function
        def simulate(scores):
            # scores: numpy array of priorities
            # Returns: number of stations
            # Greedy: sort by score descending. Assign to first station that fits.
            # Eligible: all preds assigned.
            
            # Map task index to priority rank
            # We need to process tasks in order of priority.
            # But we must respect precedence.
            
            # Efficient simulation:
            # 1. Create list of tasks sorted by priority (desc).
            # 2. Iterate through this list.
            # 3. For each task, check if all preds are assigned.
            # 4. If yes, try to assign to existing stations (in order of opening) if it fits.
            # 5. If no existing station fits, open new station.
            # 6. If preds not assigned, skip? 
            #    Wait, the standard greedy ALB algorithm is:
            #    - Maintain a set of available tasks (preds done).
            #    - Pick highest priority available task.
            #    - Assign to current station if fits.
            #    - If not fits, close current station, open new, assign there.
            #    - Repeat.
            
            # This "pick highest available" is different from "sort all and filter".
            # The prompt says: "stations are filled with the highest-priority eligible task that fits the cycle time."
            # This implies:
            # While tasks remain:
            #   Find eligible tasks (preds done).
            #   Among eligible, find highest priority.
            #   If it fits in current station: assign, update station time.
            #   Else: close station, open new, assign.
            
            assigned = [False] * n
            station_time = 0.0
            stations = 0
            remaining = n
            
            while remaining > 0:
                # Find eligible
                eligible = []
                for i in range(n):
                    if not assigned[i]:
                        # Check preds
                        ok = True
                        for p in preds[i]:
                            if not assigned[p]:
                                ok = False
                                break
                        if ok:
                            eligible.append(i)
                
                if not eligible:
                    # Deadlock? Should not happen if graph is valid.
                    # Force pick one?
                    for i in range(n):
                        if not assigned[i]:
                            eligible.append(i)
                            break
                
                # Find highest priority eligible
                # scores are higher = better
                best_task = -1
                best_score = -1e9
                for i in eligible:
                    if scores[i] > best_score:
                        best_score = scores[i]
                        best_task = i
                
                if best_task == -1:
                    break # Error
                    
                # Try to assign
                if station_time + times[best_task] <= C:
                    station_time += times[best_task]
                    assigned[best_task] = True
                    remaining -= 1
                else:
                    # Close current, open new
                    if station_time > 0:
                        stations += 1
                    station_time = times[best_task]
                    assigned[best_task] = True
                    remaining -= 1
            
            if station_time > 0:
                stations += 1
                
            return stations

        best_scores = None
        best_stations = float('inf')
        
        for scores in candidates:
            if time.time() - t0 > _DEADLINE - 5.0:
                break
            st = simulate(scores)
            if st < best_stations:
                best_stations = st
                best_scores = scores

        if best_scores is None:
            return _lpt()
            
        return best_scores

    except Exception:
        return _lpt()
