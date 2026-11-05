import time
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    if not matrix or not matrix[0]:
        return {'gates': [], 'outputs': []}
        
    n = len(matrix[0])
    m = len(matrix)
    
    # Convert rows to bitmasks
    targets_mask = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        targets_mask.append(mask)
        
    # Wire management
    # wire_id -> form (bitmask)
    wire_forms = [1 << i for i in range(n)]
    wire_forms.append(0) # Zero wire
    wire_depth = [0] * (n + 1)
    form_to_wire = {f: i for i, f in enumerate(wire_forms)}
    current_wire_count = n + 1
    gates = []
    
    # Current targets as sets of wire IDs
    current_targets = []
    for mask in targets_mask:
        t_set = set()
        for i in range(n):
            if (mask >> i) & 1:
                t_set.add(i)
        if not t_set:
            t_set.add(n) # Zero wire
        current_targets.append(t_set)
        
    active_targets = [i for i in range(m) if len(current_targets[i]) > 1]
    
    while active_targets:
        if time.time() - start_time > timeout:
            break
            
        # Count pairs in active targets
        pair_counts = defaultdict(int)
        for t_idx in active_targets:
            t_set = current_targets[t_idx]
            if len(t_set) < 2:
                continue
            wires_in_t = list(t_set)
            for i in range(len(wires_in_t)):
                for j in range(i+1, len(wires_in_t)):
                    u, v = wires_in_t[i], wires_in_t[j]
                    if u < v:
                        pair_counts[(u, v)] += 1
                    else:
                        pair_counts[(v, u)] += 1
                        
        if not pair_counts:
            break
            
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        
        # Tie-breaking: minimal depth
        best_candidate = None
        min_depth = float('inf')
        for u, v in candidates:
            d = max(wire_depth[u], wire_depth[v]) + 1
            if d < min_depth:
                min_depth = d
                best_candidate = (u, v)
                
        if best_candidate is None:
            break
            
        u, v = best_candidate
        new_depth = min_depth
        
        form_u = wire_forms[u]
        form_v = wire_forms[v]
        new_form = form_u ^ form_v
        
        if new_form in form_to_wire:
            w_id = form_to_wire[new_form]
            new_active_targets = []
            for t_idx in active_targets:
                t_set = current_targets[t_idx]
                if u in t_set and v in t_set:
                    t_set.remove(u)
                    t_set.remove(v)
                    t_set.add(w_id)
                    if len(t_set) > 1:
                        new_active_targets.append(t_idx)
                else:
                    new_active_targets.append(t_idx)
            active_targets = new_active_targets
            continue
            
        # Create new wire
        w_id = current_wire_count
        current_wire_count += 1
        wire_forms.append(new_form)
        wire_depth.append(new_depth)
        form_to_wire[new_form] = w_id
        gates.append([u, v])
        
        new_active_targets = []
        for t_idx in active_targets:
            t_set = current_targets[t_idx]
            if u in t_set and v in t_set:
                t_set.remove(u)
                t_set.remove(v)
                t_set.add(w_id)
                if len(t_set) > 1:
                    new_active_targets.append(t_idx)
            else:
                new_active_targets.append(t_idx)
        active_targets = new_active_targets
        
    outputs = []
    for i in range(m):
        t_set = current_targets[i]
        if len(t_set) == 1:
            outputs.append(list(t_set)[0])
        else:
            # Fallback: XOR remaining wires
            if t_set:
                wires = sorted(t_set)
                res = wires[0]
                for w in wires[1:]:
                    form_res = wire_forms[res]
                    form_w = wire_forms[w]
                    new_form = form_res ^ form_w
                    if new_form in form_to_wire:
                        res = form_to_wire[new_form]
                    else:
                        new_id = current_wire_count
                        current_wire_count += 1
                        wire_forms.append(new_form)
                        wire_depth.append(max(wire_depth[res], wire_depth[w]) + 1)
                        form_to_wire[new_form] = new_id
                        gates.append([res, w])
                        res = new_id
                outputs.append(res)
            else:
                outputs.append(n)
                
    return {'gates': gates, 'outputs': outputs}
