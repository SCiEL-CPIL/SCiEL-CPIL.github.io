import numpy as np
from math import ceil
import itertools

def load(boxes, container):
    W, H, D = container
    
    if not boxes:
        return []
        
    # Generate candidate slab heights from box dimensions
    all_dims = set()
    for w, h, d in boxes:
        all_dims.add(w)
        all_dims.add(h)
        all_dims.add(d)
    
    candidate_heights = set(all_dims)
    candidate_heights.add(H)
    for i in range(2, 6):
        if H % i == 0:
            candidate_heights.add(H // i)
    
    candidate_heights = [h for h in candidate_heights if 0 < h <= H]
    
    best_placements = None
    best_volume = -1
    
    box_perms = []
    for w, h, d in boxes:
        perms = set(itertools.permutations((w, h, d)))
        box_perms.append(list(perms))
        
    for slab_h in sorted(candidate_heights):
        if slab_h <= 0 or slab_h > H:
            continue
            
        n_slabs = ceil(H / slab_h)
        if n_slabs == 0:
            continue
        
        current_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(len(boxes))]
        total_volume = 0
        placed_boxes = set()
        
        for slab_idx in range(n_slabs):
            y_start = slab_idx * slab_h
            y_end = min(y_start + slab_h, H)
            actual_slab_h = y_end - y_start
            
            if actual_slab_h <= 0:
                continue
                
            valid_rotations = []
            for i in range(len(boxes)):
                if i in placed_boxes:
                    continue
                for perm in box_perms[i]:
                    a, b, c = perm
                    if b <= actual_slab_h:
                        valid_rotations.append((i, a, b, c))
            
            if not valid_rotations:
                continue
                
            strategies = [
                lambda a, c: (a * c, a, c),
                lambda a, c: (c, a),
                lambda a, c: (a, c),
                lambda a, c: (max(a, c), min(a, c)),
            ]
            
            best_slab_result = None
            best_slab_vol = -1
            
            for strat_func in strategies:
                box_best_rot = {}
                for i, a, b, c in valid_rotations:
                    key = strat_func(a, c)
                    if i not in box_best_rot:
                        box_best_rot[i] = (a, b, c, key)
                    else:
                        ea, eb, ec, ekey = box_best_rot[i]
                        if key < ekey:
                            box_best_rot[i] = (a, b, c, key)
                
                pack_list = [(i, a, b, c) for i, (a, b, c, _) in box_best_rot.items()]
                
                pack_list_s0 = sorted(pack_list, key=lambda x: (x[3], x[1]), reverse=True)
                pack_list_s1 = sorted(pack_list, key=lambda x: (x[1], x[3]), reverse=True)
                
                def run_packing(pack_list, orientation):
                    shelves = []
                    max_pos = 0
                    placed = []
                    
                    for item in pack_list:
                        idx, a, b, c = item
                        if orientation == 0:
                            len_dim = a
                            h_dim = c
                            limit_len = W
                            limit_stack = D
                        else:
                            len_dim = c
                            h_dim = a
                            limit_len = D
                            limit_stack = W
                        
                        best_s = -1
                        best_rem = float('inf')
                        
                        for s_idx, (start, sh, cur) in enumerate(shelves):
                            if h_dim <= sh and cur + len_dim <= limit_len:
                                rem = limit_len - (cur + len_dim)
                                if rem < best_rem:
                                    best_rem = rem
                                    best_s = s_idx
                                    
                        if best_s != -1:
                            start, sh, cur = shelves[best_s]
                            new_cur = cur + len_dim
                            shelves[best_s] = (start, sh, new_cur)
                            
                            if orientation == 0:
                                x, z = cur, start
                            else:
                                x, z = start, cur
                            
                            placed.append((idx, x, z, a, b, c))
                        else:
                            if max_pos + h_dim <= limit_stack:
                                start = max_pos
                                max_pos += h_dim
                                shelves.append((start, h_dim, len_dim))
                                
                                if orientation == 0:
                                    x, z = 0, start
                                else:
                                    x, z = start, 0
                                
                                placed.append((idx, x, z, a, b, c))
                    return placed
                
                res0 = run_packing(pack_list_s0, 0)
                res1 = run_packing(pack_list_s1, 1)
                
                vol0 = sum(a * b * c for _, _, _, a, b, c in res0)
                vol1 = sum(a * b * c for _, _, _, a, b, c in res1)
                
                if vol0 >= vol1:
                    res = res0
                else:
                    res = res1
                
                vol = sum(a * b * c for _, _, _, a, b, c in res)
                
                if vol > best_slab_vol:
                    best_slab_vol = vol
                    best_slab_result = res
            
            if best_slab_result is None:
                continue
            
            for idx, x, z, a, b, c in best_slab_result:
                current_placements[idx] = (0, x, y_start, z, a, b, c)
                placed_boxes.add(idx)
                total_volume += a * b * c
                
        if total_volume > best_volume:
            best_volume = total_volume
            best_placements = [list(p) for p in current_placements]
            
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements
