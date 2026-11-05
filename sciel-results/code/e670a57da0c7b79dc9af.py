# EVOLVE-BLOCK-START
import numpy as np

def load(boxes, container):
    W, H, D = container
    
    # Grid resolution
    target_dim = 50
    step_x = max(1, int(np.ceil(W / target_dim)))
    step_y = max(1, int(np.ceil(H / target_dim)))
    step_z = max(1, int(np.ceil(D / target_dim)))
    
    nx = int(np.ceil(W / step_x))
    ny = int(np.ceil(H / step_y))
    nz = int(np.ceil(D / step_z))
    
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    max_x = 0
    max_y = 0
    max_z = 0
    
    box_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    results = [None] * len(boxes)
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        
        best_pos = None
        best_orient = None
        
        orientations = [
            (w, h, d), (w, d, h), (h, w, d),
            (h, d, w), (d, w, h), (d, h, w)
        ]
        
        unique_orients = []
        seen = set()
        for o in orientations:
            if o not in seen:
                unique_orients.append(o)
                seen.add(o)
        
        for ow, oh, od in unique_orients:
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            if gw > nx or gh > ny or gd > nz:
                continue
            
            # 1. Try Compact Search
            search_max_x = min(nx, max_x + gw)
            search_max_y = min(ny, max_y + gh)
            search_max_z = min(nz, max_z + gd)
            
            found = False
            for x in range(search_max_x - gw + 1):
                if found: break
                for y in range(search_max_y - gh + 1):
                    if found: break
                    for z in range(search_max_z - gd + 1):
                        if np.all(occ[x:x+gw, y:y+gh, z:z+gd] == 0):
                            best_pos = (x, y, z)
                            best_orient = (ow, oh, od)
                            found = True
                            break
            
            if found:
                break
            
            # 2. If not found, try Full Search
            # Only do full search if the box is small enough or if we have time?
            # For simplicity, always do full search if compact fails.
            # But full search is slow.
            # We can limit full search to a subset?
            # Let's just do full search.
            
            for x in range(nx - gw + 1):
                if found: break
                for y in range(ny - gh + 1):
                    if found: break
                    for z in range(nz - gd + 1):
                        if np.all(occ[x:x+gw, y:y+gh, z:z+gd] == 0):
                            best_pos = (x, y, z)
                            best_orient = (ow, oh, od)
                            found = True
                            break
            
            if found:
                break
        
        if best_pos is not None:
            bx, by, bz = best_pos
            ow, oh, od = best_orient
            
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            occ[bx:bx+gw, by:by+gh, bz:bz+gd] = 1
            
            max_x = max(max_x, bx + gw)
            max_y = max(max_y, by + gh)
            max_z = max(max_z, bz + gd)
            
            real_x = bx * step_x
            real_y = by * step_y
            real_z = bz * step_z
            
            results[idx] = (0, real_x, real_y, real_z, ow, oh, od)
        else:
            results[idx] = (-1, 0, 0, 0, w, h, d)
            
    return results
# EVOLVE-BLOCK-END
