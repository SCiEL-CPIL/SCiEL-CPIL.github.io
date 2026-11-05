import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 100.0  # 100s margin
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
        
    def get_rotations(w, h, d):
        rots = set()
        for perm in itertools.permutations((w, h, d)):
            rots.add(perm)
        return list(rots)
    
    # Sorting strategies
    def sort_key_max_dim(b):
        return -max(b)
        
    def sort_key_volume(b):
        return -b[0]*b[1]*b[2]
        
    def sort_key_min_dim(b):
        return -min(b)
        
    def sort_key_sum(b):
        return -(b[0]+b[1]+b[2])
        
    def sort_key_max_min_ratio(b):
        # Sort by ratio of max to min?
        m = max(b)
        mi = min(b)
        if mi == 0: return -1
        return -(m/mi)

    strategies = [
        ("max_dim", sort_key_max_dim),
        ("volume", sort_key_volume),
        ("min_dim", sort_key_min_dim),
        ("sum", sort_key_sum),
        ("ratio", sort_key_max_min_ratio),
    ]
    
    best_placements = None
    best_volume = -1
    
    for name, key_func in strategies:
        if time.time() > deadline:
            break
            
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda x: key_func(x[1]))
        
        layers = []
        placements = [None] * n
        
        for idx, (orig_idx, box) in enumerate(indexed_boxes):
            w, h, d = box
            placed = False
            rots = get_rotations(w, h, d)
            
            # Sort rotations by max dimension descending?
            rots.sort(key=lambda r: -max(r))
            
            for rw, rh, rd in rots:
                if placed:
                    break
                    
                for layer in layers:
                    if placed:
                        break
                        
                    # Check if box fits in depth
                    if layer['z'] + rd > D:
                        continue
                        
                    # Try existing shelves
                    for shelf in layer['shelves']:
                        if rh <= shelf['h'] and shelf['x'] + rw <= W:
                            x = shelf['x']
                            y = shelf['y']
                            z = layer['z']
                            placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                            shelf['x'] += rw
                            if rd > layer['depth']:
                                layer['depth'] = rd
                            placed = True
                            break
                            
                    if not placed:
                        if layer['y_max'] + rh <= H:
                            new_shelf = {
                                'y': layer['y_max'],
                                'h': rh,
                                'x': 0
                            }
                            layer['shelves'].append(new_shelf)
                            layer['y_max'] += rh
                            
                            x = 0
                            y = new_shelf['y']
                            z = layer['z']
                            placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                            new_shelf['x'] = rw
                            
                            if rd > layer['depth']:
                                layer['depth'] = rd
                            placed = True
        
                if not placed:
                    if not layers:
                        current_z = 0
                    else:
                        current_z = max(l['z'] + l['depth'] for l in layers)
                    
                    if current_z + rd <= D:
                        new_layer = {
                            'z': current_z,
                            'y_max': 0,
                            'depth': rd,
                            'shelves': []
                        }
                        layers.append(new_layer)
                        
                        new_shelf = {
                            'y': 0,
                            'h': rh,
                            'x': 0
                        }
                        new_layer['shelves'].append(new_shelf)
                        new_layer['y_max'] = rh
                        
                        x = 0
                        y = 0
                        z = current_z
                        placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                        new_shelf['x'] = rw
                        placed = True
        
            if not placed:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        
        if vol > best_volume:
            best_volume = vol
            best_placements = [tuple(p) for p in placements]
            
    if best_placements is None:
        W, H, D = container
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
        
    return best_placements
