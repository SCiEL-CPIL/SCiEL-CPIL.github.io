<<<<<<< SEARCH
    if best_result is None:
        return [(-1, 0, 0, 0, int(w), int(h), int(d)) for w, h, d in boxes]
    
    final_placements = []
    for i in range(n):
        if best_result[i] is not None:
            x, y, z, w, h, d = best_result[i]
            final_placements.append((0, int(x), int(y), int(z), int(w), int(h), int(d)))
        else:
            w, h, d = boxes[i]
            final_placements.append((-1, 0, 0, 0, int(w), int(h), int(d)))
            
    return final_placements
=======
    if best_result is None:
        return [(-1, 0, 0, 0, int(w), int(h), int(d)) for w, h, d in boxes]
    
    # Phase 3: Polish pass - try to re-place unplaced boxes in remaining space
    if time.time() < deadline - 5:
        # Identify unplaced boxes
        unplaced = [i for i in range(n) if best_result[i] is None]
        
        if unplaced:
            # Try to place each unplaced box at any valid position
            for idx in unplaced:
                if time.time() > deadline - 3:
                    break
                
                box = boxes[idx]
                orients = orient_cache[idx]
                
                # Scan all possible positions (coarse to fine)
                placed = False
                for ow, oh, od in orients:
                    if placed:
                        break
                    for z in range(D - od + 1):
                        if placed:
                            break
                        for y in range(H - oh + 1):
                            if placed:
                                break
                            for x in range(W - ow + 1):
                                if is_valid_grid(x, y, z, ow, oh, od):
                                    best_result[idx] = (x, y, z, ow, oh, od)
                                    place_in_grid(x, y, z, ow, oh, od)
                                    placed = True
                                    break
    
    final_placements = []
    for i in range(n):
        if best_result[i] is not None:
            x, y, z, w, h, d = best_result[i]
            final_placements.append((0, int(x), int(y), int(z), int(w), int(h), int(d)))
        else:
            w, h, d = boxes[i]
            final_placements.append((-1, 0, 0, 0, int(w), int(h), int(d)))
            
    return final_placements
>>>>>>> REPLACE
