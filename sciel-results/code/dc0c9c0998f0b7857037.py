    for base in candidates:
        base_mask = 0
        for i in base:
            base_mask |= (1 << i)
            
        orbit_masks = set()
        for p in perms:
            mask = 0
            for i in range(6):
                mask |= (1 << p[base[i]])
            orbit_masks.add(mask)
            
        if len(orbit_masks) == 132:
            result = []
            for m in orbit_masks:
                block = []
                for i in range(12):
                    if m & (1 << i):
                        block.append(i)
                result.append(block)
            return result
