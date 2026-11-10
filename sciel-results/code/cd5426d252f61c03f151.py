from array import array

def gen_filtered(vals, limit):
    sums = array('q', [0])
    masks = array('q', [0])
    for i, v in enumerate(vals):
        size = len(sums)
        # We need to append to the same array
        # array.append is slow if done one by one?
        # Better to extend with a list or array.
        
        # But we need to filter based on sums[j] + v < limit
        # So we can't just extend with all sums + v.
        
        new_sums_list = []
        new_masks_list = []
        for j in range(size):
            s = sums[j]
            ns = s + v
            if ns < limit:
                new_sums_list.append(ns)
                new_masks_list.append(masks[j] | (1 << i))
        
        if new_sums_list:
            sums.extend(new_sums_list)
            masks.extend(new_masks_list)
            
    return sums, masks
