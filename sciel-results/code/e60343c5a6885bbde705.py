import struct

# State Layout:
# 0..511: 128 pairs of (uint16 item, uint16 count)
# 512..1023: 8 rows * 32 cols * uint16 CMS counters

SPARSE_SIZE = 128
CMS_ROWS = 8
CMS_COLS = 32

# Hash multipliers for CMS
# Using large primes to spread items
P = [1035493289, 775374203, 1495862053, 2015145641, 
     1125799837, 1583407777, 1727880107, 1917545813]

def _hash(item, row):
    # Simple hash: (item * P[row]) % 2147483647 % CMS_COLS
    # 2147483647 is a Mersenne prime
    h = (item * P[row]) % 2147483647
    return h % CMS_COLS

def update(state, item):
    # Unpack sparse table
    sparse_data = struct.unpack('<256H', state[:512])
    sparse_data = list(sparse_data)
    
    # Unpack CMS
    cms_data = struct.unpack('<256H', state[512:])
    cms_data = list(cms_data)
    
    # 1. Update CMS
    for r in range(CMS_ROWS):
        idx = _hash(item, r)
        cms_idx = r * CMS_COLS + idx
        cms_data[cms_idx] += 1
        
    # 2. Update Sparse Table
    found = False
    min_idx = 0
    min_val = sparse_data[1] # count of first pair
    
    for i in range(SPARSE_SIZE):
        item_idx = i * 2
        count_idx = i * 2 + 1
        
        if sparse_data[item_idx] == item:
            sparse_data[count_idx] += 1
            found = True
            break
            
        if sparse_data[count_idx] < min_val:
            min_val = sparse_data[count_idx]
            min_idx = i * 2 # item index
            
    if not found:
        # Get CMS estimate for the new item
        # Note: CMS was just updated, so it includes this item's count
        estimates = []
        for r in range(CMS_ROWS):
            idx = _hash(item, r)
            estimates.append(cms_data[r * CMS_COLS + idx])
        cms_est = min(estimates)
        
        # Decide if we should evict
        # If the slot is empty (min_val == 0) or new item is "heavier" than min
        if min_val == 0 or cms_est > min_val:
            sparse_data[min_idx] = item
            # Set count to cms_est? Or min_val + 1?
            # If we set to cms_est, it's an overestimate potentially.
            # If we set to min_val + 1, it's consistent with Space-Saving.
            # However, if cms_est is much larger, we lose that info.
            # Let's use cms_est to capture heavy hitters better.
            sparse_data[min_idx + 1] = cms_est
            # But wait, if cms_est is 1 and min_val is 1, we replace.
            # If cms_est is 5 and min_val is 2, we replace.
            # This seems good.
            
    # Pack and return
    new_sparse = struct.pack('<256H', *sparse_data)
    new_cms = struct.pack('<256H', *cms_data)
    return new_sparse + new_cms

def query(state, item):
    sparse_data = struct.unpack('<256H', state[:512])
    
    for i in range(SPARSE_SIZE):
        if sparse_data[i * 2] == item:
            return float(sparse_data[i * 2 + 1])
            
    # Not in sparse table, use CMS
    cms_data = struct.unpack('<256H', state[512:])
    estimates = []
    for r in range(CMS_ROWS):
        idx = _hash(item, r)
        estimates.append(cms_data[r * CMS_COLS + idx])
    return float(min(estimates))
