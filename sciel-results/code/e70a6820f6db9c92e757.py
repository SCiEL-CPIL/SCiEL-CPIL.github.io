import struct
import array

def update(state, item):
    # Layout: 50 bytes bitset + 974 bytes (243 uint16 counters)
    # Bitset: 400 bits = 50 bytes
    # Counters: 243 uint16 = 486 bytes, stored at offset 50
    # We use a hash to map items to 243 buckets
    # If item is new (bitset bit not set), we allocate a bucket for it
    # We use an open-addressing-like scheme: hash item to bucket, if occupied by another item, use next
    
    # Actually, let's use a simpler approach:
    # 50 bytes bitset
    # 974 bytes: use 487 uint16 counters
    # Each item gets a fixed slot based on hash
    # But we need to handle collisions
    
    # Better: use the bitset to track which items are seen
    # For the counters, use a hash table with 487 slots
    # Store: for each slot, either empty (0) or (item_id, count)
    # But item_id needs 9 bits, count needs 7 bits for up to 127
    # That's 16 bits per slot, so 487 slots * 2 bytes = 974 bytes
    
    # Actually, let's do:
    # 50 bytes bitset (400 bits)
    # 974 bytes for counters
    # Use 487 uint16 slots
    # Each slot stores: if item is in this slot, store (item & 0x1FF) << 7 | min(count, 127)
    # But item & 0x1FF is 9 bits, count is 7 bits = 16 bits total
    
    # Wait, items are 0..399, so 9 bits needed for item id
    # Count can be up to 1024, but we cap at 127 for 7 bits
    # Actually, let's use a different scheme:
    # 487 slots, each 2 bytes
    # Slot value = 0 means empty
    # Otherwise, slot_value = (item_id + 1) * 1025 + min(count, 1024)
    # But (399+1)*1025 + 1024 = 400*1025+1024 = 411024+1024 = 412048, which fits in uint16? No, 65535 max.
    
    # Let me reconsider. Use separate arrays.
    # 50 bytes bitset
    # Then use the remaining 974 bytes as:
    # - A hash table with 487 entries
    # - Each entry: 1 byte for item_id (but we need 9 bits...) 
    # 
    # Alternative: since items are 0..399, use 2 bytes per entry:
    # Entry 0: empty
    # Entry 1..400: maps to item (entry-1)
    # But we need to store counts too.
    
    # Let's use: 50 bytes bitset + 400 bytes for counts (1 byte each, max 255)
    # That's 450 bytes. Remaining 574 bytes unused.
    # But counts can exceed 255 for Zipf.
    
    # Better: 50 bytes bitset + 400 * 2 bytes = 850 bytes for uint16 counts
    # Total: 900 bytes. 124 bytes remaining.
    # This gives exact counts for all seen items!
    
    # Actually wait - if we store exact counts for all 400 items, that's perfect.
    # 50 + 800 = 850 bytes. We have 1024 - 850 = 174 bytes spare.
    
    # Let's do exactly that:
    # Bytes 0-49: bitset (400 bits)
    # Bytes 50-849: 400 uint16 counters (little-endian)
    
    bs = bytearray(state)
    
    # Check if item already seen
    byte_idx = item // 8
    bit_idx = item % 8
    bit = 1 << bit_idx
    
    if not (bs[byte_idx] & bit):
        # New item, set bit
        bs[byte_idx] |= bit
        # Count starts at 1 (we're about to increment)
    
    # Increment count for this item
    count_offset = 50 + item * 2
    current = bs[count_offset] | (bs[count_offset + 1] << 8)
    new_count = current + 1
    bs[count_offset] = new_count & 0xFF
    bs[count_offset + 1] = (new_count >> 8) & 0xFF
    
    return bytes(bs)

def query(state, item):
    bs = bytearray(state)
    
    # Check if item was seen
    byte_idx = item // 8
    bit_idx = item % 8
    bit = 1 << bit_idx
    
    if not (bs[byte_idx] & bit):
        return 0
    
    # Get exact count
    count_offset = 50 + item * 2
    count = bs[count_offset] | (bs[count_offset + 1] << 8)
    return count
