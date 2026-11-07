import struct

def update(state, item):
    # 2 rows of 128 uint32s
    # Row 0 offset 0, Row 1 offset 512
    
    # Hash 0
    idx0 = ((item * 1035493289 + 31) % 2147483647) % 128
    # Hash 1
    idx1 = ((item * 775374203 + 61) % 2147483647) % 128
    
    # Convert state to bytearray for mutable access
    # Note: state is bytes, we need to return bytes
    # It's faster to work with a bytearray copy? 
    # Or use struct on the bytes directly? struct.pack_into requires a mutable buffer.
    # So we must convert to bytearray, modify, then return bytes.
    
    buf = bytearray(state)
    
    # Update Row 0
    offset0 = idx0 * 4
    val0 = struct.unpack_from('<I', buf, offset0)[0]
    struct.pack_into('<I', buf, offset0, val0 + 1)
    
    # Update Row 1
    offset1 = 512 + idx1 * 4
    val1 = struct.unpack_from('<I', buf, offset1)[0]
    struct.pack_into('<I', buf, offset1, val1 + 1)
    
    return bytes(buf)

def query(state, item):
    # Hash 0
    idx0 = ((item * 1035493289 + 31) % 2147483647) % 128
    # Hash 1
    idx1 = ((item * 775374203 + 61) % 2147483647) % 128
    
    # state is bytes, we can use struct.unpack_from directly on bytes?
    # Yes, struct.unpack_from works on bytes-like objects.
    
    offset0 = idx0 * 4
    val0 = struct.unpack_from('<I', state, offset0)[0]
    
    offset1 = 512 + idx1 * 4
    val1 = struct.unpack_from('<I', state, offset1)[0]
    
    return min(val0, val1)
