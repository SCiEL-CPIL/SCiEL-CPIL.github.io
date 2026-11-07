   import struct
   
   def update(state, item):
       b = bytearray(state)
       offset = item * 2
       val = struct.unpack_from('<H', b, offset)[0]
       val += 1
       # Max count is 1024, so uint16 (65535) is safe.
       struct.pack_into('<H', b, offset, val)
       return bytes(b)
   
   def query(state, item):
       offset = item * 2
       val = struct.unpack_from('<H', state, offset)[0]
       return val
   