   res = 0
   while a:
       if a & 1: res ^= b
       a >>= 1
       b <<= 1
       if b & 256: b ^= poly
   