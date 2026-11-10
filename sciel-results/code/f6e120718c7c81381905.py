    def obj(x):
        c = x.reshape(N, 2)
        s = sides
        # ...
        lo = c - s / 2.0  # (10,2) - (10,) -> Error? No, numpy broadcasts (10,) to (10,1) if needed? 
                           # Actually, (10,2) - (10,) tries to align the last dimension. 
                           # 2 != 10. So this IS the error source.
