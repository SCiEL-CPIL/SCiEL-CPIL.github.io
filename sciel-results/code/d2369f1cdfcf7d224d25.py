    def search(rem, k, path):
        if time.time() > deadline:
            return False
            
        if rem == 0:
            return k == 0
        if k == 0:
            return False
        if k == 1:
            if valid[rem]:
                path.append(rem)
                return True
            return False
        
        # Pick lowest set bit in rem
        lsb = rem & -rem
        v = lsb.bit_length() - 1
        rest_rem = rem ^ lsb
        
        s = rest_rem
        while True:
            p = s | lsb
            if valid[p]:
                path.append(p)
                if search(rem ^ p, k - 1, path):
                    return True
                path.pop()
            if s == 0:
                break
            s = (s - 1) & rest_rem
            
        return False
