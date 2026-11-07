    memo = {}
    choice = {}

    def can_partition_store(mask, k):
        if (mask, k) in memo:
            return memo[(mask, k)]
        
        if k == 0:
            res = (mask == 0)
        elif k == 1:
            res = is_valid[mask]
        elif mask == 0:
            res = (k == 0)
        elif k > bin(mask).count('1'):
            res = False
        else:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            rest_mask = mask ^ lsb
            sub_rest = rest_mask
            res = False
            while True:
                S = sub_rest | lsb
                if is_valid[S]:
                    if can_partition_store(mask ^ S, k - 1):
                        choice[(mask, k)] = S
                        res = True
                        break
                if sub_rest == 0:
                    break
                sub_rest = (sub_rest - 1) & rest_mask
        
        memo[(mask, k)] = res
        return res
