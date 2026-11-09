import math

def construct():
    def find_solution(k):
        def dfs(def_num, def_den, last_d, rem, path):
            if rem == 0:
                if def_num == 0:
                    return path
                return None
            
            if def_num <= 0:
                return None
            
            # min_d: 1/d <= def => d >= def_den / def_num
            min_d = max(last_d + 1, def_den // def_num)
            if def_den % def_num != 0:
                min_d += 1
            
            # max_d: rem/d >= def => d <= rem * def_den / def_num
            max_d = (rem * def_den) // def_num
            
            if min_d > max_d:
                return None
            
            for d in range(min_d, max_d + 1):
                # New deficit = def - 1/d
                # def_num/def_den - 1/d = (def_num * d - def_den) / (def_den * d)
                new_def_num = def_num * d - def_den
                new_def_den = def_den * d
                
                if new_def_num < 0:
                    continue
                
                if new_def_num == 0:
                    if rem - 1 == 0:
                        return path + [d]
                    else:
                        continue
                
                # Pruning: Max possible sum of remaining rem-1 terms
                # starting from d+1 is approx (rem-1)/(d+1)
                # We need (rem-1)/(d+1) >= new_def
                # (rem-1)/(d+1) >= new_def_num / new_def_den
                # (rem-1) * new_def_den >= new_def_num * (d+1)
                if (rem - 1) * new_def_den < new_def_num * (d + 1):
                    break
                
                res = dfs(new_def_num, new_def_den, d, rem - 1, path + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(1, 1, 4, k, [])

    for k in range(8, 257):
        sol = find_solution(k)
        if sol:
            return sol
    raise RuntimeError("No solution found")
