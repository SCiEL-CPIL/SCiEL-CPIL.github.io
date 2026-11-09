import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    def search(target, min_d, max_terms, current_terms):
        """Search for representation of target as sum of at most max_terms unit fractions
        with denominators >= min_d. Returns list of denominators or None."""
        if time.time() > deadline:
            return None
        if max_terms == 0:
            return [] if target == 0 else None
        if target == 0:
            return []
        
        # Lower bound on next denominator: d > 1/target, so d >= floor(1/target) + 1
        # But also d >= min_d
        # Upper bound: target <= 1/d + (max_terms-1)/(d+1)
        # Solve for d: target*d*(d+1) <= (d+1) + (max_terms-1)*d
        # target*d^2 + target*d <= d + 1 + (max_terms-1)*d
        # target*d^2 + (target - max_terms)*d - 1 <= 0
        
        t = target
        # Use Fraction for exact arithmetic
        # Find lower bound: d >= max(min_d, floor(1/t) + 1)
        if t <= 0:
            return None
        
        # 1/d < t => d > 1/t => d >= floor(1/t) + 1
        # For Fraction t = p/q, 1/t = q/p, floor(q/p) + 1
        num, den = t.numerator, t.denominator
        if den == 0:
            return None
        lower = max(min_d, den // num + 1)
        
        # Upper bound from: t <= 1/d + (max_terms-1)/(d+1)
        # t*d*(d+1) <= (d+1) + (max_terms-1)*d
        # t*d^2 + t*d <= d + 1 + (max_terms-1)*d
        # t*d^2 + (t - max_terms)*d - 1 <= 0
        # Using quadratic formula on t*d^2 + (t - max_terms)*d - 1 = 0
        # d = [-(t-max_terms) + sqrt((t-max_terms)^2 + 4t)] / (2t)
        # For large t (small target), this is approximately 1/t + something
        
        # Let's just iterate d from lower upward, with a cap
        # The upper bound: from t <= 1/d + (max_terms-1)/(d+1)
        # For d >= lower, check if feasible
        # A simpler upper bound: since each of the (max_terms-1) remaining terms is at most 1/(d+1),
        # we need t - 1/d <= (max_terms-1)/(d+1)
        # Also t - 1/d > 0 => d > 1/t
        
        # Let's try denominators from lower up, but limit the range
        # For efficiency, cap the number of candidates
        max_d_to_try = lower + 10000  # reasonable cap
        if max_d_to_try > 10000000:
            max_d_to_try = 10000000
        
        for d in range(lower, min(max_d_to_try + 1, 10000001)):
            if time.time() > deadline:
                return None
            if d < 5:
                continue
            term = Fraction(1, d)
            if term >= t:
                continue
            remainder = t - term
            if remainder <= 0:
                continue
            # Check if remainder can be represented with (max_terms-1) terms, denominators > d
            # Quick feasibility: remainder <= (max_terms-1) * Fraction(1, d+1)
            if remainder > (max_terms - 1) * Fraction(1, d + 1):
                continue
            # Also remainder must be positive and we need at least 1 term
            if max_terms - 1 == 0:
                if remainder == 0:
                    return [d]
                continue
            
            result = search(remainder, d + 1, max_terms - 1, current_terms + [d])
            if result is not None:
                return [d] + result
        
        return None
    
    # Try increasing number of terms
    # Known: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100 + 1/101 + 1/102 + 1/103 + 1/104 + 1/105 + 1/106 + 1/107 + 1/108 + 1/109 + 1/110 + 1/111 + 1/112 + 1/113 + 1/114 + 1/115 + 1/116 + 1/117 + 1/118 + 1/119 + 1/120 + 1/121 + 1/122 + 1/123 + 1/124 + 1/125 + 1/126 + 1/127 + 1/128 + 1/129 + 1/130 + 1/131 + 1/132 + 1/133 + 1/134 + 1/135 + 1/136 + 1/137 + 1/138 + 1/139 + 1/140 + 1/141 + 1/142 + 1/143 + 1/144 + 1/145 + 1/146 + 1/147 + 1/148 + 1/149 + 1/150 + 1/151 + 1/152 + 1/153 + 1/154 + 1/155 + 1/156 + 1/157 + 1/158 + 1/159 + 1/160 + 1/161 + 1/162 + 1/163 + 1/164 + 1/165 + 1/166 + 1/167 + 1/168 + 1/169 + 1/170 + 1/171 + 1/172 + 1/173 + 1/174 + 1/175 + 1/176 + 1/177 + 1/178 + 1/179 + 1/180 + 1/181 + 1/182 + 1/183 + 1/184 + 1/185 + 1/186 + 1/187 + 1/188 + 1/189 + 1/190 + 1/191 + 1/192 + 1/193 + 1/194 + 1/195 + 1/196 + 1/197 + 1/198 + 1/199 + 1/200 + 1/201 + 1/202 + 1/203 + 1/204 + 1/205 + 1/206 + 1/207 + 1/208 + 1/209 + 1/210 + 1/211 + 1/212 + 1/213 + 1/214 + 1/215 + 1/216 + 1/217 + 1/218 + 1/219 + 1/220 + 1/221 + 1/222 + 1/223 + 1/224 + 1/225 + 1/226 + 1/227 + 1/228 + 1/229 + 230*0 + ... 
    # This is getting ridiculous. Let me just search.
    
    # Start with small term counts
    best_result = None
    for num_terms in range(1, 50):  # try up to 50 terms
        if time.time() > deadline:
            break
        result = search(Fraction(1, 1), 5, num_terms, [])
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback: use a known representation
        # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100 + 1/101 + 1/102 + 1/103 + 1/104 + 1/105 + 1/106 + 1/107 + 1/108 + 1/109 + 1/110 + 1/111 + 1/112 + 1/113 + 1/114 + 1/115 + 1/116 + 1/117 + 1/118 + 1/119 + 1/120 + 1/121 + 1/122 + 1/123 + 1/124 + 1/125 + 1/126 + 1/127 + 1/128 + 1/129 + 1/130 + 1/131 + 1/132 + 1/133 + 1/134 + 1/135 + 1/136 + 1/137 + 1/138 + 1/139 + 1/140 + 1/141 + 1/142 + 1/143 + 1/144 + 1/145 + 1/146 + 1/147 + 1/148 + 1/149 + 1/150 + 1/151 + 1/152 + 1/153 + 1/154 + 1/155 + 1/156 + 1/157 + 1/158 + 1/159 + 1/160 + 1/161 + 1/162 + 1/163 + 1/164 + 1/165 + 1/166 + 1/167 + 1/168 + 1/169 + 1/170 + 1/171 + 1/172 + 1/173 + 1/174 + 1/175 + 1/176 + 1/177 + 1/178 + 1/179 + 1/180 + 1/181 + 1/182 + 1/183 + 1/184 + 1/185 + 1/186 + 1/187 + 1/188 + 1/189 + 1/190 + 1/191 + 1/192 + 1/193 + 1/194 + 1/195 + 1/196 + 1/197 + 1/198 + 1/199 + 1/200 + 1/201 + 1/202 + 1/203 + 1/204 + 1/205 + 1/206 + 1/207 + 1/208 + 1/209 + 1/210 + 1/211 + 1/212 + 1/213 + 1/214 + 1/215 + 1/216 + 1/217 + 1/218 + 1/219 + 1/220 + 1/221 + 1/222 + 1/223 + 1/224 + 1/225 + 1/226 + 1/227 + 1/228 + 1/229 + 1/230 + 1/231 + 1/232 + 1/233 + 1/234 + 1/235 + 1/236 + 1/237 + 1/238 + 1/239 + 1/240 + 1/241 + 1/242 + 1/243 + 1/244 + 1/245 + 1/246 + 1/247 + 1/248 + 1/249 + 1/250 + 1/251 + 1/252 + 1/253 + 1/254 + 1/255 + 1/256
        # Let me just use a simpler fallback
        best_result = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]
        # Verify and adjust if needed
        s = sum(Fraction(1, d) for d in best_result)
        if s != 1:
            # Just return a valid one we can compute
            # Use the original approach as fallback
            target, reached = 2520, {0: []}
            for d in range(5, target + 1):
                if target % d:
                    continue
                w = target // d
                for total, values in list(reached.items()):
                    if total + w <= target and total + w not in reached:
                        reached[total + w] = values + [d]
                if target in reached:
                    best_result = reached[target]
                    break
    
    return best_result
