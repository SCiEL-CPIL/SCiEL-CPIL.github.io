import time
import random

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety
    
    N = 512
    K = 32
    
    def score_bits(bits):
        count = 0
        # Local variable for speed
        b = bits
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))) != 0:
                count += 1
        return count

    def bits_to_list(bits):
        return [i + 1 for i in range(N) if (bits >> i) & 1]

    # 1. Greedy Initialization with Randomization for Diversity
    # We run greedy a few times with different candidate orderings to find a good start
    best_initial_bits = 0
    best_initial_score = -1
    
    # Run 3 different greedy variants
    for variant in range(3):
        S = set()
        current_bits = 0
        # Create a shuffled list of candidates
        candidates = list(range(1, N + 1))
        if variant > 0:
            random.shuffle(candidates)
            
        for _ in range(K):
            best_x = -1
            best_score = -1
            for x in candidates:
                if x in S:
                    continue
                new_bits = current_bits | (1 << (x - 1))
                sc = score_bits(new_bits)
                if sc > best_score:
                    best_score = sc
                    best_x = x
            
            if best_x != -1:
                S.add(best_x)
                current_bits |= (1 << (best_x - 1))
                
        if best_score > best_initial_score:
            best_initial_score = best_score
            best_initial_bits = current_bits

    # 2. Simulated Annealing
    current_bits = best_initial_bits
    current_score = best_initial_score
    best_bits = current_bits
    best_score = current_score
    
    # Maintain a set and list for fast random selection
    current_set = set(bits_to_list(current_bits))
    current_list = list(current_set)
    
    T_start = 5.0
    T_end = 0.01
    sa_start_time = time.time()
    sa_duration = time_limit - (sa_start_time - start_time)
    
    no_improvement_count = 0
    kick_threshold = 5000  # If no improvement for 5000 steps, do a kick
    
    while time.time() - start_time < time_limit:
        # Pick random x in S, y not in S
        x = random.choice(current_list)
        y = random.randint(1, N)
        
        if y in current_set:
            continue
            
        # Compute new bits
        new_bits = current_bits
        new_bits &= ~(1 << (x - 1))
        new_bits |= (1 << (y - 1))
        
        new_score = score_bits(new_bits)
        delta = new_score - current_score
        
        # Calculate T based on time elapsed in SA phase
        elapsed_sa = time.time() - sa_start_time
        if sa_duration > 0:
            progress = elapsed_sa / sa_duration
            if progress >= 1.0:
                break
            T = T_end + (T_start - T_end) * (1.0 - progress)
        else:
            T = T_end
            
        # Acceptance Criterion
        if delta > 0 or (delta < 0 and random.random() < pow(2.718281828, delta / T)):
            # Accept move
            current_bits = new_bits
            current_score = new_score
            
            # Update set and list
            current_set.remove(x)
            current_set.add(y)
            current_list.remove(x)
            current_list.append(y)
            
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
            
        # Random Kick if stuck
        if no_improvement_count > kick_threshold:
            # Replace 2 random elements with new random ones
            for _ in range(2):
                x_kick = random.choice(current_list)
                y_kick = random.randint(1, N)
                while y_kick in current_set:
                    y_kick = random.randint(1, N)
                
                current_bits &= ~(1 << (x_kick - 1))
                current_bits |= (1 << (y_kick - 1))
                
                current_set.remove(x_kick)
                current_set.add(y_kick)
                current_list.remove(x_kick)
                current_list.append(y_kick)
                
            current_score = score_bits(current_bits)
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
            no_improvement_count = 0
            kick_threshold = int(kick_threshold * 1.5) # Increase threshold

    return bits_to_list(best_bits)
