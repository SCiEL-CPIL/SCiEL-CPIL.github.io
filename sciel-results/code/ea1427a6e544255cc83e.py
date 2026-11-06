    class SolverOpt:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.missing_list = list(range(num_triples))
            self.missing_set = set(range(num_triples))
            self.idx_in_missing = {t: i for i, t in enumerate(self.missing_list)}
            
            self.present_list = []
            self.idx_in_present = {}

        def can_add(self, t_id):
            if t_id in self.present:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if t_id in self.present:
                return True
            if not self.can_add(t_id):
                return False
            
            self.present.add(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            
            # Update present_list
            self.present_list.append(t_id)
            self.idx_in_present[t_id] = len(self.present_list) - 1
            
            # Update missing_list
            if t_id in self.idx_in_missing:
                idx = self.idx_in_missing[t_id]
                last_idx = len(self.missing_list) - 1
                last_item = self.missing_list[last_idx]
                
                self.missing_list[idx] = last_item
                self.idx_in_missing[last_item] = idx
                self.missing_list.pop()
                del self.idx_in_missing[t_id]
                self.missing_set.remove(t_id)
            
            return True

        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
            
            # Update present_list
            if t_id in self.idx_in_present:
                idx = self.idx_in_present[t_id]
                last_idx = len(self.present_list) - 1
                last_item = self.present_list[last_idx]
                
                self.present_list[idx] = last_item
                self.idx_in_present[last_item] = idx
                self.present_list.pop()
                del self.idx_in_present[t_id]
                
            # Update missing_list
            self.missing_list.append(t_id)
            self.idx_in_missing[t_id] = len(self.missing_list) - 1
            self.missing_set.add(t_id)

        def size(self):
            return len(self.present)
