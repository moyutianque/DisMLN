from unittest import result
import torch
from torchmetrics.metric import Metric, _LIGHTNING_AVAILABLE, Tensor

from collections import defaultdict
from utils.time_tools import execution_time
import copy

from heapq import heapify, heappush, heappushpop, nlargest

class MaxHeap(object):
    def __init__(self, top_n):
        self.h = []
        self.length = top_n
        heapify( self.h)
        
    def add(self, element):
        if len(self.h) < self.length:
            heappush(self.h, element)
        else:
            heappushpop(self.h, element)
            
    def getTop(self, n):
        if n< self.length:
            return nlargest(n, self.h)
        return nlargest(self.length, self.h)

class MLN_SuccessRate(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("success_rate", default=torch.tensor(0), dist_reduce_fx="mean")
        
        self.history = defaultdict(lambda: MaxHeap(top_n=10)) # has not consider sync of history dict
        # TODO: sync of history accumulator
    
    def update(self, results):
        # update metric states
        for res in results:
            res_local = copy.deepcopy(res)
            ep = res_local['ep_id'].split('-')[0]
            score = res_local['pred']
            self.history[ep].add((score, (res_local['dis_score'], res_local['ndtw'], res_local["ep_id"])))

    def compute(self):
        # compute final result
        success_rate, pred_results = self.__get_success_rate()
        self.success_rate = torch.tensor(success_rate)
        return self.success_rate, pred_results

    def reset(self) -> None:
        """ modified reset """
        self._update_called = False
        self._forward_cache = None
        # lower lightning versions requires this implicitly to log metric objects correctly in self.log
        if not _LIGHTNING_AVAILABLE or self._LIGHTNING_GREATER_EQUAL_1_3:
            self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                setattr(self, attr, [])

        setattr(self, "history", defaultdict(lambda: MaxHeap(top_n=10)))

        # reset internal states
        self._cache = None
        self._is_synced = False

    def __get_success_rate(self):
        correct = 0
        tot = 0
        pred_results = []
        assert len(self.history) > 10, f"History length {len(self.history)} is impossible for normal validation"
        for k,v in self.history.items():
            tot += 1
            value = v.getTop(1)[0]
            if value[1][0] > 4: # in 3 meter to goal point
                correct += 1
                pred_results.append({"episode_id": k, "ep_id": value[1][2], "pred": value[0], "is_correct": True, "ndtw": value[1][1], "dist_score": value[1][0]})
            else:
                pred_results.append({"episode_id": k, "ep_id": value[1][2], "pred": value[0], "is_correct": False, "ndtw": value[1][1], "dist_score": value[1][0]})

        if tot == 0:
            return 0, {}
        return correct / tot, pred_results

