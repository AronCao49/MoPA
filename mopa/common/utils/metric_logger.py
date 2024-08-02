# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
from __future__ import division
from collections import defaultdict
from collections import deque

import numpy as np
import torch

from typing import Dict, List
import openpyxl
from mopa.data.utils.evaluate import Evaluator


class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    default_fmt = '{avg:.4f} ({global_avg:.4f})'
    default_summary_fmt = '{global_avg:.4f}'

    def __init__(self, window_size=20, fmt=None, summary_fmt=None):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0
        self.fmt = fmt or self.default_fmt
        self.summary_fmt = summary_fmt or self.default_summary_fmt

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    @property
    def avg(self):
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def global_avg(self):
        return self.sum / self.count if self.count != 0 else float('nan')

    def reset(self):
        self.values.clear()
        self.counts.clear()
        self.sum = 0.0
        self.count = 0

    def __str__(self):
        return self.fmt.format(avg=self.avg, global_avg=self.global_avg)

    @property
    def summary_str(self):
        return self.summary_fmt.format(global_avg=self.global_avg)


class MetricLogger(object):
    """Metric logger.
    All the meters should implement following methods:
        __str__, summary_str, reset
    """

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                count = v.numel()
                value = v.item() if count == 1 else v.sum().item()
            elif isinstance(v, np.ndarray):
                count = v.size
                value = v.item() if count == 1 else v.sum().item()
            else:
                assert isinstance(v, (float, int))
                value = v
                count = 1
            self.meters[k].update(value, count)
    
    # Add remove func to get rid of NaN results        
    def remove(self, name):
        if name in self.meters.keys():
            self.meters.pop(name)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def add_meters(self, meters):
        if not isinstance(meters, (list, tuple)):
            meters = [meters]
        for meter in meters:
            self.add_meter(meter.name, meter)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return getattr(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append('{}: {}'.format(name, str(meter)))
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append('{}: {}'.format(name, meter.summary_str))
        return self.delimiter.join(metric_str)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()


def iou_to_excel(evaluator_dict: Dict[str, Evaluator], excel_pth: str, eval_keys: List[str]):
    """Function to automatically save class-wise IoU to excel
    Args:
        evaluator_dict (Dict): dictionary of evaluators (e.g., 3D, 2D, xM)
        excel_pth (str): path to save excel file of performance
        eval_keys (str): keys of evaluators that want to save
    """
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    class_names = [name for _, name in enumerate(evaluator_dict['2D'].class_names)]
    table_head = ["Modal"] + class_names + ["avg"]
    sheet.append(table_head)
    for key in eval_keys:
        key_row = [key] + [iou * 100 for iou in evaluator_dict[key].class_iou] \
            + [evaluator_dict[key].overall_iou * 100]
        sheet.append(key_row)
    
    workbook.save(excel_pth)