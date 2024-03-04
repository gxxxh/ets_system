import logging
import os
from enum import Enum
from pathlib import Path

import pandas as pd

from ets_system import util

BASE_DIR = Path(__file__).resolve().parent.parent
logs_dir = os.path.join(BASE_DIR, 'data', 'predictor_logs')
script_path = os.path.join(BASE_DIR, 'benchmark', 'run.py')

nsys_path = '/usr/local/cuda/bin/nsys'
python_path = '/root/miniconda3/envs/gh-torch/bin/python'
predictor_path = os.path.join(BASE_DIR, 'data/predictors')

gpu = os.environ.get('GPU_NAME', 'T4CPUALL')


class LogStatus(Enum):
    Pending = 1
    Perfed = 2
    Predicted = 3
    Error = 4


def list_logs():
    dirs = os.listdir(logs_dir)
    res = []
    for _, log_id in enumerate(dirs):
        log_dir = os.path.join(logs_dir, log_id)
        status = LogStatus.Pending
        if util.find_file_with_suffix(log_dir, '.measure.csv'):
            status = LogStatus.Perfed
        if util.find_file_with_suffix(log_dir, '.predict.csv'):
            status = LogStatus.Predicted
        meta_file = os.path.join(log_dir, 'meta_info.txt')
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta_info = eval(f.read())
            meta_info['status'] = status.name
            res.append(meta_info)
        else:
            continue
    return res


def list_detail(log_id: str):
    log_path = os.path.join(logs_dir, log_id)
    if not os.path.exists(log_path):
        return 'no such uuid'
    predict_file = os.path.join(log_path, 'predict.csv')
    print(predict_file)
    detail = dict()
    if os.path.exists(predict_file):
        predict_df = pd.read_csv(os.path.join(log_path, predict_file))
        predict_df = predict_df[0:30]
        detail['measured'] = predict_df['total'].sum()
        detail['predict'] = predict_df['total_pred'].sum()
        detail['details'] = predict_df.to_dict(orient='records')
        return detail
    else:
        return 'no predict file, please do predict first'




if __name__ == '__main__':
    # densenet报错
    print("1")
