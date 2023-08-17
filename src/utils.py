import os
import sys

import pandas as pd
from dataclasses import dataclass
import numpy as np
import dill

from src.logger import logging
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        logging.error("Error while saving object")
        raise CustomException(e,sys)