# Simply model to both log through training and save outputs for later
# Values are saved as: seconds_since_start, x_value, y_value and name is x_value__y_value

from logging import Logger
from typing import Dict
import datetime
import time
import os

class DetrLogger():
    def __init__(self, name: str = "Detr logger", level: int | str = 0) -> None:
        self.memory: Dict[str, list] = {}
        self.logger = Logger(name, level)
        self.start_time = time.time()

    def info(self, key, x_value, y_value):
        if key not in self.memory: self.memory[key] = []
        seconds_since_start = str(datetime.timedelta(seconds=int(time.time() - self.start_time)))
        self.memory[key].append((seconds_since_start, str(x_value), str(y_value)))

    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)

    def save(self):
        for key, values in self.memory.items():
            with open(os.path.join("outputs", key + ".csv"), "w") as file:
                print(values)
                file.writelines([",".join([time, x_value, y_value]) + "\n" for (time, x_value, y_value) in values])

if __name__ == "__main__":
    logger = DetrLogger()
    logger.info("Epoch__TrainLoss", 0, 141.92)
    logger.info("Epoch__TrainLoss", 1, 131.12)
    logger.info("Epoch__TrainLoss", 2, 111.12)
    logger.info("Epoch__TrainLoss", 3, 111.22)
    logger.info("Epoch__TrainLoss", 4, 101.32)
    logger.info("Epoch__EvalAcc", 0, 49.91)
    logger.info("Epoch__EvalAcc", 1, 59.91)
    logger.info("Epoch__EvalAcc", 2, 69.91)
    logger.info("Epoch__EvalAcc", 3, 79.91)
    logger.info("Epoch__EvalAcc", 4, 89.91)
    logger.save()
