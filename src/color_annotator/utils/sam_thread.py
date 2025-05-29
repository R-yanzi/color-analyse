# utils/sam_thread.py
from PyQt5.QtCore import QThread, pyqtSignal

class SAMWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, sam_predictor, points=None, labels=None, single_point=None, multimask_output=False):
        super().__init__()
        self.sam = sam_predictor
        self.points = points
        self.labels = labels
        self.single_point = single_point  # 保留单点模式用
        self.multimask_output = multimask_output

    def run(self):
        try:
            if self.single_point is not None:
                mask = self.sam.predict_from_point(self.single_point)
            else:
                # 走 wrapper，内部包含坐标缩放逻辑
                mask = self.sam.predict_from_points(self.points, self.labels, self.multimask_output)
            self.finished.emit(mask)
        except Exception as e:
            print(f"[SAMWorker] 分割失败：{e}")
            self.finished.emit(None)

