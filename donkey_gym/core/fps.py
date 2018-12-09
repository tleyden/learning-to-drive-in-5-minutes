# Original author: Tawn Kramer
import time


class FPSTimer(object):
    def __init__(self, verbose=0):
        self.t = time.time()
        self.iter = 0
        self.verbose = verbose

    def reset(self):
        self.t = time.time()
        self.iter = 0

    def on_frame(self):
        self.iter += 1
        if self.iter == 100:
            e = time.time()
            if self.verbose >= 1:
                print('{:.2f} fps'.format(100.0 / (e - self.t)))
            self.t = time.time()
            self.iter = 0
