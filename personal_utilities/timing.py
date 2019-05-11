import time

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
    def reset(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.markings  = []
        self.splits    = []
    def mark(self, marker):
        if self.verbose:
            print('   ...marking', marker)
        self.markings.append(marker)
        timing = time.time()
        self.splits.append(timing-self.last_time)
        self.last_time = timing
    def __call__(self):
        return time.time() - self.start_time
    def print(self):
        l = 0
        s = 0
        for mark in self.markings:
            l = max(l, len(mark))
        for split in self.splits:
            s = max(s, len('{:0.1f}'.format(split*1000)))
        l += 5
        s += 2
        for mark, split in zip(self.markings, self.splits):
            print(mark.rjust(l) + ':' + '{:0.1f}'.format(split*1000).rjust(s), 'ms')
