import numpy as np

class SimpleFourierFilter(object):
    """
    Class to apply simple Fourier Filtration to a vector

    Filter types:
        'fraction' (requires kwarg: 'fraction' to be set)
        'rule 36'  (can set kwarg: 'power' but not necessary)
    """
    def __init__(self, modes, filter_type, **kwargs):
        self.n = modes.shape[0]
        self.modes = modes
        self.filter_type = filter_type
        self._get_filter(**kwargs)
    def __call__(self, fin, input_type='space', output_type='space'):
        input_is_real = fin.dtype == float and input_type == 'space'
        if input_type=='space':
            fin = np.fft.fft(fin)
        fout = fin*self.filter
        if output_type == 'space':
            fout = np.fft.ifft(fout)
            if input_is_real:
                fout = fout.real
        return fout
    def _get_filter(self, **kwargs):
        if self.filter_type == 'fraction':
            max_k = np.abs(self.modes).max()
            self.filter = np.ones(self.n, dtype=float)
            self.filter[np.abs(self.modes) > max_k*kwargs['fraction']] = 0.0
        elif self.filter_type == 'rule 36':
            max_k = np.abs(self.modes).max()
            if 'power' in kwargs:
                power36 = kwargs['power']
            else:
                power36 = 36
            self.filter = np.exp(-power36*(np.abs(self.modes)/max_k)**power36)
        else:
            raise Exception('Filter type not defined.')



