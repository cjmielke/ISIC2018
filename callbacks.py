import collections
import csv
import time

from keras.callbacks import Callback, CSVLogger
import numpy as np

# this didn't work out for non-numeric fields .....
def buildMetricCallback(args, model):

    class AddMetrics(Callback):
        def on_epoch_end(self, epoch, logs):
            print(vars(args))
            for k, v in vars(args).items():
                logs[k] = v
            #logs['params'] = model.count_params()

    return AddMetrics()


# hacky logging
class MyCSVLogger(CSVLogger):
    hyperparams=None
    otherfields = {}


    def __init__(self, filename):
        self.startime = time.time()
        super(MyCSVLogger, self).__init__(filename)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch', 'time'] + self.keys + list(vars(self.hyperparams).keys()) + list(self.otherfields.keys())

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        row_dict.update(vars(self.hyperparams))
        row_dict.update(self.otherfields)
        row_dict['time'] = time.time() - self.startime
        self.writer.writerow(row_dict)
        self.csv_file.flush()
