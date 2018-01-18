# -*- coding: utf-8 -*-
import numpy as np


def generator(arrays, batch_size):
  """Generate batches, one with respect to each array's first axis.

  Reference:
    https://github.com/edwardlib/observations#how-do-i-use-minibatches-of-data
  """

  starts = [0] * len(arrays)  # pointers to where we are in iteration
  while True:
    batches = []
    for i, array in enumerate(arrays):
      start = starts[i]
      stop = start + batch_size
      diff = stop - array.shape[0]
      if diff <= 0:
        batch = array[start:stop]
        starts[i] += batch_size
      else:
        batch = np.concatenate((array[start:], array[:diff]))
        starts[i] = diff
      batches.append(batch)
    yield batches
