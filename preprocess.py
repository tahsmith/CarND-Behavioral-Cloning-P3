from itertools import zip_longest
from math import ceil
import sys

from sklearn.utils import shuffle

from model import *
from pickle import dump
from operator import itemgetter

batch_size = 2 ** 10
columns = load_driving_log('data/driving_log.csv')

validation_ratio = 0.2
total_records = columns[0].shape[0]

columns = shuffle(*columns)


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    args = [iter(iterable)] * n
    for items in zip_longest(*args, fillvalue=None):
        yield tuple(item for item in items if item is not None)


validation_set_count = int(ceil(total_records * validation_ratio))
validation_set = list(map(itemgetter(slice(0, validation_set_count)), columns))
training_set_count = total_records - validation_set_count
training_set = list(map(itemgetter(slice(validation_set_count, total_records)), columns))

print('Total', total_records)
print('Validation', validation_set_count)
print('Training', training_set_count)

if not os.path.isdir('data_cache'):
    os.mkdir('data_cache')

augmented_training_set_count = 0
for i, batch in enumerate(grouper(generate_training_points(zip(*training_set)), batch_size)):
    images, steering = zip(*batch)
    images = np.stack(images)
    steering = np.array(steering)
    with open('data_cache/training-{}.p'.format(i), 'wb') as file:
        dump((images, steering), file)
    augmented_training_set_count += steering.shape[0]
    sys.stdout.write('Processed {}\n'.format(augmented_training_set_count))
    sys.stdout.flush()

for i in range(validation_set_count // batch_size + 1):
    end = min((i + 1) * batch_size, validation_set_count)
    left, center, right, steering = map(
        itemgetter(slice(i * batch_size, end)),
        training_set)
    images = crop_image(preprocess_images(load_images(center)))
    with open('data_cache/validation-{}.p'.format(i), 'wb') as file:
        dump((images, steering), file)

    sys.stdout.write('Processed {} of {}\n'.format(end, validation_set_count))
    sys.stdout.flush()

print('Wrote {} training points.'.format(augmented_training_set_count))
