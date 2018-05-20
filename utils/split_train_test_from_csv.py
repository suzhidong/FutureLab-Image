import random
import argparse
import os

parser = argparse.ArgumentParser(description='Parser for all the training options')
parser.add_argument('--train_ratio', type=float, default=0.9)
parser.add_argument('--csv_source', type=str, default='./data/image_scene_training/training-list-0511.csv')
parser.add_argument('--dst_location', type=str, default='./data/image_scene_training/')
args = parser.parse_args()

if __name__ == '__main__':
    lines = open(args.csv_source).readlines()

    train_set = open(os.path.join(args.dst_location, 'train.txt'), 'w')
    test_set = open(os.path.join(args.dst_location, 'test.txt'), 'w')
    for i in range(1, len(lines)):
        if random.random() <= args.train_ratio:
            train_set.write(lines[i].replace(',', '.jpg '))
        else:
            test_set.write(lines[i].replace(',', '.jpg '))

