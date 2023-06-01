import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

   
    process = dict()
    process['classification'] = import_class('process.classification.CLS_Process')

    subparsers = parser.add_subparsers(dest='process')
    for k, p in process.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    arg = parser.parse_args()

    # start
    Process = process[arg.process]

    p = Process(sys.argv[2:])

    p.start() 
