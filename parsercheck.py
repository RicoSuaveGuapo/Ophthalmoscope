import argparse

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wut', type=bool,default=False)

    return parser

def check_argparse(args):
    print(args.wut)


if __name__ == "__main__":
    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)