import os

def make_input():
    dataset='lerf_ovs/waldo_kitchen'
    input_path=os.path.join('data', dataset, 'images')
    output_path=os.path.join('data', dataset, 'input')
    inputs=os.listdir(input_path)
    os.makedirs(output_path, exist_ok=True)
    for input in inputs:
        with open(os.path.join(input_path, input), 'rb') as fin, open(os.path.join(output_path, input[6:]), 'wb') as fout:
            fout.write(fin.read())
def main()->None:
    make_input()

if __name__ == '__main__':
    main()