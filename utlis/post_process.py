import ast


def export_r3ds_format(intersects: list, export_path: str):

    output = []
    print(f'Intersects: {len(intersects)}')
    for i, shape in enumerate(intersects):
        output.append([shape.idx, 0.3, 0.4])

    with open(export_path, 'w') as file:
        file.write('[')
        for index, shape in enumerate(output):
            file.write(str(shape))
            if index < len(output) - 1:
                file.write(',')
        file.write(']')


def process_feature_txt(c_path: str, flag: list):
    '''
    c_path: canonical_feature_path

    '''

    with open(c_path, 'r') as c_data:
        c_content = ast.literal_eval(c_data.read())

    c_process_content = [
        value for i, value in enumerate(c_content) if flag[i] is True
    ]

    print(f'c_Length{len(c_process_content)}')

    with open(c_path, 'w') as file:
        file.write('[')
        for index, shape in enumerate(c_process_content):
            file.write(str(shape))
            if index < len(c_process_content) - 1:
                file.write(',')
        file.write(']')