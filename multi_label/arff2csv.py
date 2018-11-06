"""trans multi-label *.arff file to *.csv file."""
import re


def parse_arff(file_in):
    """parse *.arff file.
    :param file_in: *.arff file to parse.
    :return content: the csv format content.
    """
    columns = []
    data = []
    with open(file_in, 'r') as f:
        data_flag = 0
        for line in f:
            if line[:2] == '@a':
                # find indices
                indices = [i for i, x in enumerate(line) if x == ' ']
                columns.append(re.sub(r'^[\'\"]|[\'\"]$|\\+', '', line[indices[0] + 1:indices[-1]]))
            elif line[:2] == '@d':
                data_flag = 1
            elif data_flag == 1:
                data.append(line)

    content = ','.join(columns) + '\n' + ''.join(data)
    return content


def save_file(content, file_out):
    """save file.
    :param content: content to save.
    :param file_out: save path.
    """
    with open(file_out, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    # setting arff file path
    file_attr_in = r'D:\Downloads\birds\birds-test.arff'
    # setting output csv file path
    file_csv_out = r"D:\Downloads\birds\birds-test.csv"
    # trans
    save_file(parse_arff(file_attr_in), file_csv_out)
