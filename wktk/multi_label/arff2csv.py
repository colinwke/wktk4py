"""trans multi-label *.arff file to *.csv file."""
import re


def trans_arff2csv(file_in, file_out):
    """trans *.arff file to *.csv file."""
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

    # save to file
    with open(file_out, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    from wktk.multi_label.arff2csv import trans_arff2csv

    # setting arff file path
    file_attr_in = r'D:\Downloads\birds\birds-test.arff'
    # setting output csv file path
    file_csv_out = r"D:\Downloads\birds\birds-test.csv"
    # trans
    trans_arff2csv(file_attr_in, file_csv_out)
