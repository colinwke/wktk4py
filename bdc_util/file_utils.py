def read_comment_data(path):
    """ 读取带有注释的数据(使用"#"进行注释) """
    pieces = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '': continue
            shape_index = line.find('#')
            if shape_index is -1:
                pieces.append(line)
            elif shape_index is 0:
                continue
            else:
                pieces.append(line[:shape_index])
    return pieces


def get_folder_files(folder_path, file_suffix={'txt', 'csv', 'dat'}):
    """ 获取目录下的所有文件 """
    from os import listdir
    from os.path import isfile, join

    if isinstance(file_suffix, str):
        file_suffix = {file_suffix}

    if isinstance(file_suffix, set):
        files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.split('.')[-1] in file_suffix]
        folder_files = [join(folder_path, f) for f in files]
    else:
        raise Exception("""file suffix error, do like file_suffix={'txt', 'csv', 'dat'}""")

    print('\n'.join(['in folder: %s' % folder_path] + ['    |%s' % x for x in files]))
    return folder_files
