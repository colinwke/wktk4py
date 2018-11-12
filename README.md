# wktk4py
A personal toolkit for python.

### Multi-Lable

`*.arff` file to `*.csv` file [[source]](wktk/multi_label/arff2csv.py)

``` python
from multi_label.arff2csv import trans_arff2csv

# setting arff file path
file_attr_in = r'D:\Downloads\birds\birds-test.arff'
# setting output csv file path
file_csv_out = r"D:\Downloads\birds\birds-test.csv"
# trans
trans_arff2csv(file_attr_in, file_csv_out)
```

