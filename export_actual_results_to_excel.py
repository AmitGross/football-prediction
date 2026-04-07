import pandas as pd
import numpy as np

# This script will generate an Excel file with the actual results from the WC 2022 test set.
# It reads the data/wc2022.csv file and writes it to results_wc2022_actual.xlsx.

def main():
    df = pd.read_csv('data/wc2022.csv')
    df.to_excel('results_wc2022_actual.xlsx', index=False)
    print('Excel file created: results_wc2022_actual.xlsx')

if __name__ == '__main__':
    main()
