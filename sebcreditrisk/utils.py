import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from os.path import join
from pathlib import Path

# Root, data, and result directories
project_root = str(Path(__file__).parent.parent)
data_dir = join(project_root, 'data')
raw_data_dir = join(data_dir, 'raw')
result_dir = join(project_root, 'results')

# Raw migration data
gcd_sample_raw_file_path = join(raw_data_dir, 'Mig_Mat_Nordic_SME_LC_RE.xlsx')
gcd_raw_file_path = join(raw_data_dir, 'gcd_nordic_migration.csv')
seb_raw_file_path = join(raw_data_dir, 'seb_raw_data.xlsx')

# Raw indicator data
indicator_small_raw_path = join(raw_data_dir, 'systemic_indicators_small_2.xlsx')
indicator_large_raw_path = join(raw_data_dir, 'Swe_Fin_Nor_Den_EconomicData_WorldBank_2280395.xlsx')

# Processed migration data
gcd_sample_path = join(data_dir, 'gcd_sample.csv')
gcd_data_path = join(data_dir, 'gcd_data.csv')
seb_data_path = join(data_dir, 'seb_data.csv')

# Processed indicator data
indicator_small_path = join(data_dir, 'indicator_small.csv')
indicator_large_path = join(data_dir, 'indicator_large.csv')
indicator_pca_path = join(data_dir, 'indicator_pca.csv')

# Rating and industry code data
nace_gcd_raw_path = join(raw_data_dir, 'GCD_nace.csv')
rating_table_path = join(data_dir, 'rating_table.csv')
nace_seb_table_path = join(data_dir, 'nace_seb_table.csv')
nace_gcd_table_path = join(data_dir, 'nace_gcd_table.csv')

# Custom industry code data
code_gcd_path = join(data_dir, 'code_table_gcd.csv')
code_seb_path = join(data_dir, 'code_table_seb.csv')

# Result files
portfolio_result_path = join(result_dir, 'portfolio_rmse.csv')
portfolio_kf_result_path = join(result_dir, 'portfolio_kf_rmse.csv')
sector_result_path = join(result_dir, 'sector_rmse.csv')
sector_grouped_result_path = join(result_dir, 'sector_grouped_rmse.csv')
country_sector_result_path = join(result_dir, 'country_sector_rmse.csv')

# Path dictionaries
raw_path_dict = {'gcd_sample': gcd_sample_raw_file_path, 'gcd': gcd_raw_file_path,  'seb': seb_raw_file_path, 'nace_gcd': nace_gcd_raw_path, 
                    'indicator_small': indicator_small_raw_path, 'indicator_large': indicator_large_raw_path}
processed_path_dict = {'gcd_sample': gcd_sample_path, 'gcd':gcd_data_path, 'seb': seb_data_path, 
                        'rating': rating_table_path, 'nace_seb': nace_seb_table_path, 'nace_gcd': nace_gcd_table_path, 
                        'code_gcd': code_gcd_path, 'code_seb': code_seb_path, 
                        'indicator_small': indicator_small_path, 'indicator_large': indicator_large_path, 'indicator_pca': indicator_pca_path, 'indicator_pca': indicator_pca_path, 
                        'portfolio_result': portfolio_result_path, 'portfolio_kf': portfolio_kf_result_path, 
                        'sector_result': sector_result_path, 'sector_grouped_result': sector_grouped_result_path, 'country_sector_result': country_sector_result_path}

code_types = ['0', '1', '2', 'nace']

def load_raw_data_file(dataset):
    '''Loading raw data files in pandas ExcelFile
    '''
    return pd.ExcelFile(raw_path_dict[dataset])


def process_data(dataset=[], rating='seb'):
    '''Main function for processing the data, the following was done: 
        removing unnecessary columns,
        renaming columns, 
        mapping ratings and industries, 
        filtering and normalizing indicators, 
        applying PCA on the large indicator set.
    The function body is removed to comply with NDAs.
    '''
    pass
        

def load_data(dataset):
    return pd.read_csv(processed_path_dict[dataset], keep_default_na=False)


def get_gcd_transition_matrix(year=[], quarter=[4], asset_class=[], country=[], industry=[], code=0, rating='seb', count=True, sample=False):
    '''
    Function for creating a transition matrix for given years, quarter, asset class, country, industry from GCD data set
    '''

    if sample:
        df = load_data('gcd_sample')
    else:
        df = load_data('gcd')
    code_dict = {0: 'industry_0', 1: 'industry_gcd'}

    # Filtering
    if year:
        df = df[df['year'].isin(year)]
    if quarter:
        df = df[df['quarter'].isin(quarter)]
    if asset_class:
        df = df[df['asset_class'].isin(asset_class)]
    if country:
        df = df[df['country'].isin(country)]
    if industry:
        df = df[df[code_dict[code]].isin(industry)]

    # Creating the transition matrix
    if rating == 'seb':
        matrix_size = (15,16)
    else:
        matrix_size = (24,24)

    m = np.zeros(matrix_size, dtype=int)

    for i in range(1, matrix_size[0] + 1):
        for j in range(1, matrix_size[1] + 1):
            df_ = df[df['t1']==i]
            df_ = df_[df_['t2']==j]
            m[i-1,j-1] = int(df_['count'].sum())

    if not count:
        m = (m.T / np.sum(m, axis=1)).T

    return m


def get_seb_transition_matrix(year=[], industry=[], country=[], code=0, count=True):
    '''
    Function for creating a transition matrix for given years, quarter, asset class, country, industry from SEB data set
    '''
    df = load_data('seb')
    code_dict = {0: 'industry_0', 1: 'industry_1', 2: 'industry_2'}

    # Filtering
    if year:
        df = df[df['year'].isin(year)]
    if country:
        df = df[df['country'].isin(country)]
    if industry:
        df = df[df[code_dict[code]].isin(industry)]

    # Creating the transition matrix
    matrix_size = (15,16)
    m = np.zeros(matrix_size, dtype=int)

    for i in range(1, matrix_size[0] + 1):
        for j in range(1, matrix_size[1] + 1):
            df_ = df[df['t1']==i]
            df_ = df_[df_['t2']==j]
            m[i-1,j-1] = int(df_['count'].sum())

    if not count:
        m = (m.T / np.sum(m, axis=1)).T

    return m


def get_code_name(code, table, code_type=0):
    '''
    Helper function for getting industry names from the industry code table 
    '''

    code_col = 'code_' + code_types[code_type]
    name_col = 'name_' + code_types[code_type]
    code_table = load_data(table)[[code_col, name_col]]
    code_table = code_table[code_table[code_col]==code]
    code_table.drop_duplicates(inplace=True)
    return list(code_table[name_col])


def get_px(m):
    '''
    Proportion of ratings, p(X=x)
    '''

    m = m + 1e-6
    return np.sum(m, axis=1) / np.sum(m)


def get_pdx(m):
    '''
    Conditional default probability knowing rating, p(D=1|X=x)
    '''

    m = m + 1e-10
    return m[:,-1] / (np.sum(m, axis=1) + 1e-4)


def get_pd(m):
    '''
    Marginal probability of default p(D=1)
    '''

    m = m + 1e-10  
    return np.sum(get_px(m) * get_pdx(m)) + 1e-10 
    
    
def get_transitions(data, years=[2008, 2020], country=[], quarter=[4], industry=[], code=0):
    '''
    Getting list of transition matrices of portfolio, country, or industry in chronological order
    '''
    
    M = []
    
    for year in list(range(*years)):
        if data == 'seb':
            M.append(get_seb_transition_matrix(year=[year], country=country, industry=industry, code=code))
        elif data == 'gcd':
            M.append(get_gcd_transition_matrix(year=[year], quarter=quarter, country=country, industry=industry, code=code))
        elif dataset == 'gcd_sample':
            M.append(get_gcd_transition_matrix(year=[year], quarter=quarter, country=country, industry=industry, code=code, sample=True))
            
    return np.array(M, dtype=int)
    
    
def get_z(data, years=[2008, 2020], country='SE', pca_n=0):
    '''
    Getting the Z factor matrix (shape = years x num_indicators)
    '''
    
    years = list(range(*years))
    
    if data == 'large':
        df = load_data('indicator_large')
    elif data == 'pca':
        df  = load_data('indicator_pca')
    else:
        df = load_data('indicator_small')
        if data == 'single':
            df = df[['country', 'year', 'GDP']]           
            
    df = df[df['country'] == country]
    df = df[df['year'].isin(years)]
    df = df.sort_values('year')
    df = df.drop(columns=['country', 'year'])
    Z = df.to_numpy()
    
    if data == 'pca' and pca_n > 0:
        Z = Z[:, :pca_n]
    return Z
    