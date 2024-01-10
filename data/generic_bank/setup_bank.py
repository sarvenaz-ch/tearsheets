#!/usr/bin/env python

"""
Generate data for a generic bank:

 - regions
 - branch/branch manager
 - bankers
 - clients
 - accounts
 - transaction counterparties
 - transactions
 - recommendations
"""

import numpy as np
import pandas as pd

import config


def regions(n):
   '''Return n distinct regions and regional managers'''
   x = np.arange(1, n+1, 1)
   return pd.DataFrame(np.array([x,x]).T, columns=['Region_Number', 'Regional_Manager'])


def branches(n):
   '''Return n distinct branches and branch managers'''
   x = np.arange(1, n+1, 1)
   return pd.DataFrame(np.array([x, x]).T, columns=['Branch_Number', 'Branch_Manager'])


def regions_and_branches(n_regions, n_branches):
    '''Name n_regions regions each with n_branches branches
    Todo: allow different regions to have different number of branches
    '''
    regions_ = regions(n_regions)
    df = branches(n_branches)
    df = regions_.merge(df, how='cross')

    df.Branch_Number = range(1, df.shape[0]+1, 1)
    df.Branch_Manager = range(1, df.shape[0]+1, 1)

    return df

    
def assign_personnel_to_branches(df):
    '''
    Calculates bankers per each branch. Returns to dataframes:
    1. branch-level table with headcount and number of bankers per category
    2. banker-level table with unique banker id per each banker
    '''
    headcount_per_branch = np.random.normal(config.HEADCOUNT_AVG,
        config.HEADCOUNT_STD, size=df.shape[0]).astype(int)

    df = df.assign(Headcount = headcount_per_branch)
    df = df.assign(N_deposit_ofcr = lambda x: config.f_deposit_ofcr * x.Headcount)
    df = df.assign(N_loan_ofcr = lambda x: config.f_loan_ofcr * x.Headcount)
    df = df.assign(N_wealth_ofcr = lambda x: config.f_wealth_ofcr * x.Headcount)

    df = df.round(0)

    # make dataframe of bankers at each branch
    banker_df = pd.DataFrame()
    for idx, row in df.iterrows():
        n_dep = int(row.N_deposit_ofcr)
        n_rm = int(row.N_loan_ofcr)
        n_wa = int(row.N_wealth_ofcr)
        br = row.Branch_Number
        n = int(row.Headcount)

        data_dict = {'Branch_Number': n*[br],
                     'Banker_Type': n_dep*['deposit'] + n_rm*['loan'] + n_wa*['wealth'],
                     'Banker_ID': range(1,n+1,1)
                     }

        row_df = pd.DataFrame(data_dict)

        banker_df = pd.concat([banker_df, row_df])

    # unique banker id
    banker_df.Banker_ID = range(1,banker_df.shape[0]+1,1)

    return df.astype(int), banker_df


def clients(n):
    '''
    Allocate clients of several types (person, finance, non-finance business, nonprofit)
    according to probabilities set in config.
    '''

    n = int(n)
    client_type = n*['']
    product_mix = n*['']
    random_values = np.random.randint(1,100+1, size=n)/100  # decimal probs
    random_mixes = np.random.choice(config.product_mix, size=n)  # decimal probs

    for idx, rv in enumerate(random_values):
        if rv <= config.p_person:  # person
            client_type[idx] = 'Person'
        elif config.p_person < rv <= (config.p_person + config.p_fin_business):
            client_type[idx] = 'Business - Finance'
        elif (config.p_person + config.p_fin_business) < rv <= (config.p_person + config.p_fin_business + config.p_nonfin_business):
            client_type[idx] = 'Business - Other'
        else:
            client_type[idx] = 'School/Non-Profit'

        product_mix[idx] = random_mixes[idx]

    df = pd.DataFrame(np.array([range(1,n+1, 1), client_type, product_mix]).T, columns=['Client_ID', 'Client_Type', 'Product_Mix'])

    return df


def households(df):
    '''
    Allocate households given client dataframe. Schools/nonprofits are always in their own household. The remaining clients are grouped together according to the hh_size_distribution in the config (e.g., 60% of households are singletons).
    '''

    # non-profits are automatically separate households
    households = df.query('Client_Type == "School/Non-Profit"')
    households = households.assign(Household_ID = range(1,households.shape[0]+1,1))

    # give people and businesses a change to be grouped
    person_df = df.query('Client_Type == "Person"')
    business_df = df.query('Client_Type in ["Business - Finance", "Business - Other"]')

    hh_sizes = np.random.choice(config.hh_size_distribution, size=df.shape[0])

    def _allocate_hh(client_df, sizes, households):
        '''
        Helper function to allocate client subsets into households.
        '''
        index = 0  # iterate 
        for rv in sizes:  # rv = household size
            if index > client_df.shape[0]:  # end of client list
                break

            next_id = households.Household_ID.max() + 1  # next Household_ID
            
            next_hh = client_df.iloc[index:index+rv]
            next_hh = next_hh.assign(Household_ID = next_id)  # apply Household_ID

            households = pd.concat([households, next_hh])
            index += rv  # increment index by household size

        return households

    # Person clients
    households = _allocate_hh(person_df, hh_sizes, households)

    # Business clients
    households = _allocate_hh(business_df, hh_sizes[::-1], households)
    
    return households


def account_types():
    '''
    Table of account categories (DLW) and account types. Frequencies for individuals and organizations are defined in config.py.
    '''
    # TODO create this table from config dict instead
    data_dict = {'Account_Category': 3*['Deposits'] + 5*['Loans'] + 2*['Wealth'],
                 'Account_Type': ['CHK', 'SV', 'CD'] + ['SFR', 'HELOC', 'MF', 'CRE', 'LOC'] + ['FRIM', 'FRS'],
                 'Account_Description': ['Checking', 'Saving', 'Certificate of Deposit'] +
                                        ['Single Family Residential', 'Home Equity Line of Credit', 'Multifamily',
                                         'Commercial Real Estate', 'Line of Credit'] +
                                        ['FRIM', 'FRS']
                }

    accts_df = pd.DataFrame(data_dict)
    return(accts_df)


def assign_accounts_to_clients(df):
    '''
    Use clients(n) as input.
    Assigns accounts to clients based on their product mix. Returns account-level table with unique account id. Use account frequencies defined in config.py, separate for Individual / Organizations. 
    '''
    accounts_df = pd.DataFrame()
    for idx, row in df.iterrows():
        # TODO assumes 1 acct per category - can randomly select multiple accounts here
        # can sample from distr with mean 2 std 1, threshold of min 1 if acct cat is present
        n_D=0; n_L=0; n_W=0
        if 'D' in row.Product_Mix: n_D+=1
        if 'L' in row.Product_Mix: n_L+=1
        if 'W' in row.Product_Mix: n_W+=1
        cl = row.Client_ID
        clt = row.Client_Type
        n = int(sum([n_D, n_L, n_W]))
        
        data_dict = {'Client_ID': n*[cl],
                     'Client_Type': n*[clt],  # for debugging
                     'Account_Category': n_D*['Deposits'] + n_L*['Loans'] + n_W*['Wealth'],
                     'Account_ID': range(1,n+1,1)
                     }
        row_df = pd.DataFrame(data_dict)
        # TODO fix frequencies for f_accounts from Sergey's query; also use one dict for them all
        acct_val = []
        for idx, row_ in row_df.iterrows():
            if row.Client_Type == 'Person':
                acct_val_sel = np.random.choice(
                    list(config.f_indiv_accts[row_['Account_Category']].keys()), 
                    size=1, p=list(config.f_indiv_accts[row_['Account_Category']].values()))
            else:
                acct_val_sel = np.random.choice(
                    list(config.f_org_accts[row_['Account_Category']].keys()), 
                    size=1, p=list(config.f_org_accts[row_['Account_Category']].values()))
            acct_val.append(acct_val_sel[0])
        row_df = row_df.assign(Account_Type = acct_val)
        accounts_df = pd.concat([accounts_df, row_df])

        
    # unique account id per acct category (i.e. D#, L#, W#)
    n_acct_cat = accounts_df.groupby('Account_Category').size()
    for idx_cat in accounts_df['Account_Category'].unique():
        accounts_df.loc[accounts_df.Account_Category == idx_cat, 'Account_ID'] = [idx_cat[0] + f"{k}" for k in range(1, n_acct_cat[idx_cat]+1, 1)]
 
    return(accounts_df)


if __name__ == '__main__':
    import setup_bank as m

    n_regions, n_branches, n_clients = 1, 1, 1e3

    # assign branches in two passes:
    # first pass: branches and regions
    # second pass: headcount per branch
    branches_df = m.regions_and_branches(n_regions, n_branches)
    branches_df, bankers_df = m.assign_personnel_to_branches(branches_df)

    # allocate clients
    clients_df = m.clients(n_clients)

    # group into households
    hh_df = m.households(clients_df)
