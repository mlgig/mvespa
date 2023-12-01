import pandas as pd

def show(result_dfs):
    all_results = pd.concat(result_dfs, ignore_index=True)
    sorted_df = all_results.sort_values(by=['Metric', 'Task'],
                                        ignore_index=True)
    mi = pd.MultiIndex.from_frame(sorted_df[['Metric', 'Task', 'PTM Ref']])
    sorted_df.drop(columns=['Metric', 'Task', 'PTM Ref'],
                   inplace=True)
    return sorted_df.set_index(mi)