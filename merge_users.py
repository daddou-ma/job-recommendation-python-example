import pandas as pd
import numpy as np
from clean_txt import clean_txt
from config import get_raw_file_path, get_data_file_path

# Load Raw Data Files
df_job_view = pd.read_csv(get_raw_file_path('users_views'))
df_experience = pd.read_csv(get_raw_file_path('users_experiences'))
df_poi =  pd.read_csv(get_raw_file_path('users_poi'), sep = ',')

# Process Job View Data
df_job_view = df_job_view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]
df_job_view.columns = ['applicant_id', 'job_id', 'position', 'company', 'city']
df_job_view['job_view_text'] = df_job_view['position'].map(str) + '  ' + df_job_view['company'].map(str) + ' ' + df_job_view['city'].map(str)
df_job_view['job_view_text'] = df_job_view['job_view_text'].map(str).apply(clean_txt)
df_job_view['job_view_text'] = df_job_view['job_view_text'].str.lower()
df_job_view = df_job_view.groupby('applicant_id', sort=False)['job_view_text'].apply(' '.join).reset_index()


# Process Experience Data
df_experience = df_experience[['Applicant.ID','Position.Name']]
df_experience.columns = ['applicant_id', 'experience_text']
df_experience['experience_text'] = df_experience['experience_text'].map(str).apply(clean_txt)
df_experience =  df_experience.sort_values(by='applicant_id')
df_experience = df_experience.fillna(' ')
df_experience = df_experience.groupby('applicant_id', sort=False)['experience_text'].apply(' '.join).reset_index()

# Process Position of Interest Data
df_poi = df_poi[['Applicant.ID','Position.Of.Interest']]
df_poi.columns = ['applicant_id', 'poi_text']
df_poi = df_poi.sort_values(by='applicant_id')
df_poi['poi_text'] = df_poi['poi_text'].map(str).apply(clean_txt)
df_poi = df_poi.fillna(' ')
df_poi = df_poi.groupby('applicant_id', sort=True)['poi_text'].apply(' '.join).reset_index()

# Merging
output = df_job_view.merge(df_experience, left_on='applicant_id', right_on='applicant_id', how='outer')
output = output.fillna(' ')
output = output.sort_values(by='applicant_id')

output = output.merge(df_poi, left_on='applicant_id', right_on='applicant_id', how='outer')
output = output.fillna(' ')
output = output.sort_values(by='applicant_id')

output['text'] = output['job_view_text'].map(str) + output['experience_text'].map(str) + ' ' + output['poi_text'].map(str)

output = output[['applicant_id','text']]
output['text'] = output['text'].apply(clean_txt)

output.to_csv(get_data_file_path('users'))