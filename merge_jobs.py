import pandas as pd
import numpy as np
from clean_txt import clean_txt
from config import get_raw_file_path, get_data_file_path

# Load Raw Data Files
df_jobs = pd.read_csv(get_raw_file_path('jobs'))

# Process Job Data
df_jobs = df_jobs[['Job.ID']+['Title']+['Position']+ ['Company']+['City']+['Employment.Type']+['Job.Description']]
df_jobs.columns = ['job_id', 'title', 'position', 'company','city', 'employment_type','job_description']

df_jobs['company'] = df_jobs['company'].replace(['Genesis Health Systems'], 'Genesis Health System')

df_jobs.loc[df_jobs.company == 'CHI Payment Systems', 'city'] = 'Illinois'
df_jobs.loc[df_jobs.company == 'Academic Year In America', 'city'] = 'Stamford'
df_jobs.loc[df_jobs.company == 'CBS Healthcare Services and Staffing ', 'city'] = 'Urbandale'
df_jobs.loc[df_jobs.company == 'Driveline Retail', 'city'] = 'Coppell'
df_jobs.loc[df_jobs.company == 'Educational Testing Services', 'city'] = 'New Jersey'
df_jobs.loc[df_jobs.company == 'Genesis Health System', 'city'] = 'Davennport'
df_jobs.loc[df_jobs.company == 'Home Instead Senior Care', 'city'] = 'Nebraska'
df_jobs.loc[df_jobs.company == 'St. Francis Hospital', 'city'] = 'New York'
df_jobs.loc[df_jobs.company == 'Volvo Group', 'city'] = 'Washington'
df_jobs.loc[df_jobs.company == 'CBS Healthcare Services and Staffing', 'city'] = 'Urbandale'

df_jobs['employment_type'] = df_jobs['employment_type'].fillna('Full-Time/Part-Time')

df_jobs["text"] =  df_jobs["position"].map(str) + " " + df_jobs["company"].map(str) + " " + df_jobs["city"].map(str) + " " + df_jobs['employment_type'].map(str) + " " + df_jobs['job_description'].map(str) + " " + df_jobs['title'].map(str)
df_jobs['text'] = df_jobs['text'].apply(clean_txt)
df_jobs = df_jobs[['job_id', 'text']]

df_jobs.to_csv(get_data_file_path('jobs'))