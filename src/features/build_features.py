import pandas as pd
import numpy as np
import re


def read_raw_dataset(path):
    return pd.read_csv(path)


def transform_raw_dataset(data):
    data = data.rename(columns={'Unnamed: 0': 'id'})
    data = data.assign(collector=data.collector.astype('category')) \
        .assign(country=data.country.astype('category')) \
        .assign(un_subregion=data.un_subregion.astype('category')) \
        .assign(so_region=data.so_region.astype('category')) \
        .assign(gender=data.gender.astype('category')) \
        .assign(occupation=data.occupation.astype('category')) \
        .assign(occupation_group=data.occupation_group.astype('category')) \
        .assign(aliens=data.aliens.astype('category')) \
        .assign(employment_status=data.employment_status.astype('category')) \
        .assign(industry=data.industry.astype('category')) \
        .assign(remote=data.remote.astype('category')) \
        .assign(job_discovery=data.job_discovery.astype('category')) \
        .assign(commit_frequency=data.commit_frequency.astype('category')) \
        .assign(dogs_vs_cats=data.dogs_vs_cats.astype('category')) \
        .assign(desktop_os=data.desktop_os.astype('category')) \
        .assign(why_learn_new_tech=data.why_learn_new_tech.astype('category')) \
        .assign(open_to_new_job=data.open_to_new_job.astype('category')) \
        .assign(job_search_annoyance=data.job_search_annoyance.astype('category')) \
        .assign(star_wars_vs_star_trek=data.star_wars_vs_star_trek.astype('category'))

    data = data.pipe((transform_partially_ordinal_column, 'data'),
                     col='age_range',
                     categories=['< 20', '20-24', '25-29',
                                 '30-34', '35-39', '40-49',
                                 '50-59', '> 60']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='experience_range',
              categories=['Less than 1 year',
                          '1 - 2 years',
                          '2 - 5 years',
                          '6 - 10 years',
                          '11+ years']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='salary_range',
              categories=['Less than $10,000', '$10,000 - $20,000', '$20,000 - $30,000',
                          '$30,000 - $40,000', '$40,000 - $50,000', '$50,000 - $60,000',
                          '$60,000 - $70,000', '$70,000 - $80,000', '$80,000 - $90,000',
                          '$90,000 - $100,000', '$100,000 - $110,000', '$110,000 - $120,000',
                          '$120,000 - $130,000', '$130,000 - $140,000', '$140,000 - $150,000',
                          '$150,000 - $160,000', '$160,000 - $170,000', '$170,000 - $180,000',
                          '$180,000 - $190,000', '$190,000 - $200,000', 'More than $200,000']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='company_size_range',
              categories=['1-4 employees', '5-9 employees', '10-19 employees',
                          '20-99 employees', '100-499 employees', '500-999 employees',
                          '1,000-4,999 employees', '5,000-9,999 employees', '10,000+ employees']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='team_size_range',
              categories=['1-4 people', '5-9 people', '10-14 people',
                          '15-20 people', '20+ people']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='hobby',
              categories=['1-2 hours per week', '2-5 hours per week',
                          '5-10 hours per week', '10-20 hours per week',
                          '20+ hours per week']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='rep_range',
              categories=['1', '2 - 100', '101 - 500',
                          '501 - 1,000', '5,001 - 10,000',
                          '1,001 - 5,000', '10,001+']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='interview_likelihood',
              categories=['0%', '10%', '20%', '30%', '40%',
                          '50%', '60%', '70%', '80%', '90%', '100%']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='job_satisfaction',
              categories=['I hate my job', "I'm somewhat dissatisfied with my job",
                          "I'm neither satisfied nor dissatisfied", "I'm somewhat satisfied with my job",
                          'I love my job']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='unit_testing',
              categories=['No', "I don't know", 'Yes']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='visit_frequency',
              categories=['I have never been on Stack Overflow. I just love taking surveys.',
                          'Very rarely', 'Once a week', 'Once a day', 'Multiple times a day']) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='programming_ability',
              categories=[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]) \
        .pipe((transform_partially_ordinal_column, 'data'),
              col='women_on_team',
              categories=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11+']) \
        .pipe((transform_is_important_column, 'data'), col='important_variety') \
        .pipe((transform_is_important_column, 'data'), col='important_control') \
        .pipe((transform_is_important_column, 'data'), col='important_sameend') \
        .pipe((transform_is_important_column, 'data'), col='important_newtech') \
        .pipe((transform_is_important_column, 'data'), col='important_buildnew') \
        .pipe((transform_is_important_column, 'data'), col='important_buildexisting') \
        .pipe((transform_is_important_column, 'data'), col='important_promotion') \
        .pipe((transform_is_important_column, 'data'), col='important_companymission') \
        .pipe((transform_is_important_column, 'data'), col='important_wfh') \
        .pipe((transform_is_important_column, 'data'), col='important_ownoffice') \
        .pipe((transform_agree_column, 'data'), col='agree_tech') \
        .pipe((transform_agree_column, 'data'), col='agree_notice') \
        .pipe((transform_agree_column, 'data'), col='agree_problemsolving') \
        .pipe((transform_agree_column, 'data'), col='agree_diversity') \
        .pipe((transform_agree_column, 'data'), col='agree_adblocker') \
        .pipe((transform_agree_column, 'data'), col='agree_alcohol') \
        .pipe((transform_agree_column, 'data'), col='agree_loveboss') \
        .pipe((transform_agree_column, 'data'), col='agree_nightcode') \
        .pipe((transform_agree_column, 'data'), col='agree_legacy') \
        .pipe((transform_agree_column, 'data'), col='agree_mars')

    data = data.pipe((transform_multioptional_column, 'data'), col='self_identification') \
        .pipe((transform_multioptional_column, 'data'), col='tech_do') \
        .pipe((transform_multioptional_column, 'data'), col='tech_want') \
        .pipe((transform_multioptional_column, 'data'), col='dev_environment') \
        .pipe((transform_multioptional_column, 'data'), col='education') \
        .pipe((transform_multioptional_column, 'data'), col='new_job_value') \
        .pipe((transform_multioptional_column, 'data'), col='how_to_improve_interview_process') \
        .pipe((transform_multioptional_column, 'data'), col='developer_challenges') \
        .pipe((transform_multioptional_column, 'data'), col='why_stack_overflow')

    data = data.drop(['age_midpoint', 'experience_midpoint', 'salary_midpoint', 'big_mac_index'], axis=1)
    return data


def transform_agree_column(col, data):
    return transform_partially_ordinal_column(col, data,
                                              categories=['Disagree completely', 'Disagree somewhat',
                                                          'Neutral', 'Agree somewhat', 'Agree completely'])


def transform_is_important_column(col, data):
    return transform_partially_ordinal_column(col, data,
                                              categories=["I don't care about this",
                                                          'This is somewhat important',
                                                          'This is very important'])


def transform_partially_ordinal_column(col, data, categories):
    ordered = data[col].astype('category', categories=categories, ordered=True)
    new_columns = {left: col + '_' + re.sub('\W', '', left.lower()) for left in
                   set(data[col].dropna().unique()) - set(categories)}
    data_copy = data
    for val, new_col in new_columns.items():
        data_copy = data_copy.assign(**{new_col: data[col] == val})
    return data_copy.assign(**{col: ordered})


def transform_multioptional_column(col, data):
    def column_name(value):
        return col + '_' + re.sub('\W', '', re.sub('\(.*?\)', '', value.strip())).lower()

    splitted = [x if type(pd.notnull(x)) != bool else [] for x in data[col].str.split(';').values]
    splitted = [[y.strip() for y in x] for x in splitted]
    values = {s for s in np.hstack(splitted)}
    data_copy = data
    for value in values:
        data_copy = data_copy.assign(**{
            column_name(value): [1 if value in x else 0 for x in splitted]})
    return data_copy.drop(col, axis=1)
