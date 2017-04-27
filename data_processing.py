import csv

def data_processing(filename):
    # Map the categorical attributes in the data set to number
    workclass_list = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    workclass_dict = {i:workclass_list.index(i) for i in workclass_list}
    education_list = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
    education_dict = {i:education_list.index(i) for i in education_list}
    marital_list = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    marital_dict = {i:marital_list.index(i) for i in marital_list}
    occupation_list = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
    occupation_dict = {i:occupation_list.index(i) for i in occupation_list}
    relationship_list = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    relationship_dict = {i:relationship_list.index(i) for i in relationship_list}
    race_list = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    race_dict = {i:race_list.index(i) for i in race_list}
    sex_list = ["Female", "Male"]
    sex_dict = {i:sex_list.index(i) for i in sex_list}
    country_list = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    country_dict = {i:country_list.index(i) for i in country_list}
    income_dict = {">50K":0, "<=50K":1, ">50K.":0, "<=50K.":1}

    dict_list = [{}, workclass_dict, {}, education_dict, {}, marital_dict, occupation_dict, relationship_dict, race_dict, sex_dict, {}, {}, {}, country_dict, income_dict]

    with open(filename) as f:
        # Format the data into a .csv file.
        with open(filename + '.csv', 'w') as output:
            writer = csv.writer(output)
            for i in f.readlines():
                try:
                    writer.writerow([y[x.strip()] if y else x for x,y in zip(i.split(','), dict_list)])
                except KeyError:
                    pass


if __name__ == '__main__':
    data_processing("adult.data")
    data_processing("adult.test")
