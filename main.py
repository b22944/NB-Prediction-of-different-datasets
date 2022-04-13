import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import random

def gotoClass():
    '''
    For reading the CSV file you will enter
    name of the file in file_name variable
    '''
    file_name = "data1"
    my_file = pd.read_csv(f"{file_name}.csv")
    print(my_file)

    # In this step we are encoding string to numeric
    nums = LabelEncoder()

    # In this step we are dropping condition value
    data_value = my_file.drop("Class", axis='columns')
    condition = my_file["Class"]
    # print(condition)

    # In this step we are creating new data values for numerics
    data_value["weather"] = nums.fit_transform(data_value["Weather"])
    data_value["temprature"] = nums.fit_transform(data_value["Temprature"])
    data_value["humidity"] = nums.fit_transform(data_value["Humidity"])
    data_value["wind"] = nums.fit_transform(data_value["Wind"])
    # print(data_value)

    # In this step dropping the string Values of numerics
    data_values = data_value.drop(["Weather", "Temprature", "Humidity", "Wind"], axis="columns")
    # print(data_values)

    # prediction
    c = GaussianNB()
    c.fit(data_values, condition)
    c.score(data_values, condition)

    # In this step we are getting the random values from the dataset
    p_weather = random.choice([0, 1, 2])
    p_temprature = random.choice([0, 1, 2])
    p_humidity = random.choice([0, 1])
    p_wind = random.choice([0, 1])

    print(f" Weather = {p_weather}\n Temprature = {p_temprature}\n Humidity = {p_humidity}\n Wind = {p_wind}")

    # finding the answer
    v = c.predict([[p_weather, p_temprature, p_humidity, p_wind]])

    if v == "yes":
        print("You can take the class today")

    else:
        print("You should stay home today")

''' End code go to class '''

def isMammal():
    '''
    For reading the CSV file you will enter
    name of the file in file_name variable
    '''
    file_name = "data2"
    my_file2 = pd.read_csv(f"{file_name}.csv")
    #print(my_file2)

    # In this step we are encoding string to numeric
    nums2 = LabelEncoder()

    # In this step we are dropping condition value
    data_value2 = my_file2.drop("class", axis='columns')
    condition2 = my_file2["class"]
    #print(condition2)

    # In this step we are creating new data values for numerics
    data_value2["name"] = nums2.fit_transform(data_value2["Name"])
    data_value2["fly"] = nums2.fit_transform(data_value2["Fly"])
    data_value2["water"] = nums2.fit_transform(data_value2["Water"])
    data_value2["legs"] = nums2.fit_transform(data_value2["Legs"])
    #print(data_value2)

    # In this step dropping the string Values of numerics
    data_values2 = data_value2.drop(["Name", "Fly", "Water", "Legs"], axis="columns")
    print(data_values2)

    # prediction
    c2 = GaussianNB()
    c2.fit(data_values2, condition2)
    c2.score(data_values2, condition2)

    # In this step we are getting the random values from the dataset
    p_name = random.randint(0,9)
    p_fly = random.choice([0, 1])
    p_water = random.choice([0, 1])
    p_legs = random.choice([0, 1])

    print(f" Name = {p_name}\n Fly = {p_fly}\n water = {p_water}\n Legs = {p_legs}")

    # finding the answer
    v = c2.predict([[p_name, p_fly, p_water, p_legs]])
    if v == "yes":
        print("Yes It is a Mammal")

    else:
        print("No it is not a Mammal")


#driver code
print("Naive Byese Prediction of Different Datasets")
print("Usama Hassan (Group Leader) [B-22944])")
print("Hamza Javaid [B-22685])")
print("Hammad Shahid [B-21817])\n")

print("Press 1 for GO to the class Prediction")
print("Press 2 for Mammal prediction \n")

choice = int(input("Enter your choice: "))

if choice == 1:
    gotoClass()
elif choice == 2:
    isMammal()
else: print("Wrong Choice")