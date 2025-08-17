import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

median_age = train_data['Age'].median()
train_data['Age'] = train_data['Age'].fillna(median_age)
test_data['Age'] = test_data['Age'].fillna(median_age)

median_RoomService = train_data['RoomService'].median()
train_data['RoomService'] = train_data['RoomService'].fillna(median_RoomService)
test_data['RoomService'] = test_data['RoomService'].fillna(median_RoomService)

median_FoodCourt = train_data['FoodCourt'].median()
train_data['FoodCourt'] = train_data['FoodCourt'].fillna(median_FoodCourt)
test_data['FoodCourt'] = test_data['FoodCourt'].fillna(median_FoodCourt)

median_ShoppingMall = train_data['ShoppingMall'].median()
train_data['ShoppingMall'] = train_data['ShoppingMall'].fillna(median_ShoppingMall)
test_data['ShoppingMall'] = test_data['ShoppingMall'].fillna(median_ShoppingMall)

median_Spa = train_data['Spa'].median()
train_data['Spa'] = train_data['Spa'].fillna(median_Spa)
test_data['Spa'] = test_data['Spa'].fillna(median_Spa)

median_VRDeck = train_data['VRDeck'].median()
train_data['VRDeck'] = train_data['VRDeck'].fillna(median_VRDeck)
test_data['VRDeck'] = test_data['VRDeck'].fillna(median_VRDeck)

mode_HomePlanet = train_data['HomePlanet'].mode()[0]
train_data['HomePlanet'] = train_data['HomePlanet'].fillna(mode_HomePlanet)
test_data['HomePlanet'] = test_data['HomePlanet'].fillna(mode_HomePlanet)

mode_CryoSleep = train_data['CryoSleep'].mode()[0]
train_data['CryoSleep'] = train_data['CryoSleep'].fillna(mode_CryoSleep)
test_data['CryoSleep'] = test_data['CryoSleep'].fillna(mode_CryoSleep)

mode_Destination = train_data['Destination'].mode()[0]
train_data['Destination'] = train_data['Destination'].fillna(mode_Destination)
test_data['Destination'] = test_data['Destination'].fillna(mode_Destination)

mode_VIP = train_data['VIP'].mode()[0]
train_data['VIP'] = train_data['VIP'].fillna(mode_VIP)
test_data['VIP'] = test_data['VIP'].fillna(mode_VIP)

mode_Cabin = train_data['Cabin'].mode()[0]
train_data['Cabin'] = train_data['Cabin'].fillna(mode_Cabin)
test_data['Cabin'] = test_data['Cabin'].fillna(mode_Cabin)

train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
test_data = pd.get_dummies(test_data, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])

X = train_data.drop(['Name', 'Cabin', 'PassengerId', 'Transported'], axis = 1)
y = train_data['Transported']

submission_ids = test_data['PassengerId']
test_data = test_data.drop(['Name', 'Cabin', 'PassengerId',], axis=1)
X, test_data = X.align(test_data, join='inner', axis=1)

final_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,  random_state=42)
final_model.fit(X, y)
test_predictions = final_model.predict(test_data)

submission_data = pd.DataFrame({
    'PassengerId' : submission_ids,
    'Transported' :
test_predictions.astype(bool)
})

submission_data.to_csv('submission.csv', index = False)

print("Submission file created successfully")







