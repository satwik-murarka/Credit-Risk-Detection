from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
clf = pickle.load(open("models/clf.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = request.form.get('age')
    np_age = np.log(int(age))
    duration = request.form.get('duration')
    np_duration = np.log(int(duration))
    credit = request.form.get('credit')
    np_credit = np.log(int(credit))
    gender = request.form.get('gender')
    checking_account = request.form.get('checking_account')
    job = request.form.get('job')
    x = pd.DataFrame(columns = ['Age','Job','Credit amount',
                                'Duration','Sex_female','Sex_male',
                                'Checking account_No Account','Checking account_little','Checking account_moderate',
                                'Checking account_rich'])
    if(gender=="Male"):
        sex_male = 1
        sex_female = 0
    else:
        sex_male = 0
        sex_female = 1
    if(checking_account=="No Account"):
        ch_no = 1
        ch_l = 0
        ch_m = 0
        ch_r = 0
    elif(checking_account=="Little"):
        ch_no = 0
        ch_l = 1
        ch_m = 0
        ch_r = 0
    elif(checking_account=="Moderate"):
        ch_no = 0
        ch_l = 0
        ch_m = 1
        ch_r = 0
    else:
        ch_no = 0
        ch_l = 0
        ch_m = 0
        ch_r = 1
    new_row = {'Age': np_age,'Job': int(job),'Credit amount': np_credit,
                                'Duration': np_duration,'Sex_female': sex_female,'Sex_male': sex_male,
                                'Checking account_No Account':ch_no,'Checking account_little':ch_l,'Checking account_moderate':ch_m,
                                'Checking account_rich':ch_r}
    x = x.append(new_row, ignore_index=True)

    prediction = clf.predict(x)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction,age=age,duration=duration,credit=credit,checking_account=checking_account,
                           gender=gender,job=job)


if __name__ == "__main__":
    app.run(debug=True)