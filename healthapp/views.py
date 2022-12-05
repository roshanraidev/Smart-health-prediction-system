from this import d
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import datetime
from .forms import DoctorForm
from .models import *
from django.contrib.auth import authenticate, login, logout
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.svm import SVC
from django.http import HttpResponse, HttpResponseRedirect
# Create your views here.
from pickle import encode_long
from scipy.stats import mode
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def Home(request):
    return render(request,'carousel.html')

def Admin_Home(request):
    dis = Search_Data.objects.all()
    pat = Patient.objects.all()
    doc = Doctor.objects.all()

    d = {'dis':dis.count(),'pat':pat.count(),'doc':doc.count()}
    return render(request,'admin_home.html',d)

@login_required(login_url="login")
def assign_status(request,pid):
    doctor = Doctor.objects.get(id=pid)
    if doctor.status == 1:
        doctor.status = 2
        messages.success(request, 'Selected doctor are successfully withdraw his approval.')
    else:
        doctor.status = 1
        messages.success(request, 'Selected doctor is successfully approved.')
    doctor.save()
    return redirect('view_doctor')

@login_required(login_url="login")
def User_Home(request):
    return render(request,'patient_home.html')

@login_required(login_url="login")
def Doctor_Home(request):
    return render(request,'doctor_home.html')

def Login_User(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        sign = ""
        if user:
            try:
                sign = Patient.objects.get(user=user)
            except:
                pass
            if sign:
                login(request, user)
                error = "pat1"
            else:
                pure=False
                try:
                    pure = Doctor.objects.get(status=1,user=user)
                except:
                    pass
                if pure:
                    login(request, user)
                    error = "pat2"
                else:
                    login(request, user)
                    error="notmember"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'login.html', d)

def Login_admin(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        if user.is_staff:
            login(request, user)
            error="pat"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'admin_login.html', d)

def Signup_User(request):
    error = ""
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        u = request.POST['uname']
        e = request.POST['email']
        p = request.POST['pwd']
        d = request.POST['dob']
        con = request.POST['contact']
        add = request.POST['add']
        type = request.POST['type']
        im = request.FILES['image']
        dat = datetime.date.today()
        user = User.objects.create_user(email=e, username=u, password=p, first_name=f,last_name=l)
        if type == "Patient":
            Patient.objects.create(user=user,contact=con,address=add,image=im,dob=d)
        else:
            Doctor.objects.create(dob=d,image=im,user=user,contact=con,address=add,status=2)
        error = "create"
    d = {'error':error}
    return render(request,'register.html',d)
def Logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url="login")
def Change_Password(request):
    sign = 0
    user = User.objects.get(username=request.user.username)
    error = ""
    if not request.user.is_staff:
        try:
            sign = Patient.objects.get(user=user)
            if sign:
                error = "pat"
        except:
            sign = Doctor.objects.get(user=user)
    terror = ""
    if request.method=="POST":
        n = request.POST['pwd1']
        c = request.POST['pwd2']
        o = request.POST['pwd3']
        if c == n:
            u = User.objects.get(username__exact=request.user.username)
            u.set_password(n)
            u.save()
            terror = "yes"
        else:
            terror = "not"
    d = {'error':error,'terror':terror,'data':sign}
    return render(request,'change_password.html',d)

@login_required(login_url="login")
def add_doctor(request,pid=None):
    doctor = None
    if pid:
        doctor = Doctor.objects.get(id=pid)
    if request.method == "POST":
        form = DoctorForm(request.POST, request.FILES, instance = doctor)
        if form.is_valid():
            new_doc = form.save()
            new_doc.status = 1
            if not pid:
                user = User.objects.create_user(password=request.POST['password'], username=request.POST['username'], first_name=request.POST['first_name'], last_name=request.POST['last_name'])
                new_doc.user = user
            new_doc.save()
            return redirect('view_doctor')
    d = {"doctor": doctor}
    return render(request, 'add_doctor.html', d)

@login_required(login_url="login")
def predict_desease(request, pred, accuracy):
    doctor = Doctor.objects.filter(address__icontains=Patient.objects.get(user=request.user).address)
    d = {'pred': pred, 'accuracy':accuracy, 'doctor':doctor}
    return render(request, 'predict_disease.html',d)

@login_required(login_url="login")
def view_search_pat(request):
    doc = None
    try:
        doc = Doctor.objects.get(user=request.user)
        data = Search_Data.objects.filter(patient__address__icontains=doc.address).order_by('-id')
    except:
        try:
            doc = Patient.objects.get(user=request.user)
            data = Search_Data.objects.filter(patient=doc).order_by('-id')
        except:
            data = Search_Data.objects.all().order_by('-id')
    return render(request,'view_search_pat.html',{'data':data})

@login_required(login_url="login")
def delete_doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    doc.delete()
    return redirect('view_doctor')


@login_required(login_url="login")
def delete_patient(request,pid):
    doc = Patient.objects.get(id=pid)
    doc.delete()
    return redirect('view_patient')

@login_required(login_url="login")
def delete_searched(request,pid):
    doc = Search_Data.objects.get(id=pid)
    doc.delete()
    return redirect('view_search_pat')

@login_required(login_url="login")
def View_Doctor(request):
    doc = Doctor.objects.all()
    d = {'doc':doc}
    return render(request,'view_doctor.html',d)

@login_required(login_url="login")
def View_Patient(request):
    patient = Patient.objects.all()
    d = {'patient':patient}
    return render(request,'view_patient.html',d)


@login_required(login_url="login")
def View_My_Detail(request):
    terror = ""
    user = User.objects.get(id=request.user.id)
    error = ""
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    d = {'error': error,'pro':sign}
    return render(request,'profile_doctor.html',d)

@login_required(login_url="login")
def Edit_Doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    error = ""
    # type = Type.objects.all()
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        cat = request.POST['type']
        try:
            im = request.FILES['image']
            doc.image=im
            doc.save()
        except:
            pass
        dat = datetime.date.today()
        doc.user.first_name = f
        doc.user.last_name = l
        doc.user.email = e
        doc.contact = con
        doc.category = cat
        doc.address = add
        doc.user.save()
        doc.save()
        error = "create"
    d = {'error':error,'doc':doc,'type':type}
    return render(request,'edit_doctor.html',d)

@login_required(login_url="login")
def Edit_My_deatail(request):
    terror = ""
    print("Hii welvome")
    user = User.objects.get(id=request.user.id)
    error = ""
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        try:
            im = request.FILES['image']
            sign.image = im
            sign.save()
        except:
            pass
        to1 = datetime.date.today()
        sign.user.first_name = f
        sign.user.last_name = l
        sign.user.email = e
        sign.contact = con
        if error != "pat":
            cat = request.POST['type']
            sign.category = cat
            sign.save()
        sign.address = add
        sign.user.save()
        sign.save()
        terror = "create"
    d = {'error':error,'terror':terror,'doc':sign}
    return render(request,'edit_profile.html',d)

def add_genralhealth(request):
    predictiondata = None
    deseaseli = []
    if request.method=="POST":
        for i,j in request.POST.items():
            if "csrfmiddlewaretoken" != i:
                deseaseli.append(i)
        # training.csv
        DATA_PATH = Admin_Helath_CSV.objects.get(id=2)
        data = pd.read_csv(DATA_PATH.csv_file).dropna(axis = 1)
        data = data.reindex(labels=data.columns,axis=1)

        # Checking whether the dataset is balanced or not
        disease_counts = data["prognosis"].value_counts()
        temp_df = pd.DataFrame({
            "Disease": disease_counts.index,
            "Counts": disease_counts.values
        })
       

        # Encoding the target value into numerical
        # value using LabelEncoder
        encoder = LabelEncoder()
        data["prognosis"] = encoder.fit_transform(data["prognosis"])
        X = data.iloc[:,:-1]
        y = data.iloc[:, -1]
     
         #spliting model
        X_train, X_test, y_train, y_test = train_test_split(
            X,y,train_size= 0.80, shuffle= True, random_state=24)
         #training model
        final_svm_model = SVC()
        final_rf_model = RandomForestClassifier()
        final_svm_model.fit(X, y)
        final_rf_model.fit(X, y)
         # building model 
        svm_preds = final_svm_model.predict(X_test)
        rf_preds = final_rf_model.predict(X_test)
    

        final_preds = [mode([i,j])[0][0] for i,j
                     in zip(svm_preds, rf_preds)]

        print(f"Accuracy on Test dataset by the combined model\
        : {accuracy_score(y_test, final_preds)*100}")

        cf_matrix = confusion_matrix(y_test, final_preds)
        
        symptoms = X.columns.values
        symptom_index = {}
        for index, value in enumerate(symptoms):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            symptom_index[symptom] = index
   
        
        data_dict = {
            "symptom_index":symptom_index,
            "predictions_classes":encoder.classes_
        }
      
        def predictDisease(symptoms):
            # print("All Symptoms = ", symptoms)
            # symptoms = symptoms.split(",")
            
            # # creating input data for the models
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in symptoms:
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1
                
            # reshaping the input data and converting it
            # into suitable format for model predictions
            input_data = np.array(input_data).reshape(1,-1)
            
            # generating individual outputs
            rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
            svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
            
            # making final prediction by taking mode of all predictions
            final_prediction = mode([ rf_prediction, svm_prediction])[0][0]
            predictions = {
                # "GaussianNB Prediction": nb_prediction,
                "RandomForestClassifier Prediction": rf_prediction,
                "SVC Prediction": svm_prediction,
                "Final Prediction":final_prediction
            }
            return predictions

        # Testing the function
        predictiondata = predictDisease(deseaseli)
        patient = Patient.objects.get(user=request.user)
        Search_Data.objects.create(patient=patient, prediction_accuracy=round(accuracy_score(y_test, final_preds)*100,2), result=predictiondata["Final Prediction"], values_list=deseaseli, predict_for="General Health Prediction")

        # print(deseaseli)
    alldisease = ['Itching','Skin Rash','Nodal Skin Eruptions','Continuous Sneezing','Shivering','Chills','Joint Pain',	'Stomach Pain','Acidity','Ulcers On Tongue','Muscle Wasting','Vomiting','Burning Micturition','Spotting Urination','Fatigue','Weight Gain','Anxiety','Cold Hands And Feets','Mood Swings','Weight Loss','Restlessness','Lethargy','Patches In Throat','Irregular Sugar Level','Cough','High Fever','Sunken Eyes','Breathlessness','Sweating','Dehydration',	'Indigestion','Headache','Yellowish Skin','Dark Urine','Nausea','Loss Of Appetite','Pain Behind The Eyes','Back Pain','Constipation','Abdominal Pain','Diarrhoea','Mild Fever','Yellow Urine','Yellowing Of Eyes','Acute Liver Failure','Fluid Overload','Swelling Of Stomach','Swelled Lymph Nodes','Malaise','Blurred And Distorted Vision','Phlegm','Throat Irritation','Redness Of Eyes','Sinus Pressure','Runny Nose','Congestion','Chest Pain','Weakness In Limbs','Fast Heart Rate',	'Pain During Bowel Movements','Pain In Anal Region','Bloody Stool','Irritation In Anus','Neck Pain','Dizziness','Cramps','Bruising','Obesity','Swollen Legs','Swollen Blood Vessels','Puffy Face And Eyes','Enlarged Thyroid','Brittle Nails','Swollen Extremeties','Excessive Hunger','Extra Marital Contacts','Drying And Tingling Lips','Slurred Speech','Knee Pain','Hip Joint Pain','Muscle Weakness','Stiff Neck','Swelling Joints','Movement Stiffness','Spinning Movements','Loss Of Balance','Unsteadiness','Weakness Of One Body Side','Loss Of Smell','Bladder Discomfort','Continuous Feel Of Urine','Passage Of Gases','Internal Itching','Toxic Look (Typhos)',	'Depression','Irritability','Muscle Pain','Altered Sensorium','Red Spots Over Body','Belly Pain','Abnormal Menstruation','Dischromic Patches','Watering From Eyes','Increased Appetite','Polyuria','Family History','Mucoid Sputum','Rusty Sputum','Lack Of Concentration',	'Visual Disturbances','Receiving Blood Transfusion','Receiving Unsterile Injections','Coma','Stomach Bleeding',	'Distention Of Abdomen','History Of Alcohol Consumption','Fluid Overload','Blood In Sputum','Prominent Veins On Calf','Palpitations','Painful Walking','Pus Filled Pimples', 'Blackheads','Scurring','Skin Peeling','Silver Like Dusting','Small Dents In Nails','Inflammatory Nails','Blister','Red Sore Around Nose','Yellow Crust Ooze','Prognosis']
    return render(request,'add_genralhealth.html', {'alldisease':alldisease, 'predictiondata':predictiondata})
