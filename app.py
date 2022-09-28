
from flask import Flask, render_template,request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack




app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Item_Weight = float(request.form['Item_Weight'])
            Item_Fat_Content = str(request.form['Item_Fat_Content'])
            Item_Visibility = float(request.form['Item_Visibility'])
            Item_Type = str(request.form['Item_Type'])
            Item_MRP = float(request.form['Item_MRP'])
            Outlet_Size = str(request.form['Outlet_Size'])
            Outlet_Location_Type = str(request.form['Outlet_Location_Type'])
            Outlet_Type = str(request.form['Outlet_Type'])
            Outlet_Age = int(request.form['Outlet_Age'])

            data = {
                'Item_Weight': Item_Weight,
                'Item_Fat_Content': Item_Fat_Content,
                'Item_Visibility': Item_Visibility,
                'Item_Type': Item_Type,
                'Item_MRP': Item_MRP,
                'Outlet_Size': Outlet_Size,
                'Outlet_Location_Type': Outlet_Location_Type,
                'Outlet_Type': Outlet_Type,
                'Outlet_Age': Outlet_Age

            }

            df = pd.DataFrame(data, index=[0])

            quali = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
            quant = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']

            ohe = pickle.load(open('ohe.pkl', 'rb'))
            print(ohe)
            sc = pickle.load(open('sc.pkl', 'rb'))
            print(sc)
            dtregressor = pickle.load(open('dtregressor.pkl', 'rb'))
            print(dtregressor)

            encoded_data = ohe.transform(df[quali])

            scaled_data = sc.transform(df[quant])

            data_point = hstack((encoded_data, scaled_data)).tocsr()

            prediction=dtregressor.predict(data_point)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=prediction[0])
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app