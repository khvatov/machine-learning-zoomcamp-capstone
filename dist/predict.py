import pickle
from flask import Flask
from flask import request
from flask import jsonify


input_file = 'model.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# SAMPLE DATA TO SEND TO THE API
#{"quality":"1","pre_screening":"1","ma1":22,"ma2":22,"ma3":22,"ma4":19,"ma5":18,"ma6":14,"exudate1":49.895756,"exudate2":17.775994,"exudate3":5.27092,"exudate4":0.771761,"exudate5":0.018632,"exudate6":0.006864,"exudate7":0.003923,"exudate8":0.003923,"macula_opticdisc_distance":0.486903,"opticdisc_diameter":0.100025,"am_fm_classification":"1"}
#class = 0

#{"quality":"1","pre_screening":"1","ma1":24,"ma2":24,"ma3":22,"ma4":18,"ma5":16,"ma6":13,"exudate1":57.709936,"exudate2":23.799994,"exudate3":3.325423,"exudate4":0.234185,"exudate5":0.003903,"exudate6":0.003903,"exudate7":0.003903,"exudate8":0.003903,"macula_opticdisc_distance":0.520908,"opticdisc_diameter":0.144414,"am_fm_classification":"0"}
#class = 0

#{"quality":"1","pre_screening":"1","ma1":62,"ma2":60,"ma3":59,"ma4":54,"ma5":47,"ma6":33,"exudate1":55.831441,"exudate2":27.993933,"exudate3":12.687485,"exudate4":4.852282,"exudate5":1.393889,"exudate6":0.373252,"exudate7":0.041817,"exudate8":0.007744,"macula_opticdisc_distance":0.530904,"opticdisc_diameter":0.128548,"am_fm_classification":"0"}
#class = 1

#{"quality":"1","pre_screening":"1","ma1":55,"ma2":53,"ma3":53,"ma4":50,"ma5":43,"ma6":31,"exudate1":40.467228,"exudate2":18.445954,"exudate3":9.118901,"exudate4":3.079428,"exudate5":0.840261,"exudate6":0.272434,"exudate7":0.007653,"exudate8":0.001531,"macula_opticdisc_distance":0.483284,"opticdisc_diameter":0.11479,"am_fm_classification":"0"}
#class = 0

#{"quality":"1","pre_screening":"1","ma1":44,"ma2":44,"ma3":44,"ma4":41,"ma5":39,"ma6":27,"exudate1":18.026254,"exudate2":8.570709,"exudate3":0.410381,"exudate4":0.0,"exudate5":0.0,"exudate6":0.0,"exudate7":0.0,"exudate8":0.0,"macula_opticdisc_distance":0.475935,"opticdisc_diameter":0.123572,"am_fm_classification":"0"}
#class = 1


app = Flask('diabetic_retinopathy')  # create an app

@app.route('/predict', methods=["POST"])
def predict():
    patient = request.get_json()
    X=dv.transform([patient])
    y_pred = model.predict_proba(X)[0,1]
    print(f"Patient's probability to be diagnosed with the diabetic retinopathy: {y_pred:.3f}")  # 0.000
    will_be_diagnosed:bool = (y_pred >= 0.45)

    result = { 
        "diagnosis_probability": float(y_pred),
        "will_get_diagnosed": bool(will_be_diagnosed)
        }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)  # run the app


#launched gunicorn 
#gunicorn --bind 0.0.0.0:9696 predict:app 