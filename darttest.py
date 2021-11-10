import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from math import exp

st.header("Glucose Prediction")
# create columns in streamlit for inputs

# create number inputs for the class in streamlit
carbs = st.sidebar.number_input("Carbs", value=10)
currentbg = st.sidebar.number_input("Current BG", value=100)
targetbg = st.sidebar.number_input("Target BG", value=90)
intervals =st.sidebar.number_input("Intervals", value=1)
time = np.arange(0,st.sidebar.number_input("Time", value=4)*60,intervals)
inscarbRatio = st.sidebar.number_input("Insulin Carb Ratio", value=10)
correctionRatio = st.sidebar.number_input("Correction Ratio", value=75)
timetotarget = st.sidebar.number_input("Time to Target", value=120)
timetopeak = st.sidebar.number_input("Time to Peak", value=60)



# texp = st.sidebar.number_input("Time Exponent", value=10)
# bolus = st.sidebar.number_input("Bolus", value=10)
list =[]

class PredData:
    def __init__(self, vtime, value):
        self.vtime = vtime
        self.value = value



class GlucosePredict:
    def __init__(self, carbs, inscarbRatio,  correctionRatio, timetotarget,timetopeak,currentbg,targetbg,time,):
        self.carbs = carbs
        self.inscarbRatio = inscarbRatio        
        self.correctionRatio = correctionRatio
        self.timetotarget = timetotarget
        self.timetopeak = timetopeak
        self.currentbg = currentbg
        self.targetbg = targetbg
        self.time = time
        # texp = exp(1)
        # self.bolus

    def calculatebolus(self,currentbg,carbs):
        bolus = ((currentbg - targetbg) / correctionRatio) + (carbs / inscarbRatio)
        # st.write("Bolus")
        # st.write(bolus)
        return bolus

    def getcorrection(self,time):
        top = (-pow((exp(1)*time)/timetotarget,2))/2
        self.correction = 1-exp(top)
        # st.write("correction")
        # st.write(self.correction)
        return self.correction

    def getfall(self,time,currentbg,carbs):
           self.fall = -self.getcorrection(time)*correctionRatio*self.calculatebolus(currentbg,carbs)
        #    st.write(self.fall)
           return self.fall

    def getcarbs(self,time):
        bot = (-pow((exp(1)*time)/timetopeak,2))/2
        self.carbs = 1 - exp(bot)
        # st.write(self.carbs)
        return self.carbs

    def getrise(self,time,currentbg,carbs,):
        self.rise = self.getcarbs(time)*carbs*correctionRatio/inscarbRatio
        # st.write(self.rise)
        return self.rise

    def getprediction(self,time,currentbg,carbs):
        self.prediction = self.getfall(time,currentbg,carbs) + self.getrise(time,currentbg,carbs)+currentbg
        # st.write("Prediction")
        # st.write(self.prediction)
        # return PredData(time,self.prediction)
        return self.prediction

    def getpredictions(self,currentbg,carbs,time):
         self.predictions = []
         for i in range(0,len(time)):
                self.predictions.append(self.getprediction(time[i],currentbg,carbs))
         st.write(self.predictions.__len__())
         list.append(self.predictions)
         return self.predictions

    def getpredictionsdf(self,currentbg,carbs,time):
       self.predictionsdf = pd.DataFrame(self.getpredictions(time,currentbg,carbs))
       st.write(self.predictionsdf)
       return self.predictionsdf

    def getpredictionsdfplot(self,currentbg,carbs):
        self.predictionsdfplot = getpredictionsdf(currentbg,carbs)
        st.write(self.predictionsdfplot)
        return self.predictionsdfplot

glucosepredict = GlucosePredict(carbs,inscarbRatio,correctionRatio,timetotarget,timetopeak,currentbg,targetbg,time)
glucosepredict.getpredictions(currentbg,carbs,time)   

# create a new instance of the class using the inputs

# run the prediction on a button click
# if st.button("Predict"):
#     # glucosepredict.calculatebolus()
#     # glucosepredict.getcorrection(time)
#     # glucosepredict.getfall(time,currentbg,carbs)
#     # glucosepredict.getcarbs(time)
#     # glucosepredict.getrise(time,currentbg,carbs)
#     # glucosepredict.getprediction(time,currentbg,carbs)
#     glucosepredict.getpredictions(currentbg,carbs,time)
#     # glucosepredict.getpredictionsdf(currentbg,carbs,time)
#     # glucosepredict.getpredictionsdfplot(currentbg,carbs)


df = pd.DataFrame(list)
# transpose the dataframe


# if list is not empty, print the dataframe
if list.__len__() > 0:
    st.write(df)
    df = df.T
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, ax=ax)
    # add axis labels
    ax.set(xlabel='Time (min)', ylabel='Glucose (mg/dL)')
    # add title
    ax.set_title('Glucose Prediction')
    # add gridlines
    ax.grid(True)
    # display the y-axis to 40 mg/dL
    ax.set_ylim(40, )
    #  add color bands above and below target
    ax.axhline(y=targetbg, color='g', linestyle='-')
    # add lines +-20% of the target
    ax.axhline(y=targetbg*1.2, color='r', linestyle='--')
    ax.axhline(y=targetbg*0.8, color='r', linestyle='--')


    st.pyplot(fig)
    # fig = px.line(df, x=df.index, y=df.columns[0])
    # change the color of the background
    # fig.update_layout(plot_bgcolor='#f5f5f5')
       
    # add axis labels
    # fig.update_layout(xaxis_title="Time (minutes)", yaxis_title="Glucose (mg/dL)")
    # st.plotly_chart(fig)




# print the list.values
# st.write(list)

# create an api to run the prediction
# create a function to run the prediction









# create a UI element to display the prediction
# st.write(glucosepredict.getpredictions(currentbg=10,carbs=10,time=np.arange(0,4,0.1)))

# create a list of time values up to 4 hours in steps of 1 minute
# time = np.arange(0,4,0.1)

# help debug the code


