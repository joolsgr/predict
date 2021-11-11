
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from math import exp
from os.path import basename, exists

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Glucose Prediction")
# create columns in streamlit for inputs
def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print("Downloaded",  local)

download('https://github.com/AllenDowney/ModSimPy/raw/master/' +
         'modsim.py')

from modsim import *
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


# t_0 = time[0]
# t_end = time[-1]
# G0 = carbs
# k1 = 0.02
# k2 = 0.02
# k3 = 1.5e-05
# params = G0, k1, k2, k3

# texp = st.sidebar.number_input("Time Exponent", value=10)
# bolus = st.sidebar.number_input("Bolus", value=10)
list =[]
flist =[]
rlist =[]
list2 =[]
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
    def minimal():
        t_0 = time[0]
        t_end = time[-1]
        G0 = currentbg
        k1 = 0.02
        k2 = 0.02
        k3 = 1.5e-05
        params = G0, k1, k2, k3
        data = pd.read_csv('glucose_insulin.csv', index_col='time')

        def make_system(params,data):
            G0, k1, k2, k3 = params
            t_0 = time[0]
            t_end = time[-1]
            Gb = data.glucose[t_0]
            Ib = data.insulin[t_0]
            I = interpolate(data.insulin)
            init = State(G=G0, X=0)

            return System(init=init, params=params,
                          t_0=t_0, t_end=t_end,
                          Gb=Gb, Ib=Ib, I=I, dt=intervals)

        system = make_system(params,data)

        def update_func(t,state,system):
            G, X = state
            G0, k1, k2, k3 = system.params
            I, Ib, Gb = system.I, system.Ib, system.Gb
            dt = system.dt
            dGdt = -k1 * (G -Gb) - X*G
            dXdt = k3 * (I(t) - Ib) - k2 * X

            # G += dGdt * dt
            # G = getrise(t,G,carbs)
            X += dXdt * dt

            return State(G=G, X=X)

        update_func(system.t_0, system.init, system)

        def run_simulation(system, update_func):
            t_array = linrange(system.t_0, system.t_end, system.dt)
            n = len(t_array)
            frame = TimeFrame(index=t_array, columns=system.init.index)
            frame.iloc[0] = system.init
            for i in range(n-1):
                t = t_array[i]
                state = frame.iloc[i]
                frame.iloc[i+1] = update_func(t, state, system)

            return frame

        results = run_simulation(system, update_func)
        st.dataframe(results.G)
        data.glucose.plot(style='o', alpha=0.5, label='glucose data')
        results.G.plot(style='-', color='C0', label='simulation')

        decorate(xlabel='Time (min)',
         ylabel='Concentration (mg/dL)')
        list2.append(results.G)
# Show this plot
        st.pyplot()
        return results.G
    minimal()
    st.dataframe(list2)

# build q defualt instance of the class as a function
    def build_instance():
        return GlucosePredict(carbs=10,inscarbRatio=10,correctionRatio=75,timetotarget=100,timetopeak=60,currentbg=100,targetbg=90,time=4)



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
        fall=self.getfall(time,currentbg,carbs)
        rise=self.getrise(time,currentbg,carbs)
        self.prediction = self.getfall(time,currentbg,carbs) + self.getrise(time,currentbg,carbs)+currentbg
        # st.write("Prediction")
        # st.write(self.prediction)
        # return PredData(time,self.prediction)
        results = self.prediction
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
list2 = pd.DataFrame(list2)
# transpose the dataframe


# if list is not empty, print the dataframe
if list.__len__() > 0:
    st.write(df)
    st.write(list2)
    df = df.T
    list2 = list2.T
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, ax=ax)
    sns.lineplot(data=list2, ax=ax)
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