import streamlit as st
import numpy as np

from pandas import read_csv
from os.path import basename, exists

import pint

st.set_option('deprecation.showPyplotGlobalUse', False)
def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print("Downloaded",  local)

download('https://github.com/AllenDowney/ModSimPy/raw/master/' +
         'modsim.py')

from modsim import *

download('https://github.com/AllenDowney/ModSim/raw/main/data/' +
         'glucose_insulin.csv')
data = read_csv('glucose_insulin.csv', index_col='time')



st.dataframe(data.head())

I = interpolate(data.insulin)

t_0 = data.index[0]
t_end = data.index[-1]
t_array = linrange(t_0, t_end)

I_array = I(t_array)
#make series of t_array and I_array
I_series = make_series(t_array, I_array)

# show a plot of the data
data.insulin.plot(style='o', color='C2', label='insulin data')
I_series.plot(color='C2', label='interpolation')

decorate(xlabel='Time (min)',
         ylabel='Concentration ($\mu$U/mL)')
# Show this plot
st.pyplot()


G0 = 270
k1 = 0.02
k2 = 0.02
k3 = 1.5e-05
# make sidbar inputs for the parameters
G0 = st.sidebar.slider('G0', 0, 400, 270)


params = G0, k1, k2, k3

def make_system(params, data):
    G0, k1, k2, k3 = params
    
    t_0 = data.index[0]
    t_end = data.index[-1]
    
    Gb = data.glucose[t_0]
    Ib = data.insulin[t_0]
    I = interpolate(data.insulin)
    
    init = State(G=G0, X=0)
    
    return System(init=init, params=params,
                  Gb=Gb, Ib=Ib, I=I,
                  t_0=t_0, t_end=t_end, dt=2)

system = make_system(params, data)

def update_func(t, state, system):
    G, X = state
    G0, k1, k2, k3 = system.params 
    I, Ib, Gb = system.I, system.Ib, system.Gb
    dt = system.dt
        
    dGdt = -k1 * (G - Gb) - X*G
    dXdt = k3 * (I(t) - Ib) - k2 * X
    
    G += dGdt * dt
    X += dXdt * dt

    return State(G=G, X=X)

update_func(system.t_0, system.init, system)
# predict the glucose and insulin concentrations

st.write('sim')
def run_simulation(system, update_func):    
    t_array = linrange(system.t_0, system.t_end, system.dt)
    n = len(t_array)
    
    frame = TimeFrame(index=t_array, 
                      columns=system.init.index)
    frame.iloc[0] = system.init
    
    for i in range(n-1):
        t = t_array[i]
        state = frame.iloc[i]
        frame.iloc[i+1] = update_func(t, state, system)
    
    return frame

results = run_simulation(system, update_func)
# Show the results
st.dataframe(results.head())

# create 2 columns
col1, col2 = st.columns(2)

data.glucose.plot(style='o', alpha=0.5, label='glucose data')
results.G.plot(style='-', color='C0', label='simulation')

decorate(xlabel='Time (min)',
         ylabel='Concentration (mg/dL)')
# Show this plot
col1.pyplot()

results.X.plot(color='C1', label='remote insulin')

decorate(xlabel='Time (min)', 
         ylabel='Concentration (arbitrary units)')
# Show this plot
col2.pyplot()

def slope_func(t, state, system):
    G, X = state
    G0, k1, k2, k3 = system.params 
    I, Ib, Gb = system.I, system.Ib, system.Gb
        
    dGdt = -k1 * (G - Gb) - X*G
    dXdt = k3 * (I(t) - Ib) - k2 * X
    
    return dGdt, dXdt

results2, details = run_solve_ivp(system, slope_func,
                                  t_eval=results.index)
                            
details.success
details.message

# show the results
st.dataframe(results2.head())
results.G.plot(style='--', label='simulation')
results2.G.plot(style='-', label='solve ivp')

decorate(xlabel='Time (min)',
         ylabel='Concentration (mg/dL)')
# Show this plot
st.pyplot()
results.X.plot(style='--', label='simulation')
results2.X.plot(style='-', label='solve ivp')

decorate(xlabel='Time (min)', 
         ylabel='Concentration (arbitrary units)')
# Show this plot
st.pyplot()

st.header('Insulin')

download('https://raw.githubusercontent.com/AllenDowney/' +
         'ModSim/main/data/glucose_insulin.csv')
data = read_csv('glucose_insulin.csv', index_col='time')
# st.dataframe(data.head())

I = interpolate(data.insulin)

t_0 = data.index[0]
t_end = data.index[-1]
t_array = linrange(t_0, t_end)

I_array = I(t_array)
#make series of t_array and I_array
I_series = make_series(t_array, I_array)
I0 = 360
k = 0.25
gamma = 0.004
G_T = 80

params = I0, k, gamma, G_T

def make_isystem(params, data):
    I0, k, gamma, G_T = params
    
    t_0 = data.index[0]
    t_end = data.index[-1]
    
    Gb = data.glucose[t_0]
    Ib = data.insulin[t_0]
    I = interpolate(data.insulin)
    
    init = State(G=G_T, X=0)
    
    return System(init=init, params=params,
                  Gb=Gb, Ib=Ib, I=I,t_0=t_0, t_end=t_end, dt=2)

system = make_isystem(params, data)

def update_func(t, state, system):
    G, X = state
    I0, k, gamma, G_T = system.params
    I, Ib, Gb = system.I, system.Ib, system.Gb
    dt = system.dt

    # dGdt = -k * (G - Gb) - X*G
    # dXdt = gamma * (I(t) - Ib) - k * X
    dGdt = -k + (gamma * (G - Gb))
    dXdt = gamma * (I(t) - Ib)

    G += dGdt * dt
    X += dXdt * dt

    return State(G=G, X=X)



update_func(system.t_0, system.init, system)

def run_simulation(system, update_func):    
    t_array = linrange(system.t_0, system.t_end, system.dt)
    n = len(t_array)
    
    frame = TimeFrame(index=t_array, 
                      columns=system.init.index)
    frame.iloc[0] = system.init
    
    for i in range(n-1):
        t = t_array[i]
        state = frame.iloc[i]
        frame.iloc[i+1] = update_func(t, state, system)
    
    return frame

results = run_simulation(system, update_func)
st.dataframe(results.head())

data.insulin.plot(style='o', alpha=0.5, label='insulin data')
results.X.plot(style='-', color='C0', label='simulation')

decorate(xlabel='Time (min)',
         ylabel='Concentration (arbitrary units)')
# Show this plot
st.pyplot()

