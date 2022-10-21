import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import quad
from math import* 

#TITULOS
st.sidebar.title('Laboratorio 3 - Señales y sistemas')
st.sidebar.subheader("Jessir Florez - Mateo Muñoz - Dylan Abuchaibe")

#SELECCIONADOR DE SEÑALES
opcion = st.sidebar.selectbox(
     'Escoja la señal que desea generar a continuación:',
     ("Exponencial","Senoidal rectificada"))

st.sidebar.write('Has seleccionado la función tipo :', opcion)

n=st.number_input("Ingrese el numero de armonicos",step=5,min_value=0,max_value=50,value=5)
x=np.arange(-np.pi,np.pi,0.001) 

def funcion_grafica(x,y,sum):
    fig,ax=plt.subplots()
    ax.plot(x,sum,'g')
    plt.plot(x,y,'r--')
    ax.set_title("Serie de fourier con " + str(n) + " armonicos")
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.grid(True)
    st.pyplot(fig)

def Fourier(y,fun):
    
    fc=lambda x: np.exp(x)*cos(i*x)  
    fs=lambda x: np.exp(x)*sin(i*x)

    An=[] 
    Bn=[]

    sum=0

    for i in range(n):
        an=quad(fc,-np.pi,np.pi)[0]*(1.0/np.pi)
        An.append(an)

    for i in range(n):

        bn=quad(fs,-np.pi,np.pi)[0]*(1.0/np.pi)
        Bn.append(bn) 

    for i in range(n):
        if i==0.0:
            sum=sum+An[i]/2
            
        else:
            sum=sum+(An[i]*np.cos(i*x)+Bn[i]*np.sin(i*x))


if opcion == "Exponencial":

    y=np.exp(x) 

    fc=lambda x: np.exp(x)*cos(i*x)  
    fs=lambda x: np.exp(x)*sin(i*x)

    An=[] 
    Bn=[]

    sum=0

    for i in range(n):
        an=quad(fc,-np.pi,np.pi)[0]*(1.0/np.pi)
        An.append(an)

    for i in range(n):

        bn=quad(fs,-np.pi,np.pi)[0]*(1.0/np.pi)
        Bn.append(bn) 

    for i in range(n):
        if i==0.0:
            sum=sum+An[i]/2
            
        else:
            sum=sum+(An[i]*np.cos(i*x)+Bn[i]*np.sin(i*x))
        
    funcion_grafica(x,y,sum)  

if opcion == "Senoidal rectificada":
    
    st.title("Función Seno")

    frecuencia = st.number_input("Por favor ingrese el valor de la frecuencia f: ",step=1,min_value=1,max_value=10)
    amplitude = st.number_input("Por favor ingrese el valor de la amplitud A: ",step=1,min_value=-10,max_value=10,value=1)
    
    paso=(1/(300*frecuencia))
    x=np.arange(0,2*np.pi,paso)
    y=abs(np.sin(2*np.pi*frecuencia*x))
    y=y*amplitude

    fig,ax=plt.subplots()
    ax.plot(x,y)
    ax.set_title("Función Seno")
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.set_xlim(0,np.pi)
    ax.set_ylim(-10,10)
    ax.grid(True)
    st.pyplot(fig)