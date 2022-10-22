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

n=st.sidebar.number_input("Ingrese el numero de armonicos",step=1,min_value=2,max_value=100,value=2)
x=np.arange(-np.pi,np.pi,0.001) 

#DECLARACION DE FUNCIONES
def funcion_grafica(x,y,sum):
    st.title("Función "+ str(opcion))
    fig,ax=plt.subplots()
    ax.plot(x,sum,'g')
    plt.plot(x,y,'r--')
    ax.set_title("Serie de fourier con " + str(n) + " armonicos")
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.grid(True)
    st.pyplot(fig)

def Fourier(y,fun):

    An=[] 
    Bn=[]

    fun_cos=lambda x: np.exp(x)*cos(i*x)  
    fun_sen=lambda x: np.exp(x)*sin(i*x)

    sum=0

    for i in range(n):
        an=quad(fun_cos,-np.pi,np.pi)[0]*(1.0/np.pi)
        An.append(an)

    for i in range(n):

        bn=quad(fun_sen,-np.pi,np.pi)[0]*(1.0/np.pi)
        Bn.append(bn) 

    for i in range(n):
        if i==0.0:
            sum=sum+An[i]/2
            
        else:
            sum=sum+(An[i]*np.cos(i*x)+Bn[i]*np.sin(i*x))

    return sum


if opcion == "Exponencial":

    y= np.exp(x) 
    serie=Fourier(y,np.exp)
    funcion_grafica(x,y,serie)  

if opcion == "Senoidal rectificada":
    
    amplitude = st.number_input("Por favor ingrese el valor de la amplitud A: ",step=1,min_value=-10,max_value=10,value=1)
    
    y=abs(np.sin(2*np.pi*x))
    y=y*amplitude

    An=[] 
    Bn=[]

    fun_cos=lambda x: amplitude*abs(np.sin(2*np.pi*x))*cos(i*x)  
    fun_sen=lambda x: amplitude*abs(np.sin(2*np.pi*x))*sin(i*x)

    sum=0

    for i in range(n):
        an=quad(fun_cos,-np.pi,np.pi)[0]*(1.0/np.pi)
        An.append(an)

    for i in range(n):

        bn=quad(fun_sen,-np.pi,np.pi)[0]*(1.0/np.pi)
        Bn.append(bn) 

    for i in range(n):
        if i==0.0:
            sum=sum+An[i]/2
            
        else:
            sum=sum+(An[i]*np.cos(i*x)+Bn[i]*np.sin(i*x))
            
    
    funcion_grafica(x,y,sum)